import os
from io import BytesIO
from typing import Optional

import flammkuchen as fl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QListWidget, QPushButton, QVBoxLayout
from tifffile import imread, imwrite

from config_utils import load_config, cfg_get


def select_image_dirs(base_path: str):
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"base_path not found: {base_path}")

    subdirs = [
        d for d in sorted(os.listdir(base_path))
        if os.path.isdir(os.path.join(base_path, d))
    ]

    dlg = QDialog()
    dlg.setWindowTitle("Select image_dirs")
    layout = QVBoxLayout(dlg)

    layout.addWidget(QLabel(
        f"Base: {base_path}\n"
        "Select folders to process (Ctrl/Shift for multi-select).\n"
        "No selection = all folders."
    ))

    lw = QListWidget()
    lw.setSelectionMode(QListWidget.ExtendedSelection)
    lw.addItems(subdirs)
    layout.addWidget(lw)

    btn_ok = QPushButton("OK")
    btn_cancel = QPushButton("Cancel")
    layout.addWidget(btn_ok)
    layout.addWidget(btn_cancel)

    selected = []

    def on_ok():
        nonlocal selected
        selected = [item.text() for item in lw.selectedItems()]
        dlg.accept()

    btn_ok.clicked.connect(on_ok)
    btn_cancel.clicked.connect(dlg.reject)

    if dlg.exec_() != QDialog.Accepted:
        return []

    return selected if selected else subdirs


def load_swc_as_dataframe(swc_path: str) -> pd.DataFrame:
    return pd.read_csv(
        swc_path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["id", "type", "x", "y", "z", "radius", "parent"],
        engine="python",
    )


STD_COLS = ["id", "type", "x", "y", "z", "radius", "parent"]


def renumber_swc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["parent"] = pd.to_numeric(df["parent"], errors="coerce").fillna(-1).astype("Int64")

    df = df.sort_values("id").reset_index(drop=True)
    old_ids = df["id"].astype(int).tolist()
    id_map = {old_id: new_id for new_id, old_id in enumerate(old_ids, start=1)}

    df["id"] = df["id"].map(lambda v: id_map.get(int(v), None)).astype("Int64")
    df["parent"] = df["parent"].map(lambda p: id_map.get(int(p), -1) if int(p) > 0 else -1).astype(int)

    return df[STD_COLS]


def save_swc_dataframe(df: pd.DataFrame, path: str):
    df.to_csv(path, sep=" ", header=False, index=False)


def save_overlay_stack(
    save_path: str,
    mask: np.ndarray,
    base_name: str,
    alpha: float = 0.4,
    out_path: Optional[str] = None
):
    tif_path = os.path.join(save_path, f"{base_name}.tif")
    image = imread(tif_path)

    overlay_stack = []
    for z in range(image.shape[0]):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.imshow(image[z], cmap="gray")
        ax.imshow(mask[z].astype(np.uint8), cmap="Reds", alpha=float(alpha))
        ax.axis("off")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        overlay_stack.append(np.array(Image.open(buf).convert("RGB")))

    if out_path is None:
        out_path = os.path.join(save_path, f"{base_name}_overlay.tif")

    imwrite(out_path, np.stack(overlay_stack))


def image_split(
    image_id: str,
    image_path: str,
    mask_path: str,
    swc_path: str,
    tile_size: int,
    stride: int,
    save_path: str,
    overlay_alpha: float = 0.4,
):
    os.makedirs(save_path, exist_ok=True)

    image = imread(image_path)
    mask = fl.load(mask_path)["mask"]
    mask = np.transpose(mask, (0, 2, 1))
    swc_df = load_swc_as_dataframe(swc_path)

    z_dim, h, w = mask.shape
    count = 0
    metadata = []

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile_stack = []
            tile_mask_stack = []
            tile_mask_for_overlay_stack = []
            tile_swc_all = []

            x_min, x_max = x, x + tile_size
            y_min, y_max = y, y + tile_size

            for z in range(z_dim):
                tile = image[z, y:y + tile_size, x:x + tile_size]
                tile_mask = mask[z, y:y + tile_size, x:x + tile_size]

                tile_stack.append(tile)
                tile_mask_for_overlay_stack.append(tile_mask)
                tile_mask_stack.append(np.transpose(tile_mask))

                tile_swc_df = swc_df[
                    (swc_df["z"].round().astype(int) == z)
                    & (swc_df["x"] >= x_min) & (swc_df["x"] < x_max)
                    & (swc_df["y"] >= y_min) & (swc_df["y"] < y_max)
                ].copy()

                tile_swc_df["x"] -= x_min
                tile_swc_df["y"] -= y_min
                tile_swc_all.append(tile_swc_df)

            tile_stack = np.stack(tile_stack)
            tile_mask_stack = np.stack(tile_mask_stack)
            tile_mask_for_overlay_stack = np.stack(tile_mask_for_overlay_stack)

            base_name = f"{image_id}_t{count:04d}"
            imwrite(os.path.join(save_path, f"{base_name}.tif"), tile_stack)
            fl.save(os.path.join(save_path, f"{base_name}.mask"), {"mask": tile_mask_stack})

            save_overlay_stack(
                save_path,
                tile_mask_for_overlay_stack,
                base_name,
                alpha=float(overlay_alpha),
            )

            merged_swc = pd.concat(tile_swc_all, ignore_index=True)
            if not merged_swc.empty:
                merged_swc = renumber_swc(merged_swc)
            save_swc_dataframe(merged_swc, os.path.join(save_path, f"{base_name}.swc"))

            metadata.append({"tile": base_name, "x_start": x, "y_start": y})
            count += 1

    pd.DataFrame(metadata).to_csv(os.path.join(save_path, "tile_metadata.csv"), index=False)


if __name__ == "__main__":
    cfg = load_config("config.yaml")

    # Required
    base_path = cfg_get(cfg, ["base_path"], None)
    if not base_path:
        raise KeyError("config.yaml must define: base_path")

    # Optional (defaults)
    tile_size = int(cfg_get(cfg, ["image_split", "tile", "size"], 256))
    stride = int(cfg_get(cfg, ["image_split", "tile", "stride"], 256))
    overlay_alpha = float(cfg_get(cfg, ["image_split", "overlay", "alpha"], 0.4))

    app = QApplication.instance() or QApplication([])

    image_dirs = select_image_dirs(base_path)
    if not image_dirs:
        raise SystemExit(0)

    for image_id in image_dirs:
        folder_path = os.path.join(base_path, image_id)
        image_path = os.path.join(folder_path, f"{image_id}.tif")
        mask_path = os.path.join(folder_path, f"{image_id}.mask")
        swc_path = os.path.join(folder_path, f"{image_id}.swc")
        save_path = os.path.join(folder_path, "split")

        if not (os.path.isfile(image_path) and os.path.isfile(mask_path) and os.path.isfile(swc_path)):
            print(f"[SKIP] Missing files in {folder_path}")
            continue

        image_split(
            image_id=image_id,
            image_path=image_path,
            mask_path=mask_path,
            swc_path=swc_path,
            tile_size=tile_size,
            stride=stride,
            save_path=save_path,
            overlay_alpha=overlay_alpha,
        )
