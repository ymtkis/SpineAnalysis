import os
import shlex
import subprocess
from glob import glob
from pathlib import Path
from typing import List, Dict

from PyQt5.QtWidgets import (
    QApplication, QDialog, QLabel, QPushButton, QVBoxLayout,
    QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import Qt

import numpy as np
import tifffile as tiff
import flammkuchen as fl

from config_utils import load_config, cfg_get


def select_files_gui_grouped(
    groups: Dict[str, List[str]],
    title: str = "Select inputs",
) -> List[str]:
    all_files: List[str] = []
    for _, files in groups.items():
        all_files.extend(files)
    all_files = sorted(set(all_files))

    if not all_files:
        return []

    dlg = QDialog()
    dlg.setWindowTitle(title)
    layout = QVBoxLayout(dlg)

    total = len(all_files)
    n_cond = sum(1 for _, fs in groups.items() if fs)
    layout.addWidget(QLabel(
        f"Total candidates: {total}  |  conditions with files: {n_cond}\n"
        "Select images to infer\n"
        "No selection will process all images."
    ))

    tree = QTreeWidget()
    tree.setHeaderLabels(["Condition / File", "Count"])
    tree.setUniformRowHeights(True)

    for cond in sorted(groups.keys()):
        files = sorted(set(groups.get(cond, [])))
        if not files:
            continue

        parent = QTreeWidgetItem(tree)
        parent.setText(0, cond)
        parent.setText(1, str(len(files)))
        parent.setFlags(parent.flags() | Qt.ItemIsUserCheckable)
        parent.setCheckState(0, Qt.Unchecked)

        for fp in files:
            child = QTreeWidgetItem(parent)
            child.setText(0, str(Path(fp).name))
            child.setToolTip(0, fp)
            child.setData(0, Qt.UserRole, fp)
            child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
            child.setCheckState(0, Qt.Unchecked)

    tree.expandAll()

    def on_item_changed(item, column):
        if column != 0:
            return
        if item.parent() is None:
            state = item.checkState(0)
            tree.blockSignals(True)
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, state)
            tree.blockSignals(False)

    tree.itemChanged.connect(on_item_changed)

    layout.addWidget(tree)

    btn_ok = QPushButton("OK")
    btn_cancel = QPushButton("Cancel")
    layout.addWidget(btn_ok)
    layout.addWidget(btn_cancel)

    selected: List[str] = []

    def on_ok():
        nonlocal selected
        checked: List[str] = []
        root = tree.invisibleRootItem()
        for i in range(root.childCount()):
            parent = root.child(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.checkState(0) == Qt.Checked:
                    checked.append(child.data(0, Qt.UserRole))
        selected = checked
        dlg.accept()

    btn_ok.clicked.connect(on_ok)
    btn_cancel.clicked.connect(dlg.reject)

    if dlg.exec_() != QDialog.Accepted:
        return []

    return selected if selected else all_files


def _bool(x) -> bool:
    return bool(x)


def save_roi_map_tif(rois_path: Path) -> Path:
    """
    Convert DeepD3 *.rois (flammkuchen) into an ImageJ-readable label TIFF (*_roi_map.tif).
    Returns the output tif path.
    """
    out_path = rois_path.with_suffix("").with_name(rois_path.stem + "_roi_map.tif")

    d = fl.load(rois_path)
    if "roi_map" not in d:
        raise KeyError(f"{rois_path} does not contain 'roi_map'. keys={list(d.keys())}")

    roi_map = d["roi_map"]

    # Choose dtype for TIFF compatibility
    if int(np.max(roi_map)) < int(np.iinfo(np.uint16).max):
        roi_map = roi_map.astype(np.uint16, copy=False)
    else:
        roi_map = roi_map.astype(np.uint32, copy=False)

    tiff.imwrite(out_path, roi_map, imagej=True)
    print("Saved:", out_path)
    return out_path


def run_one(
    image_path: Path,
    model_path: str,
    tile_size: int,
    inset_size: int,
    average: bool,
    plane: bool,
    clean_dendrite: bool,
    clean_dendrite_3d: bool,
    dendrite_threshold: float,
    min_dendrite_size: int,
    dendrite_dilation: int,
    clean_spines: bool,
    build_rois_2d: bool,
    build_rois_3d: bool,
    roi_method: str,
    roi_areaThreshold: float,
    roi_peakThreshold: float,
    roi_seedDelta: float,
    roi_distanceToSeed: float,
    watershed: bool,
    clean_rois: bool,
    min_roi_size: int,
    max_roi_size: int,
    min_planes: int,
    env: dict,
    export_roi_map_tif: bool,
):
    # --- sanity checks ---
    if average and plane:
        raise ValueError("Config error: deepd3_inference.average and .plane cannot both be true.")
    if build_rois_2d and build_rois_3d:
        raise ValueError("Config error: deepd3_inference.build_rois_2d and .build_rois_3d cannot both be true.")
    if clean_dendrite and clean_dendrite_3d:
        raise ValueError("Config error: clean_dendrite and clean_dendrite_3d cannot both be true. Choose one.")

    cmd = [
        "python", "-m", "deepd3.inference.batch",
        str(image_path),
        str(model_path),
        "--tile_size", str(int(tile_size)),
        "--inset_size", str(int(inset_size)),
    ]

    # --- inference mode ---
    if average:
        cmd += ["--average"]
    if plane:
        cmd += ["--plane"]

    # --- cleaning ---
    if clean_dendrite:
        cmd += ["--clean_dendrite"]
    if clean_dendrite_3d:
        cmd += ["--clean_dendrite_3d"]

    cmd += ["--dendrite_threshold", str(float(dendrite_threshold))]
    cmd += ["--min_dendrite_size", str(int(min_dendrite_size))]
    cmd += ["--dendrite_dilation", str(int(dendrite_dilation))]

    if clean_spines:
        cmd += ["--clean_spines"]

    # --- ROI building ---
    if build_rois_2d:
        cmd += ["--build_rois_2d"]
        if watershed:
            cmd += ["--watershed"]
        if clean_rois:
            cmd += ["--clean_rois"]
        cmd += ["--min_roi_size", str(int(min_roi_size))]
        cmd += ["--max_roi_size", str(int(max_roi_size))]
        cmd += ["--min_planes", str(int(min_planes))]

    elif build_rois_3d:
        cmd += ["--build_rois_3d"]
        cmd += ["--roi_method", str(roi_method)]
        cmd += ["--roi_areaThreshold", str(float(roi_areaThreshold))]
        cmd += ["--roi_peakThreshold", str(float(roi_peakThreshold))]
        cmd += ["--roi_seedDelta", str(float(roi_seedDelta))]
        cmd += ["--roi_distanceToSeed", str(float(roi_distanceToSeed))]

        if watershed:
            cmd += ["--watershed"]
        if clean_rois:
            cmd += ["--clean_rois"]

        cmd += ["--min_roi_size", str(int(min_roi_size))]
        cmd += ["--max_roi_size", str(int(max_roi_size))]
        cmd += ["--min_planes", str(int(min_planes))]

    print("\n>>> RUN:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True, env=env)

    # --- export roi_map.tif after ROI building ---
    if export_roi_map_tif and (build_rois_2d or build_rois_3d):
        rois_path = image_path.with_suffix(".rois")
        if rois_path.exists():
            try:
                save_roi_map_tif(rois_path)
            except Exception as e:
                print(f"[WARN] Failed to export roi_map.tif for {rois_path}: {e}")
        else:
            print(f"[WARN] .rois not found: {rois_path}")


def main():
    cfg = load_config("config.yaml")

    docker_spine_root = cfg_get(cfg, ["deepd3_inference", "docker_spine_root"], None)
    if not docker_spine_root:
        raise KeyError("config.yaml must define: deepd3_inference.docker_spine_root")

    conditions = list(cfg_get(cfg, ["conditions"], []))
    if not conditions:
        raise KeyError("config.yaml must define: deepd3_inference.conditions (non-empty)")

    input_glob_template = str(cfg_get(
        cfg,
        ["deepd3_inference", "input_glob_template"],
        "{condition}_deconv/*_cmle/*_cmle.tif",
    ))

    model_path = cfg_get(cfg, ["deepd3_inference", "model_path"], None)
    if not model_path:
        model_base = cfg_get(cfg, ["docker_base_path"], None)
        if not model_base:
            raise KeyError("config.yaml must define deepd3_inference.model_path or docker_base_path")
        model_path = str(Path(model_base) / "DeepD3_None_model_s0.h5")

    tile_size = int(cfg_get(cfg, ["deepd3_inference", "tile_size"], 128))
    inset_size = int(cfg_get(cfg, ["deepd3_inference", "inset_size"], 96))

    average = _bool(cfg_get(cfg, ["deepd3_inference", "average"], False))
    plane = _bool(cfg_get(cfg, ["deepd3_inference", "plane"], False))

    clean_dendrite = _bool(cfg_get(cfg, ["deepd3_inference", "clean_dendrite"], False))
    clean_dendrite_3d = _bool(cfg_get(cfg, ["deepd3_inference", "clean_dendrite_3d"], True))
    dendrite_threshold = float(cfg_get(cfg, ["deepd3_inference", "dendrite_threshold"], 0.7))
    min_dendrite_size = int(cfg_get(cfg, ["deepd3_inference", "min_dendrite_size"], 100))
    dendrite_dilation = int(cfg_get(cfg, ["deepd3_inference", "dendrite_dilation"], 11))
    clean_spines = _bool(cfg_get(cfg, ["deepd3_inference", "clean_spines"], True))

    build_rois_2d = _bool(cfg_get(cfg, ["deepd3_inference", "build_rois_2d"], False))
    build_rois_3d = _bool(cfg_get(cfg, ["deepd3_inference", "build_rois_3d"], False))

    roi_method = str(cfg_get(cfg, ["deepd3_inference", "roi_method"], "floodfill"))
    roi_areaThreshold = float(cfg_get(cfg, ["deepd3_inference", "roi_areaThreshold"], 0.25))
    roi_peakThreshold = float(cfg_get(cfg, ["deepd3_inference", "roi_peakThreshold"], 0.80))
    roi_seedDelta = float(cfg_get(cfg, ["deepd3_inference", "roi_seedDelta"], 0.2))
    roi_distanceToSeed = float(cfg_get(cfg, ["deepd3_inference", "roi_distanceToSeed"], 10))

    watershed = _bool(cfg_get(cfg, ["deepd3_inference", "watershed"], False))
    clean_rois = _bool(cfg_get(cfg, ["deepd3_inference", "clean_rois"], False))
    min_roi_size = int(cfg_get(cfg, ["deepd3_inference", "min_roi_size"], 10))
    max_roi_size = int(cfg_get(cfg, ["deepd3_inference", "max_roi_size"], 1000))
    min_planes = int(cfg_get(cfg, ["deepd3_inference", "min_planes"], 1))

    cuda_visible_devices = str(cfg_get(cfg, ["deepd3_inference", "cuda_visible_devices"], "0"))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    export_roi_map_tif = _bool(cfg_get(cfg, ["deepd3_inference", "export_roi_map_tif"], True))

    base = Path(docker_spine_root)
    groups = {}
    for cond in conditions:
        pattern = base / input_glob_template.format(condition=cond)
        groups[cond] = glob(str(pattern))

    if not any(groups.get(c) for c in groups):
        print("No inputs found.")
        return

    app = QApplication.instance() or QApplication([])
    selected = select_files_gui_grouped(groups, title="Select inputs")
    if not selected:
        return

    for p in selected:
        run_one(
            image_path=Path(p),
            model_path=model_path,
            tile_size=tile_size,
            inset_size=inset_size,
            average=average,
            plane=plane,
            clean_dendrite=clean_dendrite,
            clean_dendrite_3d=clean_dendrite_3d,
            dendrite_threshold=dendrite_threshold,
            min_dendrite_size=min_dendrite_size,
            dendrite_dilation=dendrite_dilation,
            clean_spines=clean_spines,
            build_rois_2d=build_rois_2d,
            build_rois_3d=build_rois_3d,
            roi_method=roi_method,
            roi_areaThreshold=roi_areaThreshold,
            roi_peakThreshold=roi_peakThreshold,
            roi_seedDelta=roi_seedDelta,
            roi_distanceToSeed=roi_distanceToSeed,
            watershed=watershed,
            clean_rois=clean_rois,
            min_roi_size=min_roi_size,
            max_roi_size=max_roi_size,
            min_planes=min_planes,
            env=env,
            export_roi_map_tif=export_roi_map_tif,
        )


if __name__ == "__main__":
    main()