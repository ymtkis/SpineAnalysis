import re
from pathlib import Path
import numpy as np
import tifffile as tiff
from skimage import filters, morphology
from config_utils import load_config, cfg_get


def read_tif_stack(path: Path) -> np.ndarray:
    with tiff.TiffFile(str(path)) as tf:
        best = None
        best_key = (-1, -1, -1)
        for s in tf.series:
            axes = getattr(s, "axes", "")
            shape = tuple(s.shape)
            score = (int("Z" in axes), int(len(shape) >= 3), int(np.prod(shape)))
            if score > best_key:
                best, best_key = s, score

        if best is None:
            arr = np.stack([p.asarray() for p in tf.pages], axis=0)
            axes = "ZYX" if arr.ndim == 3 else "YX"
        else:
            arr = best.asarray()
            axes = getattr(best, "axes", "")

        if "Z" not in axes and "T" in axes:
            axes = axes.replace("T", "Z", 1)

        if "C" in axes:
            c_ax = axes.index("C")
            arr = np.take(arr, 0, axis=c_ax)
            axes = axes.replace("C", "")

        if arr.ndim == 2:
            arr = arr[None, ...]
            axes = "Z" + axes

        ax_map = {ax: i for i, ax in enumerate(axes)}
        order = [ax_map.get("Z", None), ax_map["Y"], ax_map["X"]]
        if order[0] is None:
            arr = arr[None, ...]
            order = [0, arr.ndim - 2, arr.ndim - 1]

        arr = np.moveaxis(arr, order, (0, 1, 2)).astype(np.float32, copy=False)
        return arr  # (Z,Y,X)


def to_unit(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    mx = float(np.max(img)) if img.size else 0.0
    if mx > 1.0:
        img = img / mx
    return img


def make_mask(
    pred_tif: Path,
    out_mask: Path,
    prob_thresh,
    open_close_radius: int,
    min_hole_voxels: int,
):
    dend = read_tif_stack(pred_tif)
    dend = to_unit(dend)

    if prob_thresh is None:
        th = float(filters.threshold_otsu(dend.reshape(-1)))
    else:
        th = float(prob_thresh)

    mask = dend >= th

    se = morphology.ball(int(open_close_radius))
    mask = morphology.binary_opening(mask, se)
    mask = morphology.binary_closing(mask, se)
    mask = morphology.remove_small_holes(mask, area_threshold=int(min_hole_voxels))

    out_mask.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(out_mask), (mask.astype(np.uint8) * 255))


def main():
    cfg = load_config("config.yaml")

    dendrite_root = Path(cfg_get(cfg, ["dendrite_mask", "dendrite_root"], "/mnt/d/Q_project/SpineAnalysis"))
    conditions = list(cfg_get(cfg, ["conditions"], []))
    deconv_suffix = str(cfg_get(cfg, ["dendrite_mask", "deconv_suffix"], "_deconv"))
    input_glob = str(cfg_get(cfg, ["dendrite_mask", "input_glob"], "*_bg.tif"))

    pred_suffix = str(cfg_get(cfg, ["dendrite_mask", "pred_suffix"], "_bg_dendrites.tif"))
    out_suffix = str(cfg_get(cfg, ["dendrite_mask", "out_suffix"], "_bg_dendrites_mask.tif"))

    prob_thresh = cfg_get(cfg, ["dendrite_mask", "prob_thresh"], None)
    open_close_radius = int(cfg_get(cfg, ["dendrite_mask", "open_close_radius"], 1))
    min_hole_voxels = int(cfg_get(cfg, ["dendrite_mask", "min_hole_voxels"], 50))

    for condition in conditions:
        cond_dir = dendrite_root / f"{condition}{deconv_suffix}"
        files = sorted(cond_dir.glob(input_glob))

        for file in files:
            prefix = re.sub(r"_bg$", "", file.stem)

            pred_tif = cond_dir / f"{prefix}{pred_suffix}"
            out_mask = cond_dir / f"{prefix}{out_suffix}"

            if not pred_tif.is_file():
                print(f"[SKIP] pred tif not found: {pred_tif}")
                continue

            make_mask(
                pred_tif=pred_tif,
                out_mask=out_mask,
                prob_thresh=prob_thresh,
                open_close_radius=open_close_radius,
                min_hole_voxels=min_hole_voxels,
            )
            print(f"[OK] {out_mask}")


if __name__ == "__main__":
    main()
