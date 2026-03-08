from pathlib import Path
from glob import glob
import numpy as np
import tifffile as tiff
import flammkuchen as fl

from config_utils import load_config, cfg_get


def match_shape(bg, mask):
    if bg.shape == mask.shape:
        return mask
    if bg.ndim == 2 and mask.ndim == 3:
        m2 = mask.any(axis=0)
        if m2.shape == bg.shape:
            return m2
    if bg.ndim == 3 and mask.ndim == 2 and mask.shape == bg.shape[-2:]:
        return np.broadcast_to(mask, bg.shape)
    raise ValueError(f"shape mismatch: bg={bg.shape}, mask={mask.shape}")


def to_u8_binary(prob, thr):
    return ((prob >= float(thr)).astype(np.uint8) * 255)


def process(pred_path: Path, spine_thr: float, dendrite_thr: float, skip_if_exists: bool):
    stem = pred_path.stem
    out_d = pred_path.with_name(stem + "_dendrites.tif")
    out_s = pred_path.with_name(stem + "_spines.tif")

    if skip_if_exists and (out_d.exists() or out_s.exists()):
        return

    data = fl.load(pred_path)
    dend = data.get("dendrites")
    spin = data.get("spines")
    if dend is None or spin is None:
        return

    bg_path = pred_path.with_name(stem + ".tif")
    if not bg_path.exists():
        return
    bg = tiff.imread(bg_path)

    # dendrite mask (binary)
    tiff.imwrite(out_d, to_u8_binary(dend, dendrite_thr), dtype=np.uint8)

    # spine cutout (bg masked by spine prob)
    mask = (spin >= float(spine_thr))
    try:
        mask = match_shape(bg, mask)
    except ValueError:
        return

    spines_cut = np.where(mask, bg, 0)
    tiff.imwrite(out_s, spines_cut.astype(bg.dtype), dtype=bg.dtype)


def main():
    cfg = load_config("config.yaml")

    deconv_root = Path(cfg_get(cfg, ["deconv_root"], "/mnt/d/Q_project/SpineAnalysis/Data"))
    conditions = list(cfg_get(cfg, ["conditions"], []))
    pred_glob_template = str(cfg_get(
        cfg,
        ["prediction_masks", "pred_glob_template"],
        "{condition}_deconv/*_cmle/*_cmle_bg.prediction",
    ))

    spine_thr = float(cfg_get(cfg, ["prediction_masks", "spine_thr"], 0.10))
    dendrite_thr = float(cfg_get(cfg, ["prediction_masks", "dendrite_thr"], 0.70))
    skip_if_exists = bool(cfg_get(cfg, ["prediction_masks", "skip_if_exists"], True))

    preds = []
    for cond in conditions:
        pattern = str(deconv_root / pred_glob_template.format(condition=cond))
        preds += glob(pattern)

    for p in sorted(set(preds)):
        process(Path(p), spine_thr=spine_thr, dendrite_thr=dendrite_thr, skip_if_exists=skip_if_exists)


if __name__ == "__main__":
    main()