from __future__ import annotations

from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd
import tifffile as tiff
import flammkuchen as fl
from skimage.morphology import skeletonize_3d, remove_small_objects, binary_closing, ball


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg_get(cfg: dict, keys: list[str], default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def iter_roi_maps(cfg: dict) -> list[Path]:
    deconv_root = Path(_cfg_get(cfg, ["deconv_root"], "."))
    conditions = list(_cfg_get(cfg, ["conditions"], []))
    if not conditions:
        raise KeyError("config.yaml must define: conditions (non-empty)")

    pat_in = str(_cfg_get(cfg, ["spine_quantification", "roi_map_glob"], "*/*_roi_map.tif"))

    roi_maps: list[Path] = []
    for cond in conditions:
        cond_dir = deconv_root / f"{cond}_deconv"
        if not cond_dir.exists():
            print(f"[WARN] Condition folder not found, skip: {cond_dir}")
            continue
        roi_maps.extend(sorted(cond_dir.glob(pat_in)))

    return sorted(roi_maps)


def iter_dendrite_masks(cfg: dict) -> list[Path]:
    deconv_root = Path(_cfg_get(cfg, ["deconv_root"], "."))
    conditions = list(_cfg_get(cfg, ["conditions"], []))
    if not conditions:
        raise KeyError("config.yaml must define: conditions (non-empty)")

    pat_in = str(_cfg_get(cfg, ["dendrite_length", "mask_glob"], "*/*_dendrites.tif"))

    masks: list[Path] = []
    for cond in conditions:
        cond_dir = deconv_root / f"{cond}_deconv"
        if not cond_dir.exists():
            print(f"[WARN] Condition folder not found, skip: {cond_dir}")
            continue
        masks.extend(sorted(cond_dir.glob(pat_in)))

    return sorted(masks)


def parse_metadata_from_name(roi_map_path: Path, cfg: dict) -> dict:
    meta_cfg = _cfg_get(cfg, ["spine_quantification", "metadata_from_name"], {}) or {}
    source = str(meta_cfg.get("source", "parent"))
    strip_suffix = str(meta_cfg.get("strip_suffix", ""))
    token_map = meta_cfg.get("token_map", {}) or {}

    if source == "stem":
        name = roi_map_path.stem
    else:
        name = roi_map_path.parent.name

    if strip_suffix and name.endswith(strip_suffix):
        name = name[: -len(strip_suffix)]

    tokens = name.split("_")

    out = {}
    for key, idx in token_map.items():
        try:
            out[key] = tokens[int(idx)]
        except Exception:
            out[key] = ""

    return out


def compute_centroids(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (labels, counts, cz, cy, cx) in voxel coordinates.
    """
    roi = np.asarray(roi)
    mask = roi != 0
    if not np.any(mask):
        return (np.array([], dtype=np.int32),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64))

    z, y, x = np.nonzero(mask)
    lab = roi[z, y, x].astype(np.int64)

    max_lab = int(lab.max())
    counts = np.bincount(lab, minlength=max_lab + 1).astype(np.int64)

    sum_z = np.bincount(lab, weights=z.astype(np.float64), minlength=max_lab + 1)
    sum_y = np.bincount(lab, weights=y.astype(np.float64), minlength=max_lab + 1)
    sum_x = np.bincount(lab, weights=x.astype(np.float64), minlength=max_lab + 1)

    valid = np.where(counts > 0)[0]
    valid = valid[valid != 0] 

    cz = sum_z[valid] / counts[valid]
    cy = sum_y[valid] / counts[valid]
    cx = sum_x[valid] / counts[valid]

    return valid.astype(np.int32), counts[valid], cz, cy, cx


def spine_count(roi_map_path: Path, xy_um: float, z_um: float, min_voxels: int, cfg: dict) -> pd.DataFrame:
    roi = tiff.imread(roi_map_path)

    labels, counts, cz, cy, cx = compute_centroids(roi)

    keep = counts >= int(min_voxels)
    labels = labels[keep]
    counts = counts[keep]
    cz = cz[keep]
    cy = cy[keep]
    cx = cx[keep]

    voxel_volume_um3 = float(xy_um) * float(xy_um) * float(z_um)
    volumes_um3 = counts.astype(np.float64) * voxel_volume_um3

    centroid_z_um = cz * float(z_um)
    centroid_y_um = cy * float(xy_um)
    centroid_x_um = cx * float(xy_um)

    df = pd.DataFrame({
        "label_id": labels.astype(int),
        "volume_um3": volumes_um3,
        "centroid_z_um": centroid_z_um,
        "centroid_y_um": centroid_y_um,
        "centroid_x_um": centroid_x_um,
    })
    return df


def write_outputs(df: pd.DataFrame, base_path_noext: Path, cfg: dict, meta: dict):
    outs = list(_cfg_get(cfg, ["spine_quantification", "outputs"], ["csv"]))
    outs = [str(x).lower() for x in outs]

    if "csv" in outs:
        suf = str(_cfg_get(cfg, ["spine_quantification", "out_suffix_csv"], "_spine_stats.csv"))
        df.to_csv(base_path_noext.with_name(base_path_noext.name + suf), index=False)
    if "h5" in outs or "hdf5" in outs:
        suf = str(_cfg_get(cfg, ["spine_quantification", "out_suffix_h5"], "_spine_stats.h5"))
        out_h5 = base_path_noext.with_name(base_path_noext.name + suf)
        fl.save(out_h5, {"spine_table": df, "meta": meta}, compression="blosc")


def write_dendrite_length_outputs(df: pd.DataFrame, base_path_noext: Path, cfg: dict, meta: dict):
    outs = list(_cfg_get(cfg, ["dendrite_length", "outputs"], ["csv"]))
    outs = [str(x).lower() for x in outs]

    if "csv" in outs:
        suf = str(_cfg_get(cfg, ["dendrite_length", "out_suffix_csv"], "_dendrite_length.csv"))
        df.to_csv(base_path_noext.with_name(base_path_noext.name + suf), index=False)

    if "h5" in outs or "hdf5" in outs:
        suf = str(_cfg_get(cfg, ["dendrite_length", "out_suffix_h5"], "_dendrite_length.h5"))
        out_h5 = base_path_noext.with_name(base_path_noext.name + suf)
        fl.save(out_h5, {"dendrite_length_table": df, "meta": meta}, compression="blosc")


def _shift_and_and(a: np.ndarray, dz: int, dy: int, dx: int) -> np.ndarray:
    """Return a & shifted(a) without wrap-around."""
    z0a = max(0, dz); z1a = a.shape[0] + min(0, dz)
    y0a = max(0, dy); y1a = a.shape[1] + min(0, dy)
    x0a = max(0, dx); x1a = a.shape[2] + min(0, dx)

    z0b = max(0, -dz); z1b = a.shape[0] - max(0, dz)
    y0b = max(0, -dy); y1b = a.shape[1] - max(0, dy)
    x0b = max(0, -dx); x1b = a.shape[2] - max(0, dx)

    if z1a <= z0a or y1a <= y0a or x1a <= x0a:
        return np.zeros((0,), dtype=bool)

    return a[z0a:z1a, y0a:y1a, x0a:x1a] & a[z0b:z1b, y0b:y1b, x0b:x1b]


def skeleton_total_length_um(skel: np.ndarray, xy_um: float, z_um: float) -> float:
    """
    Counts skeleton adjacency edges in the 26-neighborhood and 
    calculates the total length using anisotropic spacing (dx = dy = xy_um, dz = z_um).
    """
    skel = np.asarray(skel).astype(bool)
    if not np.any(skel):
        return 0.0

    dx = float(xy_um)
    dy = float(xy_um)
    dz_um = float(z_um)

    total = 0.0
    offsets = []
    for oz in (-1, 0, 1):
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                if oz == 0 and oy == 0 and ox == 0:
                    continue
                if (oz > 0) or (oz == 0 and oy > 0) or (oz == 0 and oy == 0 and ox > 0):
                    offsets.append((oz, oy, ox))

    for oz, oy, ox in offsets:
        pairs = _shift_and_and(skel, oz, oy, ox)
        if pairs.size == 0:
            continue
        n_edges = int(pairs.sum())
        dist = float(np.sqrt((ox * dx) ** 2 + (oy * dy) ** 2 + (oz * dz_um) ** 2))
        total += n_edges * dist

    return float(total)


def compute_dendrite_length(mask_path: Path, xy_um: float, z_um: float, cfg: dict) -> dict:
    if skeletonize_3d is None:
        raise ImportError("scikit-image is required for dendrite length. Install: pip install scikit-image")

    m = tiff.imread(mask_path)
    m = np.asarray(m)

    if m.ndim == 2:
        m = m[None, :, :]

    bw = (m > 0)

    min_obj = int(_cfg_get(cfg, ["dendrite_length", "min_object_voxels"], 50))
    do_close = bool(_cfg_get(cfg, ["dendrite_length", "do_closing"], True))
    close_r = int(_cfg_get(cfg, ["dendrite_length", "closing_radius"], 1))

    if remove_small_objects is not None and min_obj > 0:
        bw = remove_small_objects(bw, min_size=min_obj)

    if do_close and binary_closing is not None and ball is not None and close_r > 0:
        bw = binary_closing(bw, footprint=ball(close_r))

    skel = skeletonize_3d(bw)

    total_um = skeleton_total_length_um(skel, xy_um=xy_um, z_um=z_um)

    skel_vox = int(np.asarray(skel, dtype=bool).sum())
    mask_vox = int(np.asarray(bw, dtype=bool).sum())

    if skel_vox > 0:
        ratio = mask_vox / skel_vox
    else:
        ratio = np.nan

    return {
        "dendrite_total_length_um": total_um,
        "dendrite_skel_voxels": skel_vox,
        "dendrite_mask_voxels": mask_vox,
        "mask_to_skel_ratio": ratio,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))

    xy_um = float(_cfg_get(cfg, ["spine_quantification", "xy_um"], 0.0353))
    z_um = float(_cfg_get(cfg, ["spine_quantification", "z_um"], 0.5))
    min_voxels = int(_cfg_get(cfg, ["spine_quantification", "min_voxels"], 1))

    # spine
    roi_maps = iter_roi_maps(cfg)
    if not roi_maps:
        raise FileNotFoundError("No *_roi_map.tif found")
    all_rows = []
    for p in roi_maps:
        df = spine_count(p, xy_um=xy_um, z_um=z_um, min_voxels=min_voxels, cfg=cfg)
        base_noext = p.with_suffix("") 

        meta = {
            "xy_um": xy_um,
            "z_um": z_um,
            "min_voxels": min_voxels,
            "roi_map_path": str(p),
            **parse_metadata_from_name(p, cfg),
        }

        write_outputs(df, base_noext, cfg, meta)
        print(f"[OK] {p.name}")
        all_rows.append(df)

    # dendrite
    do_dend = bool(_cfg_get(cfg, ["dendrite_length", "enabled"], True))
    if do_dend:
        masks = list(iter_dendrite_masks(cfg))
        if len(masks) == 0:
            print("[WARN] No *_dendrites.tif found. Skip dendrite length.")
        else:
            rows = []
            meta_keys = ["animal_id", "condition", "cell_id", "branch", "branch_index"]

            for mp in masks:
                meta = parse_metadata_from_name(mp, cfg)
                row = {k: meta.get(k, "") for k in meta_keys}
                rec = compute_dendrite_length(mp, xy_um=xy_um, z_um=z_um, cfg=cfg)
                row.update(rec)
                rows.append(row)
                print(f"[OK] {mp.name}")

            df_len = pd.DataFrame(rows)
            out_root = Path(_cfg_get(cfg, ["deconv_root"], "."))
            base_noext = out_root / "ALL_dendrite_length"
            meta_len = {"xy_um": xy_um, "z_um": z_um}
            write_dendrite_length_outputs(df_len, base_noext, cfg, meta_len)
            print("[OK] Saved", out_root)
            

if __name__ == "__main__":
    main()
