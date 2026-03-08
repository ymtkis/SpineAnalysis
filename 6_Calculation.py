from pathlib import Path
import re
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tiff
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.draw import polygon2mask, rectangle
from roifile import roiread, ROI_TYPE
from skimage import morphology, measure
from scipy import ndimage as ndi
import math
from typing import Optional, Tuple, List

"""
Memory-saving spine pipeline:
- Process per merged-ROI group with Z-wise cropping (reduces per-slice tensor size)
- Fixed tiling with smaller target (256) to lower peak VRAM/RAM
- Safe normalization (fallback to full-frame when ROI sparse)
- Distance filter via anisotropic 3D dilation (no full EDT allocation)
- No QC overlay by default (can be re-enabled)
"""

# ---------------------------- helpers ----------------------------

def read_tif_stack(path: str) -> np.ndarray:
    with tiff.TiffFile(path) as tif:
        pages = [p.asarray() for p in tif.pages]
    return np.stack(pages, axis=0)

def save_tif(path: Path, arr: np.ndarray, imagej: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(path), arr, imagej=imagej, metadata={'axes':'ZYX'}, compression=None)

def bbox2d(m: np.ndarray):
    yy, xx = np.where(m)
    if yy.size == 0:
        return None
    return int(yy.min()), int(yy.max())+1, int(xx.min()), int(xx.max())+1

# ---------------------------- ROI loading (union only) ----------------------------

def load_roi_zip_union(zip_path: str, ref_shape: tuple) -> np.ndarray:
    """
    ROI zip を読み込み、Z情報があればそのZに、なければ全Zに適用して
    単純に和集合（union）の 3D マスク(Z,Y,X, bool)を返す。
    """
    Z, Y, X = ref_shape
    rois = roiread(zip_path)
    if not isinstance(rois, (list, tuple)):
        rois = [rois]
    union = np.zeros((Z, Y, X), bool)
    for roi in rois:
        rt = getattr(roi, "roitype", None)
        if rt in (ROI_TYPE.LINE, getattr(ROI_TYPE, "POLYLINE", None),
                  getattr(ROI_TYPE, "FREELINE", None), getattr(ROI_TYPE, "ANGLE", None),
                  getattr(ROI_TYPE, "POINT", None)):
            continue
        coords = roi.coordinates()
        if coords is not None and len(coords) >= 3:
            rc = np.fliplr(np.asarray(coords, float))
            poly2d = polygon2mask((Y, X), rc)
        else:
            top, left, bottom, right = int(roi.top), int(roi.left), int(roi.bottom), int(roi.right)
            rr, cc = rectangle((top, left), (bottom-1, right-1), shape=(Y, X))
            poly2d = np.zeros((Y, X), bool); poly2d[rr, cc] = True
        pos = int(getattr(roi, "position", 0))
        if 1 <= pos <= Z:
            union[pos-1] |= poly2d
        else:
            union |= poly2d[None, ...]
    return union



def load_roi_zip_as_2d_list(zip_path: str, shape_yx: tuple) -> List[np.ndarray]:
    """
    ROI zip を読み込み、面積を持つROIだけを 2D bool マスク (Y,X) のリストで返す。
    Z情報（position）は使わず、投影由来の2D ROIとして扱う。
    """
    Y, X = shape_yx
    rois = roiread(zip_path)
    if not isinstance(rois, (list, tuple)):
        rois = [rois]

    masks_2d: List[np.ndarray] = []
    for roi in rois:
        rt = getattr(roi, "roitype", None)
        if rt in (ROI_TYPE.LINE, getattr(ROI_TYPE, "POLYLINE", None),
                  getattr(ROI_TYPE, "FREELINE", None), getattr(ROI_TYPE, "ANGLE", None),
                  getattr(ROI_TYPE, "POINT", None)):
            continue
        coords = roi.coordinates()
        if coords is not None and len(coords) >= 3:
            rc = np.fliplr(np.asarray(coords, float))  # (y,x)
            m2d = polygon2mask((Y, X), rc)
        else:
            top, left, bottom, right = int(roi.top), int(roi.left), int(roi.bottom), int(roi.right)
            rr, cc = rectangle((top, left), (bottom-1, right-1), shape=(Y, X))
            m2d = np.zeros((Y, X), bool); m2d[rr, cc] = True
        if m2d.any():
            masks_2d.append(m2d)
    return masks_2d
# ---------------------------- StarDist ----------------------------

_SD = None
def stardist_model():
    global _SD
    if _SD is None:
        _SD = StarDist2D.from_pretrained("2D_versatile_fluo")
    return _SD

def _auto_n_tiles(h: int, w: int, target: int = 256):
    return (max(1, math.ceil(h/target)), max(1, math.ceil(w/target)))

def _normalize_safe(img2d: np.ndarray, roi2d: Optional[np.ndarray], min_roi_pixels: int = 2000) -> np.ndarray:
    if roi2d is None:
        return normalize(img2d, 1, 99.8).astype(np.float32, copy=False)
    m = roi2d.astype(bool)
    if m.sum() < min_roi_pixels:
        return normalize(img2d, 1, 99.8).astype(np.float32, copy=False)
    vals = img2d[m]
    p1, p998 = np.percentile(vals, (1, 99.8))
    if p998 <= p1:
        return normalize(img2d, 1, 99.8).astype(np.float32, copy=False)
    out = (img2d.astype(np.float32) - p1) / (p998 - p1)
    out[out < 0] = 0; out[out > 1] = 1
    return out

def stardist_2dstack_cropped(img_zyx: np.ndarray, prob_thr: float, nms_thr: float,
                             roi_zyx: Optional[np.ndarray]) -> np.ndarray:
    """Predict per Z within ROI bounding box only. Lower memory than full-frame."""
    Z, Y, X = img_zyx.shape
    out = np.zeros((Z, Y, X), np.int32)
    model = stardist_model()
    next_id = 1
    tiles_full = _auto_n_tiles(Y, X)  # fallback
    for z in range(Z):
        roi2d = None if roi_zyx is None else roi_zyx[z]
        if roi2d is None or not roi2d.any():
            continue
        bb = bbox2d(roi2d)
        if bb is None:
            continue
        y0,y1,x0,x1 = bb
        img_crop = img_zyx[z, y0:y1, x0:x1]
        roi_crop = roi2d[y0:y1, x0:x1]
        img_n = _normalize_safe(img_crop, roi_crop)
        if img_n.max() == 0:
            continue
        lab2d, _ = model.predict_instances(img_n, prob_thresh=prob_thr, nms_thresh=nms_thr,
                                           n_tiles=_auto_n_tiles(y1-y0, x1-x0))
        lab2d = lab2d.astype(np.int32, copy=False)
        lab2d[~roi_crop] = 0
        m = lab2d > 0
        if m.any():
            lab2d[m] += (next_id - 1); next_id = int(lab2d.max()) + 1
            out[z, y0:y1, x0:x1][m] = lab2d[m]
    return out

# ---------------------------- ROI refinement ----------------------------

def refine_roi_per_z_from_union(union_mask: Optional[np.ndarray],
                                dend_mask: np.ndarray,
                                img_zyx: np.ndarray,
                                voxel_um: Tuple[float, float, float],
                                dilate_um: float = 1.0,
                                intensity_pct: float = 60.0) -> Optional[np.ndarray]:
    """Build per-Z ROI from union mask, dendrite dilation, and intensity gate."""
    if union_mask is None or not union_mask.any():
        return None
    Z, Y, X = img_zyx.shape
    dz, dy, dx = voxel_um
    px_radius = max(1, int(round(dilate_um / ((dy + dx)/2.0))))
    selem = morphology.disk(px_radius)
    roi_refined = np.zeros((Z, Y, X), bool)
    union2d = union_mask.max(axis=0)
    candidate = union2d.copy()
    for z in range(Z):
        dm = dend_mask[z] > 0
        dm_dil = morphology.binary_dilation(dm, selem)
        img = img_zyx[z]
        area = candidate & (img > 0)
        if area.any():
            thr = np.percentile(img[area], intensity_pct)
            bright = img >= thr
        else:
            bright = np.zeros((Y, X), bool)
        roi = union2d & dm_dil & bright
        #if roi.sum() < 500:
        #    roi = union2d & dm_dil
        roi_refined[z] = roi
    if not roi_refined.any():
        roi_refined = np.broadcast_to(union2d[None, ...], (Z, Y, X)).copy()
    return roi_refined



def build_roi_per_slice_selective(rois_2d: List[np.ndarray],
                                  img_zyx: np.ndarray,
                                  dend_mask: Optional[np.ndarray],
                                  voxel_um: Tuple[float, float, float],
                                  dilate_um: float = 1.0,
                                  intensity_pct: float = 60.0,
                                  min_bright_px: int = 50) -> np.ndarray:
    """
    投影由来の2D ROI（複数）を各Zに当て、各ROI内の明るさパーセンタイルで
    『シグナルがあるROIだけ』を残し、その和集合を各ZのROIにする。
    """
    Z, Y, X = img_zyx.shape
    out = np.zeros((Z, Y, X), bool)

    # 樹状突起近傍に限定（任意）
    dm_dil = None
    if dend_mask is not None:
        dz, dy, dx = voxel_um
        px = max(1, int(round(dilate_um / ((dy + dx) / 2.0))))
        se = morphology.disk(px)
        dm_dil = np.zeros_like(dend_mask, bool)
        for z in range(Z):
            dm_dil[z] = morphology.binary_dilation(dend_mask[z] > 0, se)

    for z in range(Z):
        union_z = np.zeros((Y, X), bool)
        img = img_zyx[z]
        for r2d in rois_2d:
            roi = r2d if dm_dil is None else (r2d & dm_dil[z])
            if not roi.any():
                continue
            vals = img[roi]
            if vals.size == 0:
                continue
            thr = np.percentile(vals, intensity_pct)
            bright_roi = (img >= thr) & roi
            if bright_roi.sum() >= min_bright_px:
                union_z |= roi  # ← ROIそのものを残す（和集合）
        out[z] = union_z
    return out

# ---------------------------- 3D merge & metrics ----------------------------

def merge_to_3d(labels_2dstack: np.ndarray, voxel_um: tuple, min_vol_um3: float = 0.0,
                smooth: str = 'closing') -> np.ndarray:
    bin3d = labels_2dstack > 0
    if smooth == 'closing':
        bin3d = morphology.binary_closing(bin3d, morphology.ball(1))
    elif smooth == 'none':
        pass
    else:
        bin3d = morphology.binary_opening(bin3d, morphology.ball(1))
        bin3d = morphology.binary_closing(bin3d, morphology.ball(1))
    labels3d, _ = ndi.label(bin3d, structure=np.ones((3,3,3), bool))
    if min_vol_um3 > 0:
        dz, dy, dx = voxel_um; vvoxel = dz*dy*dx
        props = measure.regionprops(labels3d)
        keep = [p.label for p in props if (p.area * vvoxel) >= min_vol_um3]
        labels3d = measure.label(np.isin(labels3d, keep), connectivity=3)
    else:
        labels3d = measure.label(labels3d, connectivity=3)
    return labels3d

def compute_table(labels3d: np.ndarray, voxel_um: tuple) -> pd.DataFrame:
    dz, dy, dx = voxel_um; vvoxel = dz*dy*dx
    rows = []
    for p in measure.regionprops(labels3d):
        zc, yc, xc = p.centroid
        zmin, ymin, xmin, zmax, ymax, xmax = p.bbox
        rows.append({
            "label": int(p.label),
            "volume_um3": float(p.area * vvoxel),
            "centroid_z_um": float(zc * dz),
            "centroid_y_um": float(yc * dy),
            "centroid_x_um": float(xc * dx),
            "z_slices": int(zmax - zmin),
            "voxels": int(p.area),
        })
    return pd.DataFrame(rows).sort_values("label")

def make_ellipsoid_footprint(voxel_um, r_um):
    dz, dy, dx = voxel_um
    rz, ry, rx = max(1, int(round(r_um/dz))), max(1, int(round(r_um/dy))), max(1, int(round(r_um/dx)))
    zz, yy, xx = np.ogrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]
    dist2 = (zz*dz)**2 + (yy*dy)**2 + (xx*dx)**2
    return dist2 <= (r_um**2)

def filter_by_dend_distance_fast(labels3d, dend_mask, voxel_um, max_dist_um):
    """Approximate distance filter by anisotropic dilation of dendrites (no full EDT)."""
    fp = make_ellipsoid_footprint(voxel_um, max_dist_um)
    near = ndi.binary_dilation(dend_mask>0, structure=fp)
    keep = []
    for p in measure.regionprops(labels3d):
        zc, yc, xc = p.centroid
        iz, iy, ix = int(round(zc)), int(round(yc)), int(round(xc))
        iz = np.clip(iz, 0, near.shape[0]-1)
        iy = np.clip(iy, 0, near.shape[1]-1)
        ix = np.clip(ix, 0, near.shape[2]-1)
        if near[iz, iy, ix]:
            keep.append(p.label)
    return measure.label(np.isin(labels3d, keep), connectivity=3)

# ---------------------------- Driver ----------------------------

if __name__ == "__main__":
    base = Path("/mnt/d/Q_project/SpineAnalysis")
    conditions = ["BSL", "QIH(48h)"]

    VOXEL_UM = (0.5, 0.0353, 0.0353)
    PROB_THR  = 0.2
    NMS_THR   = 0.75
    MIN_VOL   = 0.01
    MAX_DIST  = None    # um (set None to disable)
    DILATE_UM = 1.0    # ROI dilation for dendrites
    INT_PCT   = 60.0   # looser per-Z intensity gate

    for cond in conditions:
        cdir = base / f"{cond}_deconv"
        for tif in sorted(cdir.glob("*_bg_spines.tif")):
            prefix   = re.sub(r"_bg_spines$", "", tif.stem)
            img      = read_tif_stack(str(tif)).astype(np.float32)

            # dendrite mask
            dm = read_tif_stack(str((cdir / f"{prefix}_bg_dendrites.tif"))) 
            if dm.ndim == 2:
                dm = np.broadcast_to(dm[None, ...], img.shape)
            dend_mask = dm.astype(bool, copy=False)

            # load ROI union (no merging)
            zip_path = cdir / f"{prefix}_bg_merged.zip"
            if zip_path.exists():
                rois_2d = load_roi_zip_as_2d_list(str(zip_path), img.shape[1:])   # (Y,X)  # <<< NEW
            else:
                rois_2d = []

            # output buffers
            lab2d_total = np.zeros(img.shape, np.int32)

            # per-slice selective ROI を作って StarDist へ                # <<< CHANGED
            if len(rois_2d) > 0:
                refined = build_roi_per_slice_selective(                       # <<< NEW
                    rois_2d, img, dend_mask, VOXEL_UM,
                    dilate_um=DILATE_UM, intensity_pct=INT_PCT, min_bright_px=50
                )
                lab2d_total = stardist_2dstack_cropped(img, PROB_THR, NMS_THR, refined)

            # 3D merge
            lab3d = merge_to_3d(lab2d_total, VOXEL_UM, min_vol_um3=MIN_VOL, smooth='closing')

            # distance filter (fast, no EDT)
            if MAX_DIST is not None:
                lab3d = filter_by_dend_distance_fast(lab3d, dend_mask, VOXEL_UM, MAX_DIST)

            # metrics
            df = compute_table(lab3d, VOXEL_UM)
            total = int(lab3d.max())
            # (optional) dendrite length can be heavy; compute only if needed
            # Comment out if memory/time is still tight.
            def dend_length_um(dm, voxel_um):
                skel = morphology.skeletonize_3d(dm > 0)
                dz, dy, dx = voxel_um
                coords = np.column_stack(np.where(skel))
                s = set(map(tuple, coords))
                neigh = [(a,b,c) for a in (-1,0,1) for b in (-1,0,1) for c in (-1,0,0) if (a,b,c)!=(0,0,0)]
                def fwd(v): a,b,c=v; return (a>0) or (a==0 and b>0) or (a==0 and b==0 and c>0)
                length = 0.0
                for z,y,x in coords:
                    for oz,oy,ox in filter(fwd, neigh):
                        nb = (z+oz, y+oy, x+ox)
                        if nb in s:
                            length += ((oz*dz)**2+(oy*dy)**2+(ox*dx)**2) ** 0.5
                return length

            dlen = dend_length_um(dend_mask, VOXEL_UM)
            dens = (total / dlen) if dlen > 0 else np.nan
            df["n_spines_total"]    = total
            df["dend_length_um"]    = dlen
            df["density_per_um"]    = dens
            #df["density_per_10um"]  = dens * 10 if np.isfinite(dens) else np.nan
            #df["density_per_100um"] = dens * 100 if np.isfinite(dens) else np.nan

            # save
            save_tif(cdir / f"{prefix}_spines_label_3d.tif", lab3d.astype(np.uint16))
            df.to_csv(str(cdir / f"{prefix}_spines_table.csv"), index=False)
