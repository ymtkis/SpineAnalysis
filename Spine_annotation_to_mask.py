import os
import glob
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import flammkuchen as fl

# === 設定 ===
base_dir = "/mnt/d/Q_project/SpineAnalysis"
conditions = ["BSL", "QIH(48h)"]

def tif_to_mask(tif_path):
    """非ゼロ画素をannotation領域としてmask生成（左下↔右上の対角線で反転）"""
    img = iio.imread(tif_path)
    print(f"[INFO] {tif_path}: shape={img.shape}, dtype={img.dtype}")

    # --- 非ゼロ画素を抽出 ---
    if img.ndim == 2:
        mask = img > 0
        mask3d = mask[None, :, :]
    elif img.ndim == 3:
        if img.shape[-1] in (3, 4):  # RGB
            gray = np.mean(img, axis=-1)
            mask = gray > 0
            mask3d = mask[None, :, :]
        else:  # Z-stack
            mask3d = img > 0
    elif img.ndim == 4:  # (Z, Y, X, C)
        gray = np.mean(img, axis=-1)
        mask3d = gray > 0
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # --- 左下↔右上の対角線で反転（xとyを入れ替える）---
    mask3d = np.transpose(mask3d, axes=(0, 2, 1))

    return mask3d.astype(bool)


for cond in conditions:
    # 例: .../{cond}_deconv/*_cmle/*_bg_spines.tif
    pattern = os.path.join(base_dir, f"{cond}_deconv", "*_cmle", "*_bg_spines.tif")
    tifs = glob.glob(pattern)

    if not tifs:
        print(f"[WARN] No matching files for condition: {cond}")
        continue

    for tif_path in tifs:
        

        # === 出力ファイル名変換 ===
        # "_bg_spines.tif" → "_bg.mask"
        mask_path = tif_path.replace("_bg_spines.tif", "_bg.mask")
        if not Path(mask_path).exists():
            mask3d = tif_to_mask(tif_path)
            fl.save(mask_path, {"mask": mask3d, "files": None}, compression="blosc")

            print(f"[OK] Saved: {mask_path}, shape={mask3d.shape}")
