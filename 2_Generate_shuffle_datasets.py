import os
import shutil
import random
import glob
from typing import List, Optional

from config_utils import load_config, cfg_get


def collect_all_d3data_files(root_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root_dir, "*", "split", "*.d3data")))


def generate_d3sets(
    source_dir: str,
    out_root: str,
    repeats: int = 3,
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
):
    d3data_paths = collect_all_d3data_files(source_dir)

    if seed is not None:
        random.seed(int(seed))

    for i in range(int(repeats)):
        random.shuffle(d3data_paths)

        n_train = int(len(d3data_paths) * float(train_ratio))
        train_paths = d3data_paths[:n_train]
        val_paths = d3data_paths[n_train:]

        train_out_dir = os.path.join(out_root, "Training_dataset", f"shuffle_{i}")
        val_out_dir = os.path.join(out_root, "Validation_dataset", f"shuffle_{i}")

        os.makedirs(train_out_dir, exist_ok=True)
        os.makedirs(val_out_dir, exist_ok=True)

        # Copy training data
        for src in train_paths:
            dst = os.path.join(train_out_dir, os.path.basename(src))
            shutil.copy(src, dst)

        # Copy validation data
        for src in val_paths:
            dst = os.path.join(val_out_dir, os.path.basename(src))
            shutil.copy(src, dst)


if __name__ == "__main__":
    cfg = load_config("config.yaml")

    base_path = cfg_get(cfg, ["base_path"], None)
    if not base_path:
        raise KeyError("config.yaml must define: base_path")

    repeats = int(cfg_get(cfg, ["repeats"], 3))
    train_ratio = float(cfg_get(cfg, ["dataset_shuffle", "train_ratio"], 0.8))
    out_root = cfg_get(cfg, ["dataset_shuffle", "out_root"], None) or base_path

    seed = cfg_get(cfg, ["dataset_shuffle", "seed"], None)
    if seed is not None:
        seed = int(seed)

    generate_d3sets(
        source_dir=base_path,
        out_root=out_root,
        repeats=repeats,
        train_ratio=train_ratio,
        seed=seed,
    )
