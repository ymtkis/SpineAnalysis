# config_utils.py
import os
from typing import Any, Dict, Iterable

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML config. If relative path is given, resolve relative to this file (config_utils.py) location.
    """
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, config_path)

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must be a mapping (YAML dict) at top-level.")
    return cfg


def cfg_get(cfg: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    """
    Nested get: cfg_get(cfg, ["tile", "size"], 256)
    """
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
