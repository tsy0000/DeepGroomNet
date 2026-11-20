import copy
import os
from pathlib import Path
import yaml

CLASS_NAMES = ("Locomote", "Rear", "FaceGroom", "BodyGroom", "background")
BACKGROUND_INDEX = 4

DEFAULTS = {
    "CLASSES": {"names": list(CLASS_NAMES), "background_index": BACKGROUND_INDEX},
    "CLIP": {"num_input_frames": 32, "target_size": 224, "norm_mean": 0.45,
             "norm_std": 0.225, "yolo_crop": False},
    "MODEL": {"backbone": "tadaconvnextv2_tiny", "cpam": True, "stsa": True,
              "pretrained": None, "checkpoint": None},
    "INFERENCE": {"stride": 8, "batch_size": 8, "device": "cuda"},
    "SGATF": {"alpha_max": 1.0, "max_run": 12, "consensus_floor": 0.35,
              "consensus_span": 0.50},
    "POSTPROCESS": {"smooth_window": 0, "min_bout_frames": 0,
                    "thresholds": [0.5, 0.5, 0.5, 0.5, 0.5]},
    "PREPROCESS": {"gray_mean": 146.2030487060547, "gray_std": 49.36723709106445,
                   "yolo_weights": None, "yolo_stride": 30, "yolo_margin": 0.2,
                   "yolo_conf": 0.3},
    "PATHS": {"work_dir": "runs"},
}


def _merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None, **overrides):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as handle:
            _merge(cfg, yaml.safe_load(handle) or {})
    if overrides:
        _merge(cfg, overrides)
    return cfg


def resolve_path(value, base_dir=None):
    if value is None:
        return None
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if os.path.isabs(value) or base_dir is None:
        return value
    return str((Path(base_dir) / value).resolve())


def require_file(value, name):
    path = Path(value) if value else None
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{name} not found: {value!r}. Pass it on the command line or in the config.")
    return path
