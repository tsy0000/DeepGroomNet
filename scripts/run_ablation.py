# -*- coding: utf-8 -*-
"""
Run 3 ablations sequentially: (1) baseline (no STSA/CPAM), (2) CPAM only,
(3) STSA + CPAM. Each writes to its own outputs/<...>_<tag> dirs.

Usage:  python scripts/run_ablation.py [--subset-videos N] [--epochs E] [--lp-epochs LE]
"""
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
TRAIN = os.path.join(ROOT, "scripts", "train.py")

CONFIGS = [
    {"tag": "none",   "stsa": 0, "cpam": 0, "label": "baseline (no STSA, no CPAM)"},
    {"tag": "cpam",   "stsa": 0, "cpam": 1, "label": "CPAM only"},
    {"tag": "both",   "stsa": 1, "cpam": 1, "label": "STSA + CPAM (full)"},
]


def main():
    args = sys.argv[1:]
    for cfg in CONFIGS:
        print("\n" + "=" * 70)
        print(f"RUNNING: {cfg['label']}  (tag={cfg['tag']})")
        print("=" * 70)
        cmd = [PY, TRAIN,
               "--stsa", str(cfg["stsa"]),
               "--cpam", str(cfg["cpam"]),
               "--tag", cfg["tag"]] + args
        print("CMD:", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[FAIL] {cfg['label']} exited with code {rc}")
            sys.exit(rc)
    print("\nAll ablations finished.")


if __name__ == "__main__":
    main()