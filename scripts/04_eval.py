"""04 推理评估。python scripts/04_eval.py --video ... [--syllables ...] [--out ...]"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import infer

if __name__ == "__main__":
    infer.main()
