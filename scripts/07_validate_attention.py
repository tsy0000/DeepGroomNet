"""定量验证 MotionStatAttention（运动统计点位注意力）在真实 FG/BG 批量上的判别分离度。

用法（对照模块可加 --attention 切换）：
  python scripts/07_validate_attention.py [--attention motion_stat] [--n 60] [--device cuda]
输出：每 clip 的 (fast_mean, gate_mean, concentration)，FG vs BG 的 Cohen d。
目标：模块级判别信号 |d| >= 1.0（与输入级 gini d=-1.23 / 频带 d=0.98 对齐），
      证明"点位注意力"抓的是判别像素而非几何位置。
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.clip_sampler import sample_frames
from data.dataset import ClipDataset
from models.attention import MotionStatAttention
from models.backbones.tadaconvnext import build_model

CLIPS_CSV = r"D:\DeepGroomNet_v2\outputs\clips\clips.csv"
CLIP_DIR = r"D:\DeepGroomNet_v2\outputs\clips"


def load_config(attention):
    cfg = {
        "MODEL": {"backbone": "TAdaConvNeXtV2_Tiny", "num_classes": 5,
                  "attention": attention,
                  "attention_reductions": {"motion": 16},
                  "attention_positions": [0, 1, 2, 3], "dropout": 0.4,
                  "init_weights": "C:/Users/WIN10/PycharmProjects/mouseaction/DeepBehavior/TAdaConv/tadaconvnextv2_tiny_in1k_k400_79.6.pyth"},
        "CLASSES": {"openfield_keep": ["Locomote", "Rear", "FaceGroom", "BodyGroom", "background"]},
    }
    return cfg


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attention", default="motion_stat")
    ap.add_argument("--n", type=int, default=60, help="每类 clip 数")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = torch.device(args.device)
    cfg = load_config(args.attention)
    model = build_model(cfg).to(device).eval()

    df = pd.read_csv(CLIPS_CSV)
    clips = {l: df[df["label"] == l]["clip"].tolist() for l in ("FaceGroom", "BodyGroom")}
    stat_modules = [m for attn in model.attentions
                    for m in ([attn] if not isinstance(attn, torch.nn.Sequential) else list(attn))
                    if isinstance(m, MotionStatAttention)]
    assert stat_modules, "模型中无 MotionStatAttention"
    for m in stat_modules:
        m._capture = True

    rows = []
    with torch.no_grad():
        for label, clip_list in clips.items():
            random.shuffle(clip_list)
            for clip in clip_list[: args.n]:
                path = os.path.join(CLIP_DIR, clip)
                frames = np.stack(ClipDataset._read_clipfile(path))
                idx = sample_frames(0, len(frames), 16)
                arr = frames[idx].astype(np.float32) / 255.0
                x_arr = arr
                if int(cfg["CLIP"].get("motion_channel", 0)):
                    from data.dataset import add_motion_channel
                    x_arr = add_motion_channel(arr)
                x = torch.from_numpy(x_arr.transpose(3, 0, 1, 2)).unsqueeze(0).to(device)
                model(x)
                row = {"label": label, "clip": clip}
                for i, m in enumerate(stat_modules):
                    r_n, mu, sg, c, w_s, m1 = m._capture_cache
                    row[f"s{i}_fast"] = r_n[0].mean().item()
                    row[f"s{i}_gate"] = w_s[0, 0].mean().item()
                    row[f"s{i}_conc"] = c.mean().item()
                rows.append(row)
    dfr = pd.DataFrame(rows)
    fg, bg = dfr[dfr.label == "FaceGroom"], dfr[dfr.label == "BodyGroom"]
    print(f"[validate] attention={args.attention}  n={len(fg)}/{len(bg)}  device={device}")
    for i in range(len(stat_modules)):
        for col in (f"s{i}_fast", f"s{i}_gate", f"s{i}_conc"):
            d = cohens_d(fg[col], bg[col])
            print(f"  {col:>8s}: FG={fg[col].mean():.4f} vs BG={bg[col].mean():.4f}  Cohen d={d:+.2f}")
    out = os.path.join(os.path.dirname(CLIPS_CSV), "attention_validate.csv")
    dfr.to_csv(out, index=False)
    print(f"[validate] saved -> {out}")


if __name__ == "__main__":
    main()
