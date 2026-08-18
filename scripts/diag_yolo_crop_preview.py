# -*- coding: utf-8 -*-
"""生成 YOLO 裁剪结果预览图: 原图 vs 裁剪后(画稳定框) 并排保存.

输出: outputs/figures/yolo_crop_<video>.png  (每个视频一张, 中央帧)
用法:  D:/Anaconda3/envs/deepgroomnet3/python.exe scripts/diag_yolo_crop_preview.py
"""
import os
import sys
import json
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["Locomote", "Rear", "FaceGroom", "BodyGroom", "background"]
KEY = "superanimal"


def draw_preview(v, box, out_dir, label_name):
    cap = cv2.VideoCapture(v["video"])
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ret, fr = cap.read()
    cap.release()
    if not ret:
        print(f"  skip {v['mouse']}: read fail")
        return
    H, W = fr.shape[:2]
    img = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

    x1, y1, x2, y2 = [int(round(b)) for b in box]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    cropped = img[y1c:y2c, x1c:x2c] if (x2c - x1c >= 4 and y2c - y1c >= 4) else img

    # 原图画框
    orig = img.copy()
    cv2.rectangle(orig, (x1c, y1c), (x2c, y2c), (255, 0, 0), 2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(orig)
    ax[0].set_title(f"Original ({W}x{H})  label={label_name}")
    ax[0].set_axis_off()
    ax[1].imshow(cropped)
    ax[1].set_title(f"YOLO crop {cropped.shape[1]}x{cropped.shape[0]}")
    ax[1].set_axis_off()
    plt.tight_layout()
    key = os.path.splitext(os.path.basename(v["video"]))[0][:30]
    out = os.path.join(out_dir, f"yolo_crop_{v['mouse']}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "config.yaml"), encoding="utf-8"))
    boxes = json.load(open(os.path.join(cfg["DATA"]["meta_root"], "yolo_boxes.json"), encoding="utf-8"))
    videos = json.load(open(os.path.join(cfg["DATA"]["meta_root"], "videos.json"), encoding="utf-8"))
    out_dir = os.path.join(ROOT, "outputs", "figures", "yolo_crop")
    os.makedirs(out_dir, exist_ok=True)

    # 每个数据源挑 2 个视频
    seen = set()
    picked = []
    for v in videos:
        if v["source"] not in seen:
            seen.add(v["source"])
            picked.append(v)
        if len(picked) == len(set(v["source"] for v in videos)):
            pass
    # 至少取 4 个, 跨数据源
    picked = []
    by_src = {}
    for v in videos:
        by_src.setdefault(v["source"], []).append(v)
    for src, vs in by_src.items():
        picked += vs[:2]

    print(f"生成 {len(picked)} 个视频的裁剪预览 -> {out_dir}")
    for v in picked:
        key = os.path.splitext(os.path.basename(v["video"]))[0]
        box = boxes.get(key)
        # 取中心帧标签
        lab = np.asarray(v["labels"])[int(v["n_frames"]) // 2]
        name = CLASS_NAMES[int(lab)] if int(lab) < len(CLASS_NAMES) else "?"
        if box is None:
            print(f"  skip {v['mouse']}: no yolo box")
            continue
        draw_preview(v, box, out_dir, name)


if __name__ == "__main__":
    main()
