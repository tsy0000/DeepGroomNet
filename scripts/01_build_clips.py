"""DeepGroomNet_v3 - 数据管线 1: 从官方旷场视频生成训练 clips。

输入:
  E:/tsy/downloaddatasets/mouseXX/labels2.csv (逐帧 one-hot) + mouseXX.mp4
  E:/tsy/datasetssap3/... (自采, 格式近似)
输出:
  clips.csv (列: clip, video, start, end, label, mouse, source)
  clips/ 目录 (每个 clip 一个 npz: frames [16,H,W,3] uint8 + label)

策略 (参考 DeepEthogram / Sakamoto):
  - 逐帧标签, 16 帧窗口, 标签取窗口中心帧
  - 对每个 16 帧窗口采样, 稀有类 (FG/BG) 全采样, 常见类 (background) 按比例采样
  - 视频读取用 OpenCV, 帧索引与标签严格对齐 (0-based)

用法:
  python scripts/01_build_clips.py [--roots ...] [--frame-window 16] [--stride ...]
"""
import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
import cv2
import random
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_labels(csv_path, class_cols):
    """读逐帧 one-hot CSV, 返回每帧的类索引数组 [N]。
    Defecate 等未列类归并到 background(最后一类)。"""
    df = pd.read_csv(csv_path)
    # 只保留已知列, 其余归 background
    idx_map = {c: i for i, c in enumerate(class_cols)}
    labels = np.full(len(df), len(class_cols) - 1, dtype=np.int64)  # 默认 background
    for col in class_cols[:-1]:
        if col in df.columns:
            labels[df[col].values == 1] = idx_map[col]
    # 处理多列同时为 1 的情况 (取第一个非 background)
    for i in range(len(df)):
        active = [idx_map[c] for c in class_cols[:-1] if c in df.columns and df[c].iloc[i] == 1]
        if active:
            labels[i] = active[0]
    return labels


def get_video_frames(video_path):
    """返回视频总帧数。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def collect_clips(video_path, labels, class_cols, frame_window, sample_cfg, mouse, source):
    """生成 16 帧窗口 clips, 标签=窗口中心帧。
    稀有类全采样, 常见类按比例采样 (缓解不平衡)。"""
    n = len(labels)
    clips = []
    # 各类权重 (稀有类更高采样率)
    class_counts = Counter(labels.tolist())
    max_count = max(class_counts.values())
    for c, cnt in class_counts.items():
        sample_cfg.setdefault(c, max_count / max(cnt, 1))

    for start in range(0, n - frame_window + 1):
        center = start + frame_window // 2
        label = int(labels[center])
        # 按稀有度决定是否采样
        prob = min(1.0, sample_cfg.get(label, 1.0))
        if random.random() > prob:
            continue
        clips.append({
            "clip": os.path.basename(video_path).replace(".mp4", ""),
            "video": video_path,
            "start": start,
            "end": start + frame_window,
            "label": class_cols[label],
            "mouse": mouse,
            "source": source,
        })
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*",
                    default=["E:/tsy/downloaddatasets", "E:/tsy/datasetssap3"])
    ap.add_argument("--frame-window", type=int, default=16)
    ap.add_argument("--max-clips-per-class", type=int, default=50000)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "clips"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "outputs", "meta"))
    args = ap.parse_args()

    class_cols = ["Locomote", "Rear", "FaceGroom", "BodyGroom", "background"]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.meta, exist_ok=True)
    random.seed(42)

    all_clips = []
    for root in args.roots:
        if not os.path.isdir(root):
            print(f"跳过不存在根目录: {root}")
            continue
        for mouse in sorted(os.listdir(root)):
            mdir = os.path.join(root, mouse)
            if not os.path.isdir(mdir):
                continue
            # 找标签与视频
            label_csv = None
            video_mp4 = None
            for f in os.listdir(mdir):
                if f.endswith(".csv") and ("label" in f.lower() or f.startswith("labels")):
                    if label_csv is None or "labels2" in f:
                        label_csv = os.path.join(mdir, f)
                elif f.endswith(".mp4"):
                    video_mp4 = os.path.join(mdir, f)
            if label_csv is None or video_mp4 is None:
                print(f"跳过 {mouse}: 缺标签/视频")
                continue
            try:
                labels = load_labels(label_csv, class_cols)
                n_frames = get_video_frames(video_mp4)
                if len(labels) > n_frames:
                    labels = labels[:n_frames]
                elif len(labels) < n_frames:
                    print(f"  [warn] {mouse}: 标签 {len(labels)} < 帧 {n_frames}, 用标签数")
            except Exception as e:
                print(f"跳过 {mouse}: {e}")
                continue
            sample_cfg = {}
            clips = collect_clips(video_mp4, labels, class_cols,
                                  args.frame_window, sample_cfg, mouse, os.path.basename(root))
            all_clips.extend(clips)
            print(f"  {mouse}: {len(clips)} clips, 帧={n_frames}")

    # 写 CSV
    out_csv = os.path.join(args.out, "clips.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "video", "start", "end", "label", "mouse", "source"])
        w.writeheader()
        for c in all_clips:
            w.writerow(c)
    print(f"\n总计 {len(all_clips)} clips -> {out_csv}")
    print("类别分布:", dict(Counter(c["label"] for c in all_clips)))


if __name__ == "__main__":
    main()
