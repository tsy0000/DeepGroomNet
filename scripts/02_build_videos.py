"""DeepGroomNet_v3 - 数据准备: 构建帧级标签视频清单 + 按小鼠划分。

输出 videos.json: 每个视频的 {video, mouse, source, n_frames, labels(int array)}
  + split.json: 按小鼠 train/val/test 划分
对齐 DeepEthogram: 帧级标签, 按视频/小鼠划分 (0.8/0.2/0)
"""
import os
import sys
import json
import random
import numpy as np
import pandas as pd
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CLASS_COLS = ["Locomote", "Rear", "FaceGroom", "BodyGroom", "background"]
BG_IDX = len(CLASS_COLS) - 1


def load_labels(csv_path):
    """读逐帧 one-hot CSV -> 每帧类索引 [N]。未知列归 background。"""
    df = pd.read_csv(csv_path)
    idx_map = {c: i for i, c in enumerate(CLASS_COLS)}
    labels = np.full(len(df), BG_IDX, dtype=np.int64)
    for i in range(len(df)):
        active = [idx_map[c] for c in CLASS_COLS[:-1]
                  if c in df.columns and df[c].iloc[i] == 1]
        labels[i] = active[0] if active else BG_IDX
    return labels


def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def build_from_root(root, source):
    """扫描一个根目录的所有小鼠视频, 返回 videos 列表。"""
    videos = []
    if not os.path.isdir(root):
        print(f"  跳过不存在: {root}")
        return videos
    for mouse in sorted(os.listdir(root)):
        mdir = os.path.join(root, mouse)
        if not os.path.isdir(mdir):
            continue
        label_csv, video_mp4 = None, None
        for f in os.listdir(mdir):
            if f.endswith(".csv") and ("label" in f.lower() or f.startswith("labels")):
                if label_csv is None or "labels2" in f:
                    label_csv = os.path.join(mdir, f)
            elif f.endswith(".mp4"):
                video_mp4 = os.path.join(mdir, f)
        if not (label_csv and video_mp4):
            print(f"  跳过 {mouse}: 缺标签/视频")
            continue
        try:
            labels = load_labels(label_csv)
            n_frames = get_frames(video_mp4)
            if len(labels) > n_frames:
                labels = labels[:n_frames]
        except Exception as e:
            print(f"  跳过 {mouse}: {e}")
            continue
        videos.append({"video": video_mp4, "mouse": mouse, "source": source,
                       "n_frames": len(labels), "labels": labels.tolist()})
        print(f"  {mouse}: {len(labels)} 帧, {dict(zip(CLASS_COLS,[int((labels==i).sum()) for i in range(len(CLASS_COLS))]))}")
    return videos


def main():
    roots = ["E:/tsy/downloaddatasets", "E:/tsy/datasetssap3"]
    out_dir = os.path.join(ROOT, "outputs", "meta")
    os.makedirs(out_dir, exist_ok=True)

    all_videos = []
    for root in roots:
        src = os.path.basename(root)
        all_videos.extend(build_from_root(root, src))

    # 按小鼠划分 (train/val/test = 0.7/0.2/0.1, 小鼠不重叠, 7:2:1)
    random.seed(42)
    mice = sorted(set(v["mouse"] for v in all_videos))
    random.shuffle(mice)
    n_val = int(len(mice) * 0.2)
    n_test = int(len(mice) * 0.1)
    val_mice = set(mice[:n_val])
    test_mice = set(mice[n_val:n_val + n_test])
    train_v = [v for v in all_videos
               if v["mouse"] not in val_mice and v["mouse"] not in test_mice]
    val_v = [v for v in all_videos if v["mouse"] in val_mice]
    test_v = [v for v in all_videos if v["mouse"] in test_mice]
    print(f"\n总计 {len(all_videos)} 视频, {len(mice)} 小鼠")
    print(f"train: {len(train_v)} / val: {len(val_v)} / test: {len(test_v)} (按小鼠不重叠)")

    # 保存
    json.dump(all_videos, open(os.path.join(out_dir, "videos.json"), "w"))
    json.dump({"train": train_v, "val": val_v, "test": test_v},
              open(os.path.join(out_dir, "split.json"), "w"))
    print("-> outputs/meta/videos.json + split.json")


if __name__ == "__main__":
    main()
