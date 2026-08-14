"""
02 预裁剪 clip 到磁盘（一次性预处理，发表级 I/O 优化）。

对 openfield 每个视频：
  1) 按帧级标签生成 45 帧窗口（确定性滑窗，跨边界抖动为训练期增强，见 dataset.py）
  2) YOLO 检测窗口首/中/末帧，取并集框 + 1.2 倍外扩（clip 一致静态裁剪，与原项目策略一致）
     - 检测置信度阈值取 config.CLIP.yolo_conf（默认 0.1，避免 conf=0 的垃圾框污染裁剪）
     - 无检测时回退：全帧中心方形裁剪（小鼠通常位于画面中央）
  3) 裁剪并 resize 到 target_size，h264 编码为小 mp4
输出：
  outputs/clips/clips.csv      (clip_path, label, mouse)
  outputs/clips/<mouse>_<start>_<label>.mp4

用法：python scripts/02_gen_clips.py [--workers 2] [--limit N] [--offset N] [--overwrite]
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from data.clip_sampler import extract_uniform_clips, extract_sliding_clips, normalize_label_columns

_CFG = None
_YOLO = None
try:
    import torch as _torch
    _YOLO_DEVICE = "0" if _torch.cuda.is_available() else "cpu"
except ImportError:
    _YOLO_DEVICE = "cpu"


def _get_yolo(cfg):
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        w = cfg["CLIP"].get("yolo_weights", "")
        if not w or not os.path.exists(w):
            w = r"C:\Users\WIN10\PycharmProjects\mouseaction\DeepBehavior\yolo11n.pt"
        _YOLO = YOLO(w)
    return _YOLO


def _union_box(boxes, size, expand=1.2):
    if not boxes:
        return None
    xs1 = min(b[0] for b in boxes); ys1 = min(b[1] for b in boxes)
    xs2 = max(b[2] for b in boxes); ys2 = max(b[3] for b in boxes)
    w, h = xs2 - xs1, ys2 - ys1
    cx, cy = (xs1 + xs2) / 2, (ys1 + ys2) / 2
    nw, nh = w * expand, h * expand
    x1 = max(0, int(cx - nw / 2)); y1 = max(0, int(cy - nh / 2))
    x2 = min(size[0], int(cx + nw / 2)); y2 = min(size[1], int(cy + nh / 2))
    return (x1, y1, x2, y2)


def _center_box(size):
    """无检测时回退：以画面中心为心的最大正方形（小鼠通常居中）。"""
    w, h = size
    s = min(w, h)
    x1 = (w - s) // 2
    y1 = (h - s) // 2
    return (x1, y1, x1 + s, y1 + s)


def _crop_resize(frames, box, tgt):
    out = []
    for fr in frames:
        if box is None:
            c = fr
        else:
            x1, y1, x2, y2 = box
            c = fr[y1:y2, x1:x2]
        out.append(cv2.resize(c, (tgt, tgt), interpolation=cv2.INTER_AREA))
    return out


def extract_one(args):
    video, start, end, label, mouse, out_dir, cfg, source = args
    tgt = int(cfg["CLIP"]["target_size"])
    conf = float(cfg["CLIP"].get("yolo_conf", 0.1))
    exp = float(cfg["CLIP"].get("yolo_expand", 1.2))
    tag = "sl_" if source == "sliding" else ""
    out_path = os.path.join(out_dir, f"{mouse}_{start}_{tag}{label}.mp4")

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if end > total:
        end = total
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(end - start):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if len(frames) < 8:
        return None

    model = _get_yolo(cfg)
    det_idx = [0, len(frames) // 2, len(frames) - 1]
    boxes = []
    for i in det_idx:
        r = model.predict(frames[i], conf=conf, verbose=False, device=_YOLO_DEVICE)[0]
        if r.boxes is not None and len(r.boxes):
            b = r.boxes.xyxy[0].cpu().numpy().astype(int)
            boxes.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
    box = _union_box(boxes, (w, h), exp) or _center_box((w, h))

    cropped = _crop_resize(frames, box, tgt)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (tgt, tgt))
    for fr in cropped:
        vw.write(fr)
    vw.release()
    return {"clip": out_path, "label": label, "mouse": mouse, "source": source}


def _extract_one_retry(args, tries=3):
    """单任务重试：缓解 Windows 下 OpenCV 偶发分配失败（此前 OOM 崩溃点）。"""
    last = None
    for i in range(tries):
        try:
            return extract_one(args)
        except Exception as e:
            last = e
            time.sleep(2.0 * (i + 1))
    print(f"  [warn] 窗口失败（重试{tries}次后）: {args[1]}-{args[2]} {last}", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2,
                    help="并行进程数（Windows 下 4+ 易触发 OpenCV 内存分配失败，默认 2）")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个视频（调试/分块）")
    ap.add_argument("--offset", type=int, default=0, help="跳过前 N 个视频（断点续传）")
    ap.add_argument("--overwrite", action="store_true",
                    help="删除已有 clips.csv 与 clip 文件后全量重建（修复旧版抖动 bug 后必须使用）")
    args = ap.parse_args()

    global _CFG
    _CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "config.yaml"), encoding="utf-8"))
    cfg = _CFG
    meta_csv = os.path.join(cfg["DATA"]["meta_root"], "openfield_meta.csv")
    out_dir = cfg["DATA"]["clip_output_root"]
    os.makedirs(out_dir, exist_ok=True)

    clips_csv = os.path.join(out_dir, "clips.csv")
    if args.overwrite:
        import glob as _glob
        n_del = 0
        for f in _glob.glob(os.path.join(out_dir, "*.mp4")):
            try:
                os.remove(f)
                n_del += 1
            except OSError:
                pass
        if os.path.exists(clips_csv):
            os.remove(clips_csv)
        print(f"[02] --overwrite：已删除 {n_del} 个旧 clip 与 clips.csv（旧版抖动逻辑产出的数据不可用）")

    meta = pd.read_csv(meta_csv)
    if args.offset:
        meta = meta.iloc[args.offset:]
    if args.limit:
        meta = meta.head(args.limit)

    tasks = []
    for _, m in meta.iterrows():
        lab = pd.read_csv(m["labels"])
        lab = lab.drop(columns=[lab.columns[0]] if lab.columns[0].startswith("Unnamed") else [])
        lab = normalize_label_columns(lab)
        mouse = os.path.basename(os.path.dirname(m["video"]))
        unis = extract_uniform_clips(m["video"], lab, cfg)
        for (s, e, label) in unis:
            tasks.append((m["video"], s, e, label, mouse, out_dir, cfg, "uniform"))
        n_slide = int(len(unis) * float(cfg["CLIP"].get("sliding_windows_per_uniform", 1)))
        if n_slide > 0:
            seed = int(cfg["TRAIN"].get("seed", 42))
            for (s, e, label) in extract_sliding_clips(m["video"], lab, cfg, n_slide, seed=seed):
                tasks.append((m["video"], s, e, label, mouse, out_dir, cfg, "sliding"))

    print(f"[02] windows={len(tasks)} workers={args.workers} (offset={args.offset})")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_extract_one_retry, tasks, chunksize=32):
            if r is not None:
                results.append(r)

    csv_path = os.path.join(out_dir, "clips.csv")
    existing = []
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r.setdefault("source", "uniform")
                existing.append(r)
    seen = {(r["clip"]) for r in existing}
    merged = existing + [r for r in results if r["clip"] not in seen]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "label", "mouse", "source"])
        w.writeheader()
        w.writerows(merged)
    from collections import Counter
    print(f"[02] +{len(results)} clips, total {len(merged)} -> {csv_path}")
    print("  分布:", dict(Counter(r["label"] for r in merged)))


if __name__ == "__main__":
    main()
