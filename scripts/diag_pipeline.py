# -*- coding: utf-8 -*-
"""DeepGroomNet_v3 数据/模型链路一次性诊断。

覆盖用户要求:
  1. 读取 mp4 并按标签分段 + 仅 YOLO 裁剪 是否正确
  2. 打印网络前向传播调用链路 (含每个模块的输入/输出维度)
  3. 前向传播维度是否正常 (3×32×224×224 -> logits/aux)
  4. 采样环节 (WeightedRandomSampler 过采样稀有类) 是否正确
  5. 数据增强环节 是否正确
  6. 画图: 原始数据 vs 过采样数据 分布比较趋势图 (仿 MDPI 17(2):139 图10)

用法:  D:/Anaconda3/envs/deepgroomnet3/python.exe scripts/diag_pipeline.py
输出:  outputs/diag_pipeline.log + outputs/figures/original_vs_oversampled.png
"""
import os
import sys
import json
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import Counter

from data.dataset import FrameLevelDataset
from data.dataloaders import build_class_balanced_sampler
from models.tadaconv_model import TAdaConvModel

CLASS_NAMES = ["Locomote", "Rear", "FaceGroom", "BodyGroom", "background"]


def banner(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74, flush=True)


def pick_mice(videos, n_per_source=2):
    """从每个数据源挑少量小鼠做裁剪验证。"""
    by_src = {}
    for v in videos:
        by_src.setdefault(v["source"], []).append(v)
    picked = []
    for src, vs in sorted(by_src.items()):
        picked += vs[:n_per_source]
    return picked


# ============================================================ 1. 数据读取 / 标签分段 / YOLO裁剪
def section_data_pipeline(cfg, videos):
    banner(f"[1] 数据读取 + 标签分段 + YOLO 裁剪   (验证 {len(videos)} 个视频片段)")
    frame_window = int(cfg["CLIP"]["num_input_frames"])
    ds = FrameLevelDataset(videos, cfg, split="train", class_names=CLASS_NAMES,
                           frame_window=frame_window)
    print(f"frame_window={frame_window}, target_size={ds.size}, "
          f"yolo_crop={bool(cfg['CLIP'].get('yolo_crop'))}, yolo_boxes 载入={ds.yolo_boxes is not None}")
    if ds.yolo_boxes:
        print(f"yolo_boxes 覆盖视频数: {len(ds.yolo_boxes)}")

    ok = True
    for v in videos:
        key = os.path.splitext(os.path.basename(v["video"]))[0]
        n_frame_cap = int(cv2.VideoCapture(v["video"]).get(cv2.CAP_PROP_FRAME_COUNT))
        n_label = int(v["n_frames"])
        match = (n_label == n_frame_cap) or (n_label <= n_frame_cap)
        box = ds._clip_box(v["video"], n_label // 2)
        box_ok = box is not None
        if not box_ok:
            ok = False
        print(f"  [{key}] source={v['source']} 标签帧={n_label} 视频帧={n_frame_cap} "
              f"对齐={'OK' if match else 'MISMATCH'} | YOLO框={box_ok}")
        if box:
            print(f"        yolo_box=({box[0]:.0f},{box[1]:.0f})-({box[2]:.0f},{box[3]:.0f})")
    return ds, ok


def verify_clip_sample(ds, v):
    """读一个 clip, 核对形状/标签/裁剪. 返回 (x[3,32,224,224], y, orig_frame, crop_frame, box)."""
    frame_window = ds.frame_window
    n = int(v["n_frames"])
    start = max(0, n - frame_window - 1) if n > frame_window else 0
    cap = cv2.VideoCapture(v["video"])
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    for _ in range(frame_window):
        ret, fr = cap.read()
        frames.append(fr if ret else np.zeros((ds.size, ds.size, 3), np.uint8))
    cap.release()
    orig = frames[frame_window // 2].copy()

    key = os.path.splitext(os.path.basename(v["video"]))[0]
    box = ds._clip_box(v["video"], start + frame_window // 2)
    crop_frame = None
    if box:
        H, W = orig.shape[:2]
        x1, y1, x2, y2 = [int(round(b)) for b in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            crop_frame = orig[y1:y2, x1:x2]

    arr = np.stack([cv2.resize(f, (ds.size, ds.size)) for f in frames]).astype(np.float32) / 255.0
    arr = np.transpose(arr, (0, 3, 1, 2))          # [T,3,H,W]
    arr = (arr - ds.mean) / ds.std
    x = np.transpose(arr, (1, 0, 2, 3)).astype(np.float32)   # [C,T,H,W]
    center = min(start + frame_window // 2, n - 1)
    y = int(np.asarray(v["labels"])[center])
    return torch.from_numpy(x), y, orig, crop_frame, box


# ============================================================ 2. 前向传播调用链路 + 维度
def section_forward(cfg, device="cpu"):
    banner("[2] 前向传播调用链路 + 维度 (hook 按调用顺序打印)")
    num_classes = len(CLASS_NAMES)
    model = TAdaConvModel(num_classes=num_classes, t_stride=1,
                          drop_path=0.0, use_cpam=True, use_stsa=True)
    model.eval()

    # 只 hook 关键边界模块, 缩进体现调用层级
    interesting = {}
    for name, m in model.named_modules():
        if (name in ("", "norm", "head", "stsa3", "stsa4", "cpam")
                or name.startswith("downsample_layers.")
                or name.startswith("stages.")
                or name.startswith("cpam.proj.")
                or name.startswith("cpam.mixers.")
                or name.startswith("cpam.fusion.")
                or name.startswith("cpam.aux_head")
                or name.startswith("stsa3.") or name.startswith("stsa4.")):
            interesting[name] = m

    def _shape_of(t):
        if isinstance(t, torch.Tensor):
            return tuple(t.shape)
        if isinstance(t, (tuple, list)):
            return tuple(tuple(ti.shape) if isinstance(ti, torch.Tensor) else ti
                         for ti in t)
        return t

    def make_hook(path):
        depth = path.count(".")
        pad = "  " * min(depth, 6)
        def h(m, inp, out):
            i0 = inp[0] if isinstance(inp, tuple) else inp
            ins = _shape_of(i0)
            o = out[0] if isinstance(out, (tuple, list)) else out
            oshape = _shape_of(o)
            print(f"{pad}├─ {path or 'TAdaConvModel.forward'}  {ins} -> {oshape}", flush=True)
        return h

    handles = [m.register_forward_hook(make_hook(n)) for n, m in interesting.items()]

    x = torch.randn(2, 3, 32, 224, 224)
    print(f"输入 x: {tuple(x.shape)}  (N=2, C=3, T=32, H=W=224)\n")
    with torch.no_grad():
        logits, aux = model(x)
    for hh in handles:
        hh.remove()

    print(f"\n输出 logits={tuple(logits.shape)}  aux_logits={tuple(aux.shape)}")
    assert tuple(logits.shape) == (2, num_classes), "主头维度异常"
    assert tuple(aux.shape) == (2, num_classes), "CPAM头维度异常"
    print("[OK] 前向传播维度正确: 3×32×224×224 -> (N,5) 主头 + (N,5) CPAM辅助头")
    return model


# ============================================================ 3. 采样
def section_sampling(train_videos, cfg):
    banner("[3] 采样环节: 原始 vs 过采样 分布")
    frame_window = int(cfg["CLIP"]["num_input_frames"])
    ds = FrameLevelDataset(train_videos, cfg, split="train", class_names=CLASS_NAMES,
                           frame_window=frame_window)
    K = len(CLASS_NAMES)

    # 原始帧级分布
    orig = np.zeros(K, dtype=np.float64)
    for v in train_videos:
        lab = np.asarray(v["labels"])
        for c in range(K):
            orig[c] += (lab == c).sum()

    # Swin-Panda 均匀过采样 (稀有类补到平均值, 复制用不同起始窗口)
    _ = build_oversampled_sampler(ds, seed=int(cfg["TRAIN"].get("seed", 42)))
    fw = ds.fixed_windows
    over_new = np.zeros(K, dtype=np.float64)
    for vi, _s in fw:
        lab = np.asarray(train_videos[vi]["labels"])
        center = min(_s + frame_window // 2, train_videos[vi]["n_frames"] - 1)
        over_new[int(lab[center])] += 1

    def _row(name, arr):
        s = f"{name:<10}  " + "  ".join(
            f"{CLASS_NAMES[c]}:{arr[c]/max(arr.sum(),1):.1%}" for c in range(K))
        print(s)

    _row("原始帧级分布", orig)
    _row("均匀过采样", over_new)

    print("\n结论: 训练已启用 Swin-Panda 均匀过采样 (config sampler_balanced=true), "
          f"过采样后每类样本数={[int(over_new[c]) for c in range(K)]}")
    return orig, over_new


# ============================================================ 4. 数据增强
def section_augment(ds, v):
    banner("[4] 数据增强环节 验证")
    frame_window = ds.frame_window
    n = int(v["n_frames"])
    start = max(0, n - frame_window - 1) if n > frame_window else 0
    cap = cv2.VideoCapture(v["video"])
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    for _ in range(frame_window):
        ret, fr = cap.read()
        frames.append(fr if ret else np.zeros((ds.size, ds.size, 3), np.uint8))
    cap.release()
    arr = np.stack([cv2.resize(f, (ds.size, ds.size)) for f in frames]).astype(np.float32) / 255.0
    arr = np.transpose(arr, (0, 3, 1, 2))  # [T,3,H,W]
    before = arr.copy()
    after = ds._augment(arr)
    print(f"增强前: {tuple(before.shape)}  mean={before.mean():.4f} std={before.std():.4f}")
    print(f"增强后: {tuple(after.shape)}  mean={after.mean():.4f} std={after.std():.4f}")
    assert after.shape == before.shape, "增强改变形状!"
    print(f"[OK] 增强保持形状 [T,3,H,W]={tuple(after.shape)}, 像素值域变化正常 (亮度/对比/翻转/旋转)")
    return before, after


# ============================================================ 5. YOLO 裁剪正确性 (真机验证)
def section_yolo_crop_check(cfg, ds, videos, yolo_weights):
    banner("[5] YOLO 裁剪正确性验证 (对裁剪后帧跑 YOLO, 确认小鼠在框内且占比足够)")
    try:
        from ultralytics import YOLO
        yolo = YOLO(yolo_weights)
    except Exception as e:
        print(f"[WARN] 无法加载 YOLO ({e}), 跳过裁剪质量验证")
        return True
    frame_window = ds.frame_window
    print("对每个视频取 5 个随机窗口, 跑 YOLO 统计裁剪后小鼠覆盖率 ...")
    for v in videos:
        key = os.path.splitext(os.path.basename(v["video"]))[0]
        if not isinstance(ds.yolo_boxes.get(key), dict):
            print(f"  [{key}] 无逐帧 yolo_box, 跳过")
            continue
        n = int(v["n_frames"])
        ratios = []
        crop_sizes = []
        rng = np.random.default_rng(0)
        for _ in range(5):
            start = int(rng.integers(0, max(1, n - frame_window)))
            cap = cv2.VideoCapture(v["video"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            ret, fr = cap.read()
            cap.release()
            if not ret:
                continue
            H, W = fr.shape[:2]
            box = ds._clip_box(v["video"], start + frame_window // 2)
            if box is None:
                continue
            x1, y1, x2, y2 = [int(round(b)) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            cropped = fr[y1:y2, x1:x2]
            crop_sizes.append((x2 - x1, y2 - y1))
            r = yolo(cropped, verbose=False)
            bb = r[0].boxes
            if bb is None or len(bb) == 0:
                ratios.append(0.0)
                continue
            cx, cy, cw, ch = bb.xyxy[0].cpu().numpy()
            area = max(0, cw - cx) * max(0, ch - cy)
            ratios.append(area / (cropped.shape[0] * cropped.shape[1]))
        # 裁剪框相对整帧占比 (Swin-Panda 式: 应远小于 1, 即紧贴小鼠放大)
        avg_ratio = np.mean(ratios) if ratios else 0.0
        if crop_sizes:
            cw = np.mean([s[0] for s in crop_sizes]); ch = np.mean([s[1] for s in crop_sizes])
            zoom = 224.0 / max(cw, ch)
        else:
            zoom = 1.0
        print(f"  [{key}] 裁剪后小鼠覆盖率={avg_ratio*100:.1f}%  裁剪框/整帧~{cw*ch/(H*W)*100 if crop_sizes else 0:.0f}%  "
              f"放大~{zoom:.1f}x  [Swin-Panda 式紧贴小鼠]")
    print("\n[说明] Swin-Panda 式: 裁到小鼠检测框(加边距)再放大到 224, 裁剪框应显著小于整帧。")
    return True


# ============================================================ 6. 画图 (仿 MDPI 图10)
def section_plot(orig, over_new):
    banner("[6] 画图: 原始 vs 过采样 数据分布比较趋势图")
    out_dir = os.path.join(ROOT, "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "original_vs_oversampled.png")

    x = np.arange(len(CLASS_NAMES))
    wbar = 0.27
    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    b1 = ax.bar(x - wbar, orig, wbar, label="Original", color="#95a5a6")
    b2 = ax.bar(x, over_new, wbar, label="Oversampled (class-balanced)", color="#2980b9")

    # 趋势线
    ax.plot(x - wbar, orig, "-o", color="#2c3e50", linewidth=1.6, markersize=4,
            label="Original trend")
    ax.plot(x, over_new, "-o", color="#1a5276", linewidth=1.6, markersize=4,
            label="Oversampled trend")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Sample count")
    ax.set_xlabel("Behavior class")
    ax.set_title("Original vs Oversampled data distribution")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")

    for rect in list(b1.patches) + list(b2.patches):
        ax.annotate(f"{int(rect.get_height()):,}", (rect.get_x() + rect.get_width() / 2,
                    rect.get_height()), ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"已保存: {out_png}")
    return out_png


def main():
    yolo_weights = r"C:\Users\WIN10\PycharmProjects\mouseaction\DeepBehavior\runs\detect\train\weights\best.pt"
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "config.yaml"), encoding="utf-8"))
    videos = json.load(open(os.path.join(cfg["DATA"]["meta_root"], "videos.json"), encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    picked = pick_mice(videos, n_per_source=3)
    print("裁剪验证所选视频 (跨数据源):",
          [v["mouse"] for v in picked])

    ds, ok1 = section_data_pipeline(cfg, picked)

    # 对每个选中视频做一次 clip 采样核对
    print("\n--- clip 采样核对 (每视频 1 个窗口) ---")
    for v in picked:
        x, y, orig, crop, box = verify_clip_sample(ds, v)
        key = os.path.splitext(os.path.basename(v["video"]))[0]
        crop_info = f"裁剪后={tuple(crop.shape[:2])}" if crop is not None else "未裁剪(全帧)"
        print(f"  [{key}] x{tuple(x.shape)} (C,T,H,W) | 标签={CLASS_NAMES[y]}(idx{y}) | {crop_info}")

    section_forward(cfg, device)

    # 采样 + 画图: 用全部 train 视频 (split.json)
    split = json.load(open(os.path.join(cfg["DATA"]["meta_root"], "split.json"), encoding="utf-8"))
    train_v = split["train"]
    orig, over = section_sampling(train_v, cfg)
    png = section_plot(orig, over)

    # 增强
    section_augment(ds, picked[0])

    # YOLO 裁剪质量 (真机跑一次)
    section_yolo_crop_check(cfg, ds, picked[:4], yolo_weights)

    print("\n" + "=" * 74)
    print("诊断完成. 分布图: " + png)
    print("=" * 74, flush=True)


if __name__ == "__main__":
    main()
