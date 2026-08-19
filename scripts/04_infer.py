"""DeepGroomNet_v3 - 推理 (滑窗逐帧)。

DeepEthogram/VideoPrism 式逐帧推断:
  对每个待分类帧 i, 取覆盖它的所有 16 帧窗口, 每个窗口出一个行为概率,
  把覆盖 i 的窗口概率做平均 → 逐帧概率 → 后处理(阈值/background/min_bout/bout统计)。

可选: --boundaries 传入逐帧"音节边界"(如 Keypoint-MoSeq 的 AR-HMM 音节切换帧),
      用于"保留动作自然边界": 只在边界处允许 FG/BG 切换, 其余平滑。

用法:
  D:/Anaconda3/envs/deepgroomnet3/python.exe scripts/04_infer.py \
      --config configs/config.yaml --ckpt outputs/checkpoints/lpft_best.pt \
      --split test --out outputs/predictions/test_ethogram.csv
"""
import os, sys, json, argparse, csv
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT)

import cv2
from data.dataset import FrameLevelDataset
from postprocess.thresholds import (evaluate_thresholds, threshold_predict,
                                    compute_background, min_bout_per_behavior,
                                    smooth_probabilities, bout_stats,
                                    syllable_boundaries, moseq_boundary_refine)
from scripts.train_resume import build_model


@torch.no_grad()
def sliding_window_probs(model, video_path, cfg, device, stride=1):
    """对单个视频做滑窗逐帧推断。返回 per-frame 概率 [N,5] 与覆盖数。"""
    num_frames = int(cfg["CLIP"]["num_input_frames"])
    size = int(cfg["CLIP"]["target_size"])
    mean = float(cfg["CLIP"].get("norm_mean", 0.45)); std = float(cfg["CLIP"].get("norm_std", 0.225))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 读全部帧 (内存可承受则整段读; 否则用 frame 缓存)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret: break
        frames.append(cv2.resize(fr, (size, size)))
    cap.release()
    frames = np.stack(frames).astype(np.float32) / 255.0            # [N,H,W,3]
    frames = np.transpose(frames, (0, 3, 1, 2))                     # [N,3,H,W]
    frames = (frames - mean) / std
    N = frames.shape[0]
    acc = np.zeros((N, 5), dtype=np.float64)
    cnt = np.zeros(N, dtype=np.int32)
    for s in range(0, N - num_frames + 1, stride):
        x = torch.from_numpy(frames[s:s + num_frames]).unsqueeze(0).to(device)  # [1,3,T,H,W]
        logits = model(x)
        p = torch.sigmoid(logits.float()).cpu().numpy()             # [1,5]
        acc[s:s + num_frames] += p
        cnt[s:s + num_frames] += 1
    cnt = np.maximum(cnt, 1)
    return acc / cnt[:, None], cnt


def _load_h5(video_path, moseq_root, cache):
    """从 Keypoint-MoSeq results.h5 读某视频的逐帧音节, 返回 syllable 数组 [N]。"""
    import h5py
    key = os.path.splitext(os.path.basename(video_path))[0]   # 如 mouse672
    if moseq_root not in cache:
        # 扫描所有 run 的 results.h5, 构建 {mouse: syllable_array}
        mapping = {}
        if os.path.isdir(moseq_root):
            for h5 in sorted(__import__("glob").glob(os.path.join(moseq_root, "**", "results.h5"), recursive=True)):
                try:
                    with h5py.File(h5, "r") as f:
                        for gk in f.keys():
                            if "syllable" in f[gk]:
                                mouse = gk.split("_")[0]
                                mapping.setdefault(mouse, np.asarray(f[gk]["syllable"]))
                except Exception:
                    continue
        cache[moseq_root] = mapping
    return cache[moseq_root].get(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs", "checkpoints", "lpft_best.pt"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "predictions", "ethogram.csv"))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--boundaries", default=None,
                    help="逐帧音节边界 npy/npz (0/1); 用于保留自然边界")
    ap.add_argument("--moseq-root", default=None,
                    help="Keypoint-MoSeq 项目目录(如 C:/Users/WIN10/newproject1); "
                         "自动读各 run 的 results.h5 逐帧音节, 把音节切换帧当边界")
    ap.add_argument("--save-probs", default=None, help="保存逐帧概率 npy 到该路径")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = cfg["CLASSES"]["openfield"]
    num_classes = len(class_names)
    frame_window = int(cfg["CLIP"]["num_input_frames"])

    split = json.load(open(os.path.join(cfg["DATA"]["meta_root"], "split.json")))
    videos = split[args.split]
    if not videos:
        print(f"split '{args.split}' 为空")
        return
    print(f"推理 split={args.split} 视频数={len(videos)} device={device}")

    model = build_model(cfg, num_classes).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()

    all_pred = []
    all_true = []
    all_frame_index = []   # (video_idx, frame) 用于回溯
    video_names = []
    moseq_cache = {}   # moseq_root -> {mouse: syllable_array}
    for vi, v in enumerate(videos):
        if args.save_probs is not None:
            # 需要逐视频概率: 用滑窗, 逐视频处理
            Pv, _ = sliding_window_probs(model, v["video"], cfg, device, stride=args.stride)
            if args.save_probs:
                os.makedirs(os.path.dirname(args.save_probs), exist_ok=True)
                np.save(f"{args.save_probs}.video{vi}.npy", Pv)
        else:
            # 用 dataset 滑窗等价: 仍调滑窗 (直接读视频)
            Pv, _ = sliding_window_probs(model, v["video"], cfg, device, stride=args.stride)
        labels = np.asarray(v["labels"])
        n = Pv.shape[0]
        # 逐帧后处理(阈值/background/min_bout) 对整段概率做
        P = Pv
        if cfg["POSTPROCESS"].get("temporal_smooth", False):
            P = smooth_probabilities(P, int(cfg["POSTPROCESS"].get("smooth_window", 3)))
        thresholds, _ = evaluate_thresholds(labels[:n], P, class_names, bg_idx=4)
        # 简化阈值: evaluate_thresholds 需要 int 标签
        pred = threshold_predict(P, thresholds, class_names)
        pred = compute_background(pred, bg_idx=4)
        if cfg["POSTPROCESS"].get("min_bout_per_behavior", False):
            pred = min_bout_per_behavior(pred, labels[:n], class_names,
                                         percentile=cfg["POSTPROCESS"].get("min_bout_percentile", 1),
                                         bg_idx=4)
        # MoSeq 辅助后处理 (保留自然边界): 用音节切换约束 FG/BG
        if args.boundaries:
            bd = np.load(args.boundaries)
            bd = bd["boundaries"] if bd.ndim == 0 else bd
            pred = moseq_boundary_refine(pred, syllable_boundaries(bd[:n]))
        elif args.moseq_root:
            syl = _load_h5(v["video"], args.moseq_root, moseq_cache)
            if syl is not None and len(syl) >= n:
                pred = moseq_boundary_refine(pred, syl[:n])
                print(f"    [moseq] Keypoint-MoSeq 辅助后处理 (边界帧={int(syllable_boundaries(syl[:n]).sum())})")
            else:
                print(f"    [moseq] 未找到该视频音节, 跳过")
        all_pred.append(pred)
        all_true.append(labels[:n])
        all_frame_index.append((np.full(n, vi), np.arange(n)))
        video_names.append(os.path.basename(v["video"]))
        print(f"  [{vi+1}/{len(videos)}] {os.path.basename(v['video'])} 帧={n}")

    pred_all = np.concatenate(all_pred)
    true_all = np.concatenate(all_true)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 写 ethogram CSV (video,frame, 5 类 0/1)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame"] + class_names)
        off = 0
        for vi, n in enumerate([p.shape[0] for p in all_pred]):
            for i in range(n):
                w.writerow([video_names[vi], i] + [int(pred_all[off + i, c]) for c in range(num_classes)])
            off += n
    print(f"\n逐帧 ethogram -> {args.out}")

    # 在 test 上的指标 (若有真值)
    eye = (true_all[:, None] == np.arange(num_classes)).astype(int)
    from scripts.train_resume import binary_class_metrics, f1_macro_from
    tpc = binary_class_metrics(eye, pred_all, class_names)
    print(f"[{args.split}] F1_macro={f1_macro_from(tpc, class_names):.2f}")
    print(f"[{args.split}] {'行为':<12}{'Acc':>7}{'F1':>7}{'Recall':>8}{'Precision':>10}")
    for c in class_names:
        pc = tpc[c]
        print(f"[{args.split}] {c:<12}{pc['accuracy']:>6.1f}%{pc['f1']:>6.1f}%"
              f"{pc['recall']:>7.1f}%{pc['precision']:>9.1f}%")
    stats = bout_stats(pred_all, class_names, bg_idx=4)
    print("\n[bout] " + ", ".join(f"{c}: {stats[c]['n_bouts']}bouts/"
                                  f"{stats[c]['total_frames']}f" for c in class_names if c != 'background'))


if __name__ == "__main__":
    main()
