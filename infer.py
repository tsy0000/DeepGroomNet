"""
推理：视频 -> 帧级行为预测（模块B推理端 + 模块C TTA）。

流程：
  1) 加载 best_model.pt（class_names 随权重保存）
  2) 窗口生成（与训练完全一致）：
       - 默认 window_mode="sliding"：帧级滑窗（步进=clip_interval，与训练 uniform 同结构），
         每帧被多个窗口覆盖 -> logits 平均（天然时间平滑）
       - 可选 window_mode="syllable"：按 MoSeq 音节边界整窗预测（legacy）
  3) 每窗均匀抽 16 帧（与训练 ClipDataset 同一函数 sample_frames），归一化 + TTA
  4) 帧级 logits 平均 -> argmax + 置信度
  5) 音节多数投票（提供 MoSeq 音节文件时，保留"自然分界"）+ 短 bout 合并
  6) 输出帧级预测 CSV + 类别占比 + 可选音节聚合 CSV
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))

from data.clip_sampler import (infer_sliding_windows, infer_syllable_clips,
                               sample_frames, read_frames)
from postprocess.crop_box import build_crop_boxer
from postprocess.temporal_smoothing import smooth
from postprocess.syllable_vote import apply_syllable_pipeline


def load_model(cfg, ckpt_path, device):
    from models.backbones.tadaconvnext import build_model
    model = build_model(cfg).to(device)
    sd = torch.load(ckpt_path, map_location="cpu")
    r = model.load_state_dict({k: v.to(device) for k, v in sd["state_dict"].items()},
                               strict=False)
    if r.missing_keys or r.unexpected_keys:
        print(f"[infer] checkpoint 键差异：missing={len(r.missing_keys)} "
              f"unexpected={len(r.unexpected_keys)}（运动通道开关变化属正常，"
              f"建议用同配置新训 ckpt）")
    model.eval()
    return model, sd["class_names"]


@torch.no_grad()
def predict_clip(model, video, start, end, cfg, device, crop_boxer=None, kp_array=None,
                 kp_bodyparts=None, flow_model=None, video_nframes=None):
    """对单个窗口预测：均匀抽帧 -> 归一化 -> TTA -> logits。返回 logits [C]。

    帧采样与训练 ClipDataset 完全一致（sample_frames 均匀抽取，非连续帧）。
    crop_boxer: 可选 YOLO 窗口级裁剪（与训练 clip 生成同策略，保证 ROI 分布一致）。
    kp_array/kp_bodyparts: DLC 关键点（PoseC3D 热图分支 + 聚焦先验；None=不启用）。
    flow_model: TinyMotionNet 光流生成器（CLIP.flow_channel=true 时实时生成）。
    """
    n_frames = int(cfg["CLIP"]["num_input_frames"])
    size = int(cfg["CLIP"]["target_size"])
    motion = bool(cfg["CLIP"].get("motion_channel", False))
    use_flow = bool(cfg["CLIP"].get("flow_channel", False))
    mean = cfg["CLIP"]["mean"]
    std = cfg["CLIP"]["std"]
    if motion:
        mean = list(mean) + [cfg["CLIP"].get("motion_mean", 0.02)]
        std = list(std) + [cfg["CLIP"].get("motion_std", 0.05)]
    if use_flow:
        from data.flow_utils import FLOW_MEAN, FLOW_STD
        mean = list(mean) + [FLOW_MEAN, FLOW_MEAN]
        std = list(std) + [FLOW_STD, FLOW_STD]
    mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1, 1).to(device)
    std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1, 1).to(device)

    idx = sample_frames(start, end, n_frames)
    if len(idx) < n_frames and idx:
        idx = np.resize(np.asarray(idx), n_frames).tolist()
    frames = read_frames(video, idx)
    if crop_boxer is not None:
        frames = crop_boxer.apply(frames, size)
    arr = np.stack([cv2.resize(f, (size, size)) for f in frames]).astype(np.float32) / 255.0
    chans = [arr]
    if motion:
        d = np.abs(np.diff(arr, axis=0, prepend=arr[:1])).mean(axis=-1, keepdims=True)
        chans.append(d)
    if use_flow:
        if flow_model is not None:
            from data.flow_utils import generate_flow_online, flow_to_input
            nf_total = video_nframes or end + 5
            ctx0 = max(0, start - 5)
            ctx = read_frames(video, list(range(ctx0, min(nf_total, end + 5))))
            fl = generate_flow_online(flow_model, ctx, device,
                                      sample_idx=[i - ctx0 for i in idx])
            flow = flow_to_input(fl, size).numpy()
        else:
            flow = np.zeros((len(idx), 2, size, size), np.float32)
        chans.append(np.transpose(flow, (0, 2, 3, 1)))
    arr = np.concatenate(chans, axis=-1)
    x = torch.from_numpy(np.transpose(arr, (3, 0, 1, 2))).to(device)
    x = (x - mean_t) / std_t

    pose = kp_map = None
    if kp_array is not None and kp_bodyparts is not None:
        from data.keypoints import pose_features
        from data.pose_target import PoseTargetGenerator, SUPERANIMAL_SKELETONS
        kp_cfg = cfg.get("KEYPOINTS", {}) or {}
        p = pose_features(kp_array, idx, kp_bodyparts,
                          use_parts=kp_cfg.get("use_parts"))
        if p is not None:
            pose = torch.from_numpy(p).unsqueeze(0).to(device)
            if kp_cfg.get("render_prior", True):
                hm_size = int(kp_cfg.get("render_heatmap_size", 64))
                gen = PoseTargetGenerator(sigma=float(kp_cfg.get("render_sigma", 0.6)))
                maps = gen.gen(p, hm_size,
                               skeletons=(SUPERANIMAL_SKELETONS
                                          if kp_cfg.get("use_parts") is None else None))
                kp_map = torch.from_numpy(maps).unsqueeze(0).to(device)

    def _crop_resize(t, sc):
        if abs(sc - 1.0) < 1e-6:
            return t
        arr = t.permute(1, 2, 3, 0).cpu().numpy()
        if sc < 1.0:
            pad = int(np.ceil(size / sc))
            padder = np.zeros((arr.shape[0], pad, pad, arr.shape[3]), dtype=arr.dtype)
            o = (pad - size) // 2
            padder[:, o:o + size, o:o + size, :] = arr
            arr = np.stack([cv2.resize(f, (pad, pad)) for f in padder])
            j = (pad - size) // 2
            arr = arr[:, j:j + size, j:j + size, :]
        else:
            hs = ws = int(round(size * sc))
            arr = np.stack([cv2.resize(f, (ws, hs)) for f in arr])
            i = (hs - size) // 2
            arr = arr[:, i:i + size, i:i + size, :]
        return torch.from_numpy(np.transpose(arr, (3, 0, 1, 2))).to(device)

    logits = model(x.unsqueeze(0), pose=pose, kp_map=kp_map)[0]
    tta = [logits]
    if cfg["TRAIN"]["test_aug"].get("hflip", True):
        kp_f = torch.flip(kp_map, dims=[-1]) if kp_map is not None else None
        tta.append(model(torch.flip(x.unsqueeze(0), dims=[-1]), pose=pose,
                         kp_map=kp_f)[0])
    for sc in cfg["TRAIN"]["test_aug"].get("crop_scales", []):
        tta.append(model(_crop_resize(x, sc).unsqueeze(0), pose=pose,
                         kp_map=kp_map)[0])
    return torch.stack(tta).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--video", required=True)
    ap.add_argument("--syllables", default=None, help="Keypoint-MoSeq state csv（首列 syllable）")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--window-mode", choices=["sliding", "syllable"], default=None,
                    help="推理窗口模式（默认取 config.INFER.window_mode，缺省 sliding）")
    ap.add_argument("--crop-mode", choices=["none", "window", "static"], default=None,
                    help="ROI 裁剪模式（默认取 config.INFER.crop_mode=window，"
                         "与训练 clip 生成同策略；none=整帧 resize）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = args.ckpt or os.path.join(cfg["OUTPUT"]["checkpoint_dir"], "best_model.pt")
    model, class_names = load_model(cfg, ckpt, device)

    thresholds = None
    thr_file = os.path.join(cfg["OUTPUT"]["metric_dir"], "thresholds.json")
    if os.path.exists(thr_file):
        with open(thr_file, encoding="utf-8") as f:
            thresholds = np.asarray(json.load(f)["thresholds"], dtype=np.float32)
        print(f"[infer] 使用 DeepEthogram 逐类阈值（{thr_file}）")

    if args.crop_mode is not None:
        cfg["INFER"]["crop_mode"] = args.crop_mode
    crop_boxer = None
    try:
        crop_boxer = build_crop_boxer(cfg)
    except FileNotFoundError as e:
        print(f"[infer] 警告: {e}（回退整帧 resize）")

    nf = video_frame_count(args.video)

    syllables = None
    if args.syllables and os.path.exists(args.syllables):
        df = pd.read_csv(args.syllables)
        col = df.columns[0]
        syl = df[col].to_numpy()[:nf]
        if len(syl) < nf:
            syl = np.pad(syl, (0, nf - len(syl)), mode="edge")
        syllables = syl

    window_mode = args.window_mode or cfg["INFER"].get("window_mode", "sliding")
    if window_mode == "syllable":
        if syllables is None:
            raise RuntimeError("window_mode='syllable' 需要 --syllables 文件")
        clips = infer_syllable_clips(syllables, cfg)
    else:
        clips = infer_sliding_windows(nf, cfg)

    kp_array = kp_bodyparts = None
    kp_cfg = cfg.get("KEYPOINTS") or {}
    if kp_cfg.get("enabled", True) and kp_cfg.get("root"):
        from data.keypoints import KeypointBank
        bank = KeypointBank(kp_cfg["root"])
        mouse = os.path.basename(os.path.dirname(args.video))
        kp_bodyparts, kp_array = bank.get(mouse)
        if kp_array is not None:
            print(f"[infer] 启用关键点分支（{mouse}，{len(kp_bodyparts)} 点）")
        else:
            print(f"[infer] {mouse} 无关键点数据，跳过关键点分支")

    flow_model = None
    if cfg["CLIP"].get("flow_channel", False):
        from data.flow_utils import load_flow_model
        flow_model = load_flow_model().to(device)
        print("[infer] 启用 TinyMotionNet 光流生成")

    frame_logits = np.zeros((nf, len(class_names)), dtype=np.float32)
    counts = np.zeros(nf, dtype=np.float32)
    with torch.no_grad():
        for (s, e) in clips:
            log = predict_clip(model, args.video, s, min(e, nf), cfg, device,
                               crop_boxer=crop_boxer, kp_array=kp_array,
                               kp_bodyparts=kp_bodyparts, flow_model=flow_model,
                               video_nframes=nf)
            log = log.cpu().numpy()
            seg = slice(s, min(e, nf))
            frame_logits[seg] += log
            counts[seg] += 1.0

    valid = counts > 0
    frame_logits[valid] /= counts[valid, None]
    probs = 1.0 / (1.0 + np.exp(-frame_logits))

    if thresholds is not None:
        from postprocess.thresholds import threshold_predict
        bg_idx = class_names.index("background") if "background" in class_names else len(class_names) - 1
        pred = threshold_predict(probs, thresholds, bg_idx=bg_idx)
        conf = np.take_along_axis(probs, pred[:, None], axis=1)[:, 0]
    else:
        pred = probs.argmax(1)
        conf = probs.max(1)

    if syllables is not None:
        _, pred = apply_syllable_pipeline(pred, syllables, cfg, class_names=class_names)
    else:
        method = cfg["POSTPROCESS"].get("smoothing", "sequence")
        if method == "sequence":
            from postprocess.sequence_decode import sequence_decode
            pred = sequence_decode(pred, cfg, class_names=class_names)
        else:
            pred = smooth(pred, method=method,
                          window=cfg["POSTPROCESS"].get("vote_window", 5))

    out = args.out or os.path.join(ROOT, "outputs", "predictions",
                                   os.path.basename(os.path.splitext(args.video)[0]) + "_pred.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df_out = pd.DataFrame({
        "frame": np.arange(nf),
        "pred_class_id": pred,
        "pred_class": [class_names[i] for i in pred],
        "confidence": conf,
    })
    df_out.to_csv(out, index=False)
    print(f"[infer] saved -> {out}")
    print(df_out["pred_class"].value_counts())

    if syllables is not None:
        out_syl = os.path.splitext(out)[0] + "_syllables.csv"
        rows = []
        i = 0
        while i < nf:
            j = i + 1
            while j < nf and syllables[j] == syllables[i]:
                j += 1
            seg = pred[i:j]
            vals, counts_ = np.unique(seg, return_counts=True)
            maj = vals[int(np.argmax(counts_))]
            rows.append({"syllable": int(syllables[i]), "start_frame": i, "end_frame": j - 1,
                         "duration_frames": j - i,
                         "duration_s": round((j - i) / int(cfg["CLIP"]["fps"]), 3),
                         "behavior": class_names[maj]})
            i = j
        pd.DataFrame(rows).to_csv(out_syl, index=False)
        print(f"[infer] syllable summary saved -> {out_syl}")


def video_frame_count(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


if __name__ == "__main__":
    main()
