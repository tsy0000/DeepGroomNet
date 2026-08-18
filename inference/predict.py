"""推理模块: 用 TAdaPandaModel (TAdaConvNeXtV2 + CPAM + STSA) 做滑窗逐帧预测.

- sliding_window_probs: 对视频做滑窗逐帧推断 (T=32), 返回逐帧概率 [N, K].
- predict_video: 单视频完整流程 (概率 -> 平滑 -> 阈值 -> background -> min_bout -> MoSeq 边界).
- load_tadapanda_model: 加载 TAdaPandaModel + 权重.

用法:
    python -m inference.predict --video xxx.mp4 --ckpt outputs/checkpoints/lpft_best.pt
"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party"))

import cv2
from models.tadapanda_model import TAdaPandaModel
from postprocess.thresholds import (threshold_predict, compute_background,
                                    min_bout_per_behavior, smooth_probabilities,
                                    syllable_boundaries, moseq_boundary_refine)


def load_tadapanda_model(cfg, num_classes, ckpt_path=None, device="cpu"):
    """构建 TAdaPandaModel 并加载权重."""
    m = cfg["MODEL"]
    model = TAdaPandaModel(
        num_classes=num_classes,
        t_stride=1,
        use_cpam=bool(m.get("cpam", True)),
        use_stsa=bool(m.get("stsa", True)),
        drop_path=0.0,   # 推理关闭 stochastic depth
    ).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ck.get("state_dict", ck.get("model_state", ck))
        # 兼容 Lightning checkpoint (含 model. 前缀)
        if isinstance(sd, dict) and any(k.startswith("model.") for k in sd):
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def sliding_window_probs(model, video_path, cfg, device, stride=1):
    """滑窗逐帧推断 (T=32). 返回 per-frame 概率 [N, K] 与覆盖数.

    model 输出: logits -> sigmoid。若返回 (logits, aux), 只用主头 logits。
    """
    num_frames = int(cfg["CLIP"]["num_input_frames"])
    size = int(cfg["CLIP"]["target_size"])
    mean = float(cfg["CLIP"].get("norm_mean", 0.45))
    std = float(cfg["CLIP"].get("norm_std", 0.225))
    K = int(cfg["CLASSES"]["openfield"].__len__()) if isinstance(cfg["CLASSES"]["openfield"], list) else 5

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(fr, (size, size)))
    cap.release()
    if not frames:
        return np.zeros((0, K)), np.zeros(0, dtype=np.int32)
    frames = np.stack(frames).astype(np.float32) / 255.0   # [N,H,W,3]
    frames = np.transpose(frames, (0, 3, 1, 2))            # [N,3,H,W]
    frames = (frames - mean) / std
    N = frames.shape[0]
    acc = np.zeros((N, K), dtype=np.float64)
    cnt = np.zeros(N, dtype=np.int32)
    for s in range(0, N - num_frames + 1, stride):
        x = torch.from_numpy(frames[s:s + num_frames]).unsqueeze(0).to(device)  # [1,3,T,H,W]
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        p = torch.sigmoid(out.float()).cpu().numpy()       # [1,K]
        acc[s:s + num_frames] += p
        cnt[s:s + num_frames] += 1
    cnt = np.maximum(cnt, 1)
    return acc / cnt[:, None], cnt


def predict_video(model, video_path, cfg, device, syllable=None, stride=1):
    """单视频完整推理: 概率 -> 平滑 -> 阈值 -> background -> min_bout -> MoSeq 边界.

    返回: (probs, pred) 逐帧概率 [N,K] 与二值预测 [N,K].
    """
    probs, _ = sliding_window_probs(model, video_path, cfg, device, stride=stride)
    if probs.shape[0] == 0:
        return probs, probs
    P = probs
    if cfg["POSTPROCESS"].get("temporal_smooth", False):
        P = smooth_probabilities(P, int(cfg["POSTPROCESS"].get("smooth_window", 3)))
    class_names = cfg["CLASSES"]["openfield"]
    bg_idx = class_names.index("background") if "background" in class_names else None
    # 用 0.5 默认阈值 (推理时无真值调优, 可选传入阈值)
    pred = threshold_predict(P, {c: 0.5 for c in class_names}, class_names)
    pred = compute_background(pred, bg_idx=bg_idx)
    # min_bout 无真值, 用保守默认 (percentile 从 config)
    if cfg["POSTPROCESS"].get("min_bout_per_behavior", False):
        # 无训练标签时用固定 bout 长度 (从 config)
        bl = cfg["POSTPROCESS"].get("min_bout_len", {c: 3 for c in class_names})
        for i, c in enumerate(class_names):
            if i == bg_idx:
                continue
            from postprocess.thresholds import remove_short_bouts
            pred[:, i] = remove_short_bouts(pred[:, i], bl.get(c, 3))
    # MoSeq 边界约束
    if syllable is not None and len(syllable) >= pred.shape[0]:
        pred = moseq_boundary_refine(pred, syllable[:pred.shape[0]],
                                     pairs=((class_names.index("FaceGroom"),
                                             class_names.index("BodyGroom")),) if "FaceGroom" in class_names else ())
    return P, pred