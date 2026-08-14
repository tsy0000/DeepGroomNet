"""光流通道（DeepEthogram TinyMotionNet 官方生成）：加载/实时生成/归一化。

训练：离线 npz（scripts/13_gen_flow.py 生成，与采样帧对齐 [16,2,32,32]）。
推理：实时生成（TinyMotionNet GPU 前向，与训练同一语义）。
归一化统计（实测 50 clip）：mean=0.06 std=0.24。
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

FLOW_MEAN = 0.06
FLOW_STD = 0.24


def load_clip_flow(clip_path: str, flow_root: str):
    """加载 clip 光流 npz -> [n,2,32,32] float32；无则 None。"""
    if not flow_root:
        return None
    p = os.path.join(flow_root, os.path.basename(clip_path) + ".npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["flow"].astype(np.float32)


def flow_to_input(flow, size, mean=FLOW_MEAN, std=FLOW_STD):
    """flow: [n,2,32,32] -> [n,2,size,size] 归一化（上采样保持位移值）。"""
    t = torch.from_numpy(flow)
    if t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return (t - mean) / std


def generate_flow_online(model, frames, device, sample_idx, heat=32,
                         window=11, out_idx=5):
    """推理用：frames=窗口原始帧（含 ±5 上下文），sample_idx=窗内采样帧号（相对 frames）。
    返回 [n,2,32,32] float32 光流（与 13_gen_flow.py 同语义）。
    """
    import cv2
    T = len(frames)
    idxs = sample_idx
    with torch.no_grad():
        outs = []
        for idx in idxs:
            lo = max(0, min(idx - out_idx, T - window))
            win = frames[lo:lo + window]
            if len(win) < window:
                win = win + [win[-1]] * (window - len(win))
            arr = np.stack([cv2.resize(f, (256, 256)) for f in win]).astype(np.float32) / 255.0
            arr = np.transpose(arr, (0, 3, 1, 2))
            x = torch.from_numpy(arr.reshape(1, -1, 256, 256)).to(device)
            y = model(x)
            flow = y[0][0, out_idx * 2:(out_idx + 1) * 2]
            flow = F.interpolate(flow.unsqueeze(0), size=(heat, heat),
                                 mode="bilinear", align_corners=False)[0]
            outs.append(flow.cpu().numpy())
    return np.stack(outs).astype(np.float32)


def load_flow_model(ckpt=r"D:\pretrained_models\200221_115158_TinyMotionNet\checkpoint.pt"):
    from deepethogram.flow_generator.models.TinyMotionNet import TinyMotionNet
    m = TinyMotionNet(num_images=11)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m
