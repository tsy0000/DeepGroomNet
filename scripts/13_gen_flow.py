"""13 光流批量生成：DeepEthogram TinyMotionNet（官方权重）对每个 clip 生成光流。

流程（DeepEthogram eLife 2021 原文做法）：
  1. 读 clip 45 帧连续帧
  2. 对每个"模型采样帧"（CLIP.num_input_frames 均匀采样，与训练一致）：以该帧为中心的
     11 帧窗 → TinyMotionNet 前向 → 取窗中心帧对光流（"该时刻的运动"，DeepEthogram 帧级语义）
  3. 光流 32×32 float16 存 npz（n_sampled 张 × 2ch × 32×32×2B）
输出：outputs/flow/{clip 文件名}.npz

用途：运动输入从"帧间差"升级为"真光流"——光流只编码运动，与光照/颜色/背景
无关（DeepEthogram 靠它在 9 个分布差异极大的数据集 >85%），
是"新视频泛化"的核心升级。

注意：NUM_FRAMES 从 config.yaml 的 CLIP.num_input_frames 读取（默认 16，32 帧配置
自动对齐），采样函数复用 data.clip_sampler.sample_frames（单一实现，杜绝漂移）。
"""
import os, sys, csv
sys.path.insert(0, r'D:\DeepGroomNet_v2\third_party')
sys.path.insert(0, r'D:\DeepGroomNet_v2')

import numpy as np
import torch
import yaml

FLOW_ROOT = r"D:\DeepGroomNet_v2\outputs\flow"
CKPT = r"D:\pretrained_models\200221_115158_TinyMotionNet\checkpoint.pt"
CLIPS_CSV = r"D:\DeepGroomNet_v2\outputs\clips\clips.csv"
HEAT = 32
CFG_PATH = r"D:\DeepGroomNet_v2\configs\config.yaml"
_cfg_ = yaml.safe_load(open(CFG_PATH, encoding="utf-8"))
NUM_FRAMES_GLOBAL = int(_cfg_["CLIP"]["num_input_frames"])
WINDOW = 11
OUT_IDX = 5


def load_flow_model():
    import cv2
    from deepethogram.flow_generator.models.TinyMotionNet import TinyMotionNet
    m = TinyMotionNet(num_images=WINDOW)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def sample_indices(total, n=None):
    """与 data.clip_sampler.sample_frames 严格一致（同一函数，消除双实现漂移）。"""
    from data.clip_sampler import sample_frames
    if n is None:
        n = NUM_FRAMES_GLOBAL
    return sample_frames(0, total, n)


def clip_flow(model, frames, device):
    """frames: [T,H,W,3] uint8 -> [n_sampled, 2, 32, 32] 光流（每采样帧中心窗）。"""
    import cv2
    T = len(frames)
    idxs = sample_indices(T)
    with torch.no_grad():
        outs = []
        for idx in idxs:
            lo = max(0, min(idx - OUT_IDX, T - WINDOW))
            win = frames[lo:lo + WINDOW]
            if len(win) < WINDOW:
                win = win + [win[-1]] * (WINDOW - len(win))
            arr = np.stack([cv2.resize(f, (256, 256)) for f in win]).astype(np.float32) / 255.0
            arr = np.transpose(arr, (0, 3, 1, 2))
            x = torch.from_numpy(arr.reshape(1, -1, 256, 256)).to(device)
            y = model(x)
            flow = y[0][0, OUT_IDX * 2:(OUT_IDX + 1) * 2]
            flow = torch.nn.functional.interpolate(
                flow.unsqueeze(0), size=(HEAT, HEAT),
                mode="bilinear", align_corners=False)[0]
            outs.append(flow.cpu().numpy())
    return np.stack(outs).astype(np.float16)


def read_clip_frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def main():
    import time
    os.makedirs(FLOW_ROOT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_flow_model().to(device)
    rows = list(csv.DictReader(open(CLIPS_CSV, encoding="utf-8")))

    def _valid_npz(path, want):
        """严格校验：npz 存在且帧数 == 模型输入帧数（防"旧 16 帧 npz 被误判完成"）。"""
        if not os.path.exists(path):
            return False
        try:
            arr = np.load(path)["flow"]
            return arr.shape[0] == want
        except Exception:
            return False

    todo = [r for r in rows
            if not _valid_npz(os.path.join(FLOW_ROOT, os.path.basename(r["clip"]) + ".npz"),
                              NUM_FRAMES_GLOBAL)]
    print(f"clip 总数 {len(rows)}，需生成 {len(todo)}"
          f"（目标帧数 {NUM_FRAMES_GLOBAL}，帧数不符的旧 npz 将重生成）", flush=True)
    t0 = time.time()
    for i, r in enumerate(todo):
        src = r["clip"]
        if not os.path.exists(src):
            print("  missing:", src)
            continue
        frames = read_clip_frames(src)
        if len(frames) < WINDOW:
            continue
        flow = clip_flow(model, frames, device)
        np.savez_compressed(os.path.join(FLOW_ROOT, os.path.basename(src) + ".npz"),
                            flow=flow)
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(todo)} | {el:.0f}s | ETA {(el/(i+1))*(len(todo)-i-1)/60:.0f}min",
                  flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
