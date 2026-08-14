"""
模块级可解释性验证（FaceGroom vs BodyGroom 判别特征可视化）。

用途：取一条 FaceGroom + 一条 BodyGroom 真实 clip，跑完整前向，
可视化每个模块是否真的把注意力放在"判别性像素"上：
  1. 原帧（t = 采样帧索引 2/6/10/14）
  2. MotionAttention 空间运动显著图（w_s，运动=行为编码）
  3. PartMotionAttention front/back 部位分裂图（前端=头+前肢 / 后端=躯干+后肢）
  4. Grad-CAM（FaceGroom 类 vs BodyGroom 类，stage-4 特征）
  5. 部位运动能量表 s_front / s_back（判别性量化指标：FG 应 front 主导，BG 应 back 主导）

输出：outputs/visualization/attention_<label>.png + 终端诊断表。

用法：
  python scripts/06_visualize_attention.py [--clip_fg PATH] [--clip_bg PATH]
"""
import argparse
import csv
import os
import sys

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ClipDataset
from data.clip_sampler import sample_frames
from models.backbones.tadaconvnext import build_model
from models.attention import MotionAttention, MotionStatAttention, PartMotionAttention

CLASSES = ["background", "Locomote", "Rear", "FaceGroom", "BodyGroom"]


def find_clip(csv_path, label, prefer="mouse021"):
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["label"] == label:
                return r["clip"]
    return None


def load_clip_tensor(clip_path, cfg):
    """与 ClipDataset 完全一致的帧采样/归一化（无增强）。"""
    frames = ClipDataset._read_clipfile(clip_path)
    idx = sample_frames(0, len(frames), int(cfg["CLIP"]["num_input_frames"]))
    if len(idx) < int(cfg["CLIP"]["num_input_frames"]) and idx:
        idx = np.resize(np.asarray(idx), int(cfg["CLIP"]["num_input_frames"])).tolist()
    sel = []
    for i in idx:
        if 0 <= i < len(frames):
            sel.append(frames[i])
        else:
            sel.append(np.zeros((480, 480, 3), np.uint8))
    while len(sel) < int(cfg["CLIP"]["num_input_frames"]):
        sel.append(sel[len(sel) % max(1, len(sel))])
    size = int(cfg["CLIP"]["target_size"])
    arr = np.stack([cv2.resize(f, (size, size)) for f in sel]).astype(np.float32) / 255.0
    x_arr = arr
    if int(cfg["CLIP"].get("motion_channel", 0)):
        from data.dataset import add_motion_channel
        x_arr = add_motion_channel(arr)
    x = torch.from_numpy(np.transpose(x_arr, (3, 0, 1, 2)))
    mean = torch.tensor(cfg["CLIP"]["mean"] + ([cfg["CLIP"].get("motion_mean", 0.0085)]
                                               if x.shape[0] == 4 else [])).view(-1, 1, 1, 1)
    std = torch.tensor(cfg["CLIP"]["std"] + ([cfg["CLIP"].get("motion_std", 0.044)]
                                             if x.shape[0] == 4 else [])).view(-1, 1, 1, 1)
    x = (x - mean) / std
    return x, arr


def collect_module_maps(model):
    """设置捕获并返回 (motions, stats, parts) 每 stage 的注意力实例。"""
    motions, stats, parts = [], [], []
    for attn in model.attentions:
        for m in ([attn] if not isinstance(attn, nn.Sequential) else list(attn)):
            if isinstance(m, MotionAttention):
                m._capture = True
                motions.append(m)
            elif isinstance(m, MotionStatAttention):
                m._capture = True
                stats.append(m)
            elif isinstance(m, PartMotionAttention):
                m._capture = True
                parts.append(m)
    return motions, stats, parts


def grad_cam(model, x, class_idx, stage_idx=3):
    """Grad-CAM：stage_idx 输出特征对 class_idx logit 的梯度加权激活（mean over T）。"""
    feats = {}

    def fwd_hook(m, i, o):
        o.retain_grad()
        feats["A"] = o

    h = model.stages[stage_idx].register_forward_hook(fwd_hook)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    target = logits[0, class_idx]
    target.backward(retain_graph=True)
    h.remove()
    A = feats["A"]
    g = A.grad if A.grad is not None else torch.zeros_like(A)
    w = g.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((w * A).sum(dim=1, keepdim=True))
    cam = cam.mean(dim=2)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, logits.detach()


def heat_overlay(frame_bgr, heat, alpha=0.55):
    """heat: [H,W] in [0,1] -> 原帧上叠加伪彩色热图。"""
    heat_rgb = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, heat_rgb, alpha, 0)


def upscale_map(arr, size=224):
    m = torch.from_numpy(np.asarray(arr, dtype=np.float32))[None, None]
    m = F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)
    return m[0, 0].numpy()


def visualize_one(clip_path, label, model, cfg, out_dir, cam_cls_idx):
    x, arr = load_clip_tensor(clip_path, cfg)
    x = x.to(next(model.parameters()).device).unsqueeze(0)

    motions, stats, parts = collect_module_maps(model)
    logits = model(x)
    probs = torch.softmax(logits[0], dim=0).detach().cpu().numpy()
    pred = int(logits[0].argmax().item())

    part_rows = []
    for i, p in enumerate(parts):
        front, back, s_f, s_b, _ = p._capture_cache
        part_rows.append((i, s_f.mean().item(), s_b.mean().item()))

    cam_fg, _ = grad_cam(model, x, CLASSES.index("FaceGroom"))
    cam_bg, _ = grad_cam(model, x, CLASSES.index("BodyGroom"))

    show_t = [2, 6, 10, 14]
    frames_bgr = [(arr[t] * 255).astype(np.uint8) for t in show_t]

    rows, row_names = [], []
    p3 = parts[-1] if parts else None
    m3 = motions[-1] if motions else None
    s3 = stats[-1] if stats else None

    rows.append(frames_bgr)
    row_names.append("frame")

    if m3 is not None:
        w_s3 = m3._capture_cache[2][0, 0].mean(dim=0).detach().cpu().numpy()
        rows.append([heat_overlay(f, upscale_map(w_s3)) for f in frames_bgr])
        row_names.append("motion_gate")
    if s3 is not None:
        r_n, mu, sg, c, w_s, m1 = s3._capture_cache
        fast = r_n[0].mean(dim=0).detach().cpu().numpy()
        gate = w_s[0, 0].mean(dim=0).detach().cpu().numpy()
        rows.append([heat_overlay(f, upscale_map(fast)) for f in frames_bgr])
        row_names.append("fast_map(r=m1/m4)")
        rows.append([heat_overlay(f, upscale_map(gate)) for f in frames_bgr])
        row_names.append("stat_gate(w_s)")
        info_c = f"concentration c={c.mean().item():.3f}"
    else:
        info_c = ""
    if p3 is not None:
        f, b, _, _, _ = p3._capture_cache
        front3, back3 = f[0, 0, 0].cpu().numpy(), b[0, 0, 0].cpu().numpy()
        rows.append([heat_overlay(fr, upscale_map(front3)) for fr in frames_bgr])
        row_names.append("front_part")
        rows.append([heat_overlay(fr, upscale_map(back3)) for fr in frames_bgr])
        row_names.append("back_part")

    rows.append([heat_overlay(f, cam_fg) for f in frames_bgr])
    row_names.append("GradCAM_FG")
    rows.append([heat_overlay(f, cam_bg) for f in frames_bgr])
    row_names.append("GradCAM_BG")

    nrows = len(rows)
    h, w = frames_bgr[0].shape[:2]
    pad = 8
    canvas = np.full((nrows * (h + pad) + 240, 4 * (w + pad) + 300, 3), 255, np.uint8)
    for r in range(nrows):
        cv2.putText(canvas, row_names[r], (10, r * (h + pad) + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        for c in range(4):
            img = rows[r][c]
            canvas[r * (h + pad):r * (h + pad) + h,
                   300 + c * (w + pad):300 + c * (w + pad) + w] = img
    info = [f"{label} clip: {os.path.basename(clip_path)}",
            f"pred={CLASSES[pred]} ({probs[pred] * 100:.1f}%)"]
    if info_c:
        info.append(info_c)
    for i, sf, sb in part_rows:
        ratio = sf / (sb + 1e-8)
        info.append(f"stage{i} part: front={sf:.4f} back={sb:.4f} ratio={ratio:.2f}")
    info.append("probs: " + " ".join(f"{p * 100:.1f}" for p in probs))
    y = 16
    for line in info:
        cv2.putText(canvas, line, (10, nrows * (h + pad) + y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y += 20

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"attention_{label}.png")
    cv2.imwrite(out, canvas)
    print(f"[viz] {label} -> {out}")
    print(f"  pred={CLASSES[pred]} p={probs[pred] * 100:.1f}%  logits="
          + " ".join(f"{v:.2f}" for v in logits[0].detach().cpu().numpy()))
    for i, sf, sb in part_rows:
        print(f"  stage{i} s_front={sf:.4f} s_back={sb:.4f} "
              f"ratio={sf / (sb + 1e-8):.2f}  "
              f"({'FRONT-dominant' if sf > sb else 'BACK-dominant'})")
    return part_rows, probs


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--clip_fg", default=None)
    ap.add_argument("--clip_bg", default=None)
    ap.add_argument("--ckpt", default=None, help="可选：加载训练好的 best_model.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()

    ckpt = args.ckpt or os.path.join(cfg["OUTPUT"]["checkpoint_dir"], "best_model.pt")
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(sd["state_dict"], strict=False)
        print(f"[viz] loaded checkpoint {ckpt}")
    else:
        print("[viz] 无 checkpoint，使用预训练 backbone + 随机 head（Grad-CAM 仅供参考）")

    clips_csv = os.path.join(cfg["DATA"]["clip_output_root"], "clips.csv")
    fg = args.clip_fg or find_clip(clips_csv, "FaceGroom")
    bg = args.clip_bg or find_clip(clips_csv, "BodyGroom")
    if not fg or not bg:
        raise SystemExit("找不到 FG/BG clip，请用 --clip_fg/--clip_bg 指定")
    print(f"[viz] FG={fg}\n[viz] BG={bg}")

    out_dir = os.path.join(cfg["OUTPUT"]["log_dir"], "..", "visualization")
    out_dir = os.path.normpath(out_dir)
    print("\n===== FaceGroom =====")
    visualize_one(fg, "FaceGroom", model, cfg, out_dir, CLASSES.index("FaceGroom"))
    print("\n===== BodyGroom =====")
    visualize_one(bg, "BodyGroom", model, cfg, out_dir, CLASSES.index("BodyGroom"))


if __name__ == "__main__":
    main()
