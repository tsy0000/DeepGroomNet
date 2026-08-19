"""无泄露评估: 用已训练 5 类权重 + YOLO裁剪滑窗 + 固定阈值 + MoSeq 音节边界后处理。

关键: 评估不做任何在测试视频标签上的阈值调优(无泄露)。
  阈值 = 固定 0.5 (background=NOT others)。MoSeq 音节边界只约束 FG/BG 切换。

用法:
  D:/Anaconda3/envs/deepgroomnet3/python.exe scripts/eval_opto_noleak.py \
      --video E:/tsy/datasetsopto/mouse4325/mouse4325.mp4 \
      --labels E:/tsy/datasetsopto/mouse4325/mouse4325_labels.csv \
      --syllable <out_dir>/mouse4325_syllable.npy \
      --ckpt outputs/checkpoints/lp_best.pt
"""
import os, sys, csv, argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2, yaml
from ultralytics import YOLO
from scripts.train_resume import build_model
from postprocess.thresholds import threshold_predict, compute_background, \
    syllable_boundaries, moseq_boundary_refine

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Locomote", "Rear", "FaceGroom", "BodyGroom", "background"]
BG = 4
NUM_FRAMES = 32
SIZE = 224
MEAN, STD = 0.45, 0.225
YOLO_W = r"C:\Users\WIN10\PycharmProjects\mouseaction\DeepBehavior\runs\detect\train\weights\best.pt"


def load_gt(path, n=None):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    order = ["Locomote", "Rear", "FaceGroom", "BodyGroom", "background"]
    y = []
    for r in rows:
        idx = None
        for c in order[:4]:
            if r.get(c, "0").strip() == "1":
                idx = order.index(c); break
        y.append(order.index("background") if idx is None else idx)
    y = np.asarray(y)
    return y[:n] if n else y


def window_box(frames, yolo):
    boxes = []
    idxs = np.linspace(0, len(frames) - 1, 3).astype(int)
    for i in idxs:
        res = yolo(frames[i], conf=0.3, verbose=False)
        if res[0].boxes is not None and len(res[0].boxes) > 0:
            boxes.append(res[0].boxes.xyxy[0].cpu().numpy())
    if not boxes:
        h, w = frames[0].shape[:2]
        return 0, 0, w, h
    boxes = np.array(boxes)
    x1, y1 = boxes[:, 0].min(), boxes[:, 1].min()
    x2, y2 = boxes[:, 2].max(), boxes[:, 3].max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * 1.5
    half = side / 2
    H, W = frames[0].shape[:2]
    x1 = int(max(0, cx - half)); y1 = int(max(0, cy - half))
    x2 = int(min(W, cx + half)); y2 = int(min(H, cy + half))
    return x1, y1, x2, y2


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--syllable", default=None)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs", "checkpoints", "lp_best.pt"))
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "config.yaml"), encoding="utf-8"))
    model = build_model(cfg, 5).to(DEVICE)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model_state"] if "model_state" in ck else ck
    model.load_state_dict(sd, strict=False)
    model.eval()
    yolo = YOLO(YOLO_W, verbose=False)

    cap = cv2.VideoCapture(args.video)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video={os.path.basename(args.video)} frames={N} device={DEVICE}")

    # 逐窗口 YOLO 裁剪 + 滑窗推理 -> 逐帧概率
    acc = np.zeros((N, 5), dtype=np.float64)
    cnt = np.zeros(N, dtype=np.int32)
    starts = list(range(0, N - NUM_FRAMES + 1, args.stride))
    for wi, s in enumerate(starts):
        frames = []
        for i in range(NUM_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, s + i)
            ok, fr = cap.read()
            frames.append(fr if ok else frames[-1])
        x1, y1, x2, y2 = window_box(frames, yolo)
        win = []
        for fr in frames:
            crop = fr[y1:y2, x1:x2]
            rs = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB)
            win.append(rgb.astype(np.float32) / 255.0)
        x = torch.from_numpy(np.stack(win).transpose(0, 3, 1, 2))
        x = ((x - MEAN) / STD).unsqueeze(0).to(DEVICE)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        p = torch.sigmoid(logits.float()).cpu().numpy()[0]
        acc[s:s + NUM_FRAMES] += p
        cnt[s:s + NUM_FRAMES] += 1
        if (wi + 1) % 300 == 0:
            print(f"  window {wi+1}/{len(starts)}", flush=True)
    cap.release()
    cnt = np.maximum(cnt, 1)
    P = acc / cnt[:, None]
    Y = load_gt(args.labels, N)
    print("sliding-window probs done")

    # ---- 无泄露: 固定阈值 0.5 (不在测试标签上调), background = NOT others ----
    thr = {c: 0.5 for c in CLASS_NAMES}
    pred = threshold_predict(P, thr, CLASS_NAMES)
    pred = compute_background(pred, bg_idx=BG)

    # ---- MoSeq 音节边界后处理: 只在音节切换帧允许 FG/BG 切换 ----
    if args.syllable and os.path.exists(args.syllable):
        syl = np.load(args.syllable)[:N]
        bd = syllable_boundaries(syl)
        pred = moseq_boundary_refine(pred, bd)
        print(f"[moseq] 应用音节边界: {int(bd.sum())} 边界帧, {int(syl.max())+1} 音节")
    else:
        print("[moseq] 未提供音节文件, 跳过边界后处理")

    # ---- 指标 (含 FaceGroom/BodyGroom, 不调阈值) ----
    eye = (Y[:, None] == np.arange(5)).astype(int)
    gt = np.bincount(Y, minlength=5)
    pred_oh = pred
    print("\n===== 真实标签分布 =====")
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:<12}: {gt[i]} ({gt[i]/N*100:.1f}%)")
    print("\n===== 固定0.5阈值 + MoSeq边界 指标 (无泄露) =====")
    from scripts.train_resume import binary_class_metrics, f1_macro_from
    tpc = binary_class_metrics(eye, pred_oh, CLASS_NAMES)
    print(f"  F1_macro(含bg) = {f1_macro_from(tpc, CLASS_NAMES):.2f}%")
    print(f"  argmax 匹配率  = {np.mean(P.argmax(1)==Y)*100:.2f}%")
    print(f"  {'行为':<12}{'Acc':>7}{'F1':>7}{'Recall':>8}{'Precision':>10}")
    for i, c in enumerate(CLASS_NAMES):
        pc = tpc[c]
        print(f"  {c:<12}{pc['accuracy']:>6.1f}%{pc['f1']:>6.1f}%{pc['recall']:>7.1f}%{pc['precision']:>9.1f}%")
    pd = pred.sum(0)
    print("\n===== 预测 vs 真实 =====")
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:<12}: 预测 {int(pd[i])} ({pd[i]/N*100:.1f}%) | 真实 {gt[i]} ({gt[i]/N*100:.1f}%)")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame"] + CLASS_NAMES + ["gt"])
            for i in range(N):
                w.writerow([i] + [int(pred[i, c]) for c in range(5)] + [int(Y[i])])
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
