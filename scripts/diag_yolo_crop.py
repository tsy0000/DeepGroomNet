# -*- coding: utf-8 -*-
"""Swin-Panda 式 YOLO 裁剪效果预览 + 类平衡采样检查.

1. 用用户 YOLO 权重, 按 Swin-Panda (Diversity 2025,17,139) 3.2 节策略裁剪:
   检测小鼠框 -> 以框中心 + 边长(margin 裕量)取正方形 -> 放大到 224.
   对每个视频取 3 帧(早/中/晚), 并排保存 [原图(画框) | 裁剪后224].
2. 打印类平衡采样(Swin-Panda 均匀过采样)前后各类窗口数.

输出: outputs/figures/yolo_crop_swinpanda/*.png
用法:  D:/Anaconda3/envs/deepgroomnet3/python.exe scripts/diag_yolo_crop.py
"""
import os, sys, json, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

YOLO_WT = r"C:\Users\WIN10\PycharmProjects\mouseaction\DeepBehavior\runs\detect\train\weights\best.pt"
CLASS_NAMES = ["Locomote", "Rear", "FaceGroom", "BodyGroom", "background"]


def square_crop_box(box, margin, H, W):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    side = min(side, max(W, H))
    bx1, by1 = cx - side / 2, cy - side / 2
    bx2, by2 = cx + side / 2, cy + side / 2
    if bx1 < 0: bx2 += -bx1; bx1 = 0
    if by1 < 0: by2 += -by1; by1 = 0
    if bx2 > W: bx1 -= bx2 - W; bx2 = W
    if by2 > H: by1 -= by2 - H; by2 = H
    return int(max(0, bx1)), int(max(0, by1)), int(min(W, bx2)), int(min(H, by2))


def main():
    from ultralytics import YOLO
    yolo = YOLO(YOLO_WT)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "config.yaml"), encoding="utf-8"))
    videos = json.load(open(os.path.join(cfg["DATA"]["meta_root"], "videos.json"), encoding="utf-8"))
    out_dir = os.path.join(ROOT, "outputs", "figures", "yolo_crop_swinpanda")
    os.makedirs(out_dir, exist_ok=True)
    margin = 0.2

    # 每个数据源挑 1 个视频
    by_src = {}
    for v in videos:
        by_src.setdefault(v["source"], []).append(v)
    picks = []
    for src, vs in by_src.items():
        picks.append(vs[0])

    for v in picks:
        cap = cv2.VideoCapture(v["video"])
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frs = [0, n // 2, n - 1]          # 早/中/晚
        fig, axes = plt.subplots(3, 2, figsize=(8, 11))
        for ri, f in enumerate(frs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, fr = cap.read()
            if not ret:
                continue
            img = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            r = yolo(fr, conf=0.3, verbose=False)[0]
            bb = r.boxes
            lab = np.asarray(v["labels"])[f]
            name = CLASS_NAMES[int(lab)] if int(lab) < len(CLASS_NAMES) else "?"
            axes[ri][0].imshow(img)
            if bb is not None and len(bb) > 0:
                b = bb.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                cx1, cy1, cx2, cy2 = square_crop_box([x1, y1, x2, y2], margin, H, W)
                axes[ri][0].add_patch(plt.Rectangle((cx1, cy1), cx2 - cx1, cy2 - cy1,
                                                    fill=False, edgecolor="lime", lw=2))
                axes[ri][0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                    fill=False, edgecolor="red", lw=1.5))
                cropped = img[cy1:cy2, cx1:cx2]
                resized = cv2.resize(cropped, (224, 224))
                axes[ri][1].imshow(resized)
                axes[ri][1].set_title(f"SwinPanda crop {cx2-cx1}x{cy2-cy1}->224")
            else:
                axes[ri][1].text(0.5, 0.5, "no det", ha="center")
            axes[ri][0].set_title(f"orig {W}x{H}  frame{f}  label={name}")
            for a in axes[ri]:
                a.set_axis_off()
        cap.release()
        plt.tight_layout()
        out = os.path.join(out_dir, f"yolo_crop_{v['mouse']}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("saved:", out, flush=True)

    # ================= 类平衡采样检查 =================
    print("\n" + "=" * 66)
    print("类平衡采样 (Swin-Panda 均匀过采样) 前后分布", flush=True)
    from data.dataset import FrameLevelDataset
    from data.dataloaders import build_oversampled_sampler
    split = json.load(open(os.path.join(cfg["DATA"]["meta_root"], "split.json"), encoding="utf-8"))
    train = split["train"]
    ds = FrameLevelDataset(train, cfg, split="train", frame_window=int(cfg["CLIP"]["num_input_frames"]))
    K = len(CLASS_NAMES)
    lab_arr = [np.asarray(v["labels"]) for v in train]      # 预缓存, 避免逐窗口重建
    before = np.zeros(K, dtype=int)
    for lab in lab_arr:
        for c in range(K):
            before[c] += int((lab == c).sum())
    _ = build_oversampled_sampler(ds, seed=int(cfg["TRAIN"].get("seed", 42)))
    fw = ds.fixed_windows
    after = np.zeros(K, dtype=int)
    for vi, s in fw:
        lab = lab_arr[vi]
        center = min(s + ds.frame_window // 2, len(lab) - 1)
        after[int(lab[center])] += 1
    print(f"{'Class':<12}{'原始帧数':>12}{'占比':>9}{'过采样窗口':>12}{'占比':>9}")
    for c in range(K):
        print(f"{CLASS_NAMES[c]:<12}{before[c]:>12}{before[c]/before.sum():>8.1%}"
              f"{after[c]:>12}{after[c]/max(after.sum(),1):>8.1%}", flush=True)
    print(f"\n过采样后每类窗口数: {dict(zip(CLASS_NAMES, after.tolist()))}")
    print("说明: 稀有类(FaceGroom/BodyGroom)补到平均值, 常见类(Locomote/Rear/background)保留原量。")


if __name__ == "__main__":
    main()
