# -*- coding: utf-8 -*-
"""YOLO 裁剪预处理 (Swin-Panda 式, 用用户预训练 best.pt 检测小鼠主体)。

GPU 批次推理 + 增量保存(可断点续跑) + 检测失败降级(视频跳过)。
对每视频每 stride 帧检测, 取空间分位数得"稳定正方形裁剪框",
存 outputs/meta/yolo_boxes.json, 供 data/dataset.py 裁剪放大到 224。

用法:
  D:/Anaconda3/envs/deepgroomnet3/python.exe scripts/05_yolo_crop.py \
      --weights C:/Users/WIN10/PycharmProjects/mouseaction/DeepBehavior/runs/detect/train/weights/best.pt \
      --stride 60 --margin 0.15 --conf 0.3
"""
import os, sys, json, argparse
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def stable_box(boxes, margin):
    """对一段视频所有检测框, 取空间分位数 + 边长裕量, 得一个稳定正方形裁剪框。"""
    if len(boxes) == 0:
        return None
    b = np.array(boxes)                     # [N,4] x1,y1,x2,y2
    x1, y1 = np.percentile(b[:, 0], 5), np.percentile(b[:, 1], 5)
    x2, y2 = np.percentile(b[:, 2], 95), np.percentile(b[:, 3], 95)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    x1 = cx - side / 2; y1 = cy - side / 2
    x2 = cx + side / 2; y2 = cy + side / 2
    return [float(x1), float(y1), float(x2), float(y2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    default=r"C:\Users\WIN10\PycharmProjects\mouseaction\DeepBehavior\runs\detect\train\weights\best.pt")
    ap.add_argument("--meta", default=os.path.join(ROOT, "outputs", "meta"))
    ap.add_argument("--stride", type=int, default=60, help="每多少帧检测一次 (越大越快)")
    ap.add_argument("--margin", type=float, default=0.15, help="裁剪边长裕量比例")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=8, help="GPU 批次大小")
    args = ap.parse_args()

    import warnings; warnings.filterwarnings("ignore")
    import torch
    from ultralytics import YOLO
    model = YOLO(args.weights)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[yolo] device={device}, stride={args.stride}, conf={args.conf}")

    videos = json.load(open(os.path.join(args.meta, "videos.json"), encoding="utf-8"))
    out_path = os.path.join(args.meta, "yolo_boxes.json")
    boxes_map = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}

    done = set(boxes_map.keys())
    for vi, v in enumerate(videos):
        key = os.path.splitext(os.path.basename(v["video"]))[0]
        if key in done and boxes_map.get(key) is not None:
            print(f"[{vi+1}/{len(videos)}] {key}: (skip, done)")
            continue
        cap = cv2.VideoCapture(v["video"])
        if not cap.isOpened():
            print(f"[{vi+1}/{len(videos)}] {key}: OPEN FAIL")
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = list(range(0, max(1, n), args.stride))
        boxes = []
        # 批次推理
        for s in range(0, len(idxs), args.batch):
            ch = idxs[s:s + args.batch]
            frames = []
            for i in ch:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ret, fr = cap.read()
                if not ret:
                    continue
                frames.append(fr)
            if not frames:
                continue
            res = model(frames, verbose=False, device=device)[0] if len(frames) == 1 else \
                  model(frames, verbose=False, device=device)
            for r in res:
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    if c >= args.conf:
                        boxes.append(b.tolist())
        cap.release()
        sb = stable_box(boxes, args.margin)
        boxes_map[key] = sb
        # 每视频后立即保存 (断点续跑)
        json.dump(boxes_map, open(out_path, "w"), ensure_ascii=False)
        print(f"[{vi+1}/{len(videos)}] {key}: 检测框={len(boxes)} 稳定框={sb and [round(x,1) for x in sb]}")

    ok = sum(1 for x in boxes_map.values() if x)
    print(f"-> {out_path}  ({ok}/{len(boxes_map)} 视频有稳定框)")


if __name__ == "__main__":
    main()