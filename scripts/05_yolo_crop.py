# -*- coding: utf-8 -*-
"""YOLO 裁剪预处理 (Swin-Panda 式): 裁到检测到的小鼠主体框 -> resize 224。

依据 Swin-Panda (Diversity 2025,17,139) 3.2 节:
  "By cropping and resizing the detected panda subjects to 224×224 pixels using YOLOv9,
   we effectively reduced the image size and standardized the input scale..."
即: 用 YOLO 检测目标(小鼠)主体框, 裁到该框(加少量边距), 再放大到 224。
旧版取"全程 5-95 分位稳定框"≈整帧/场地, 对 0.1% 的小目标不起裁剪放大作用, 已废弃。

本脚本: 对每视频每 stride 帧检测小鼠框, 以其中心 + 边长(margin 裕量)得正方形裁剪框,
存 outputs/meta/yolo_boxes.json 为 {key: {"stride": s, "boxes": [[x1,y1,x2,y2], ...]}}.
boxes[k] 对应帧索引 k*stride; 相邻帧用最近采样框。检测失败帧沿用上一有效框。

用法:
  D:/Anaconda3/envs/deepgroomnet3/python.exe scripts/05_yolo_crop.py \
      --weights C:/Users/WIN10/PycharmProjects/mouseaction/DeepBehavior/runs/detect/train/weights/best.pt \
      --stride 30 --margin 0.2 --conf 0.3
"""
import os, sys, json, argparse
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def square_box(box, margin, H, W):
    """以检测框中心为中心, 边长=max(w,h)*(1+2*margin) 的正方形, clip 到帧内。"""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    side = min(side, max(W, H))           # 避免超过画面
    bx1 = cx - side / 2; bx2 = cx + side / 2
    by1 = cy - side / 2; by2 = cy + side / 2
    # clip 到帧内 (保持正方形: 平移回界)
    if bx1 < 0: bx2 += -bx1; bx1 = 0
    if by1 < 0: by2 += -by1; by1 = 0
    if bx2 > W: bx1 -= (bx2 - W); bx2 = W
    if by2 > H: by1 -= (by2 - H); by2 = H
    bx1 = max(0, bx1); by1 = max(0, by1)
    bx2 = min(W, bx2); by2 = min(H, by2)
    return [float(bx1), float(by1), float(bx2), float(by2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    default=r"C:\Users\WIN10\PycharmProjects\mouseaction\DeepBehavior\runs\detect\train\weights\best.pt")
    ap.add_argument("--meta", default=os.path.join(ROOT, "outputs", "meta"))
    ap.add_argument("--stride", type=int, default=30, help="每多少帧检测一次 (越小越准但越慢)")
    ap.add_argument("--margin", type=float, default=0.2, help="裁剪边长裕量比例 (Swin-Panda 保留环境上下文)")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=8, help="GPU 批次大小")
    ap.add_argument("--subset", type=int, default=None, help="只处理前 N 个视频 (调试)")
    args = ap.parse_args()

    import warnings; warnings.filterwarnings("ignore")
    import torch
    from ultralytics import YOLO
    model = YOLO(args.weights)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[yolo] device={device}, stride={args.stride}, margin={args.margin}, conf={args.conf}")

    videos = json.load(open(os.path.join(args.meta, "videos.json"), encoding="utf-8"))
    out_path = os.path.join(args.meta, "yolo_boxes.json")
    boxes_map = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}

    done = set(boxes_map.keys())
    if args.subset:
        videos = videos[:args.subset]
    for vi, v in enumerate(videos):
        key = os.path.splitext(os.path.basename(v["video"]))[0]
        if key in done and isinstance(boxes_map.get(key), dict) and "boxes" in boxes_map[key]:
            print(f"[{vi+1}/{len(videos)}] {key}: (skip, done)")
            continue
        cap = cv2.VideoCapture(v["video"])
        if not cap.isOpened():
            print(f"[{vi+1}/{len(videos)}] {key}: OPEN FAIL")
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        idxs = list(range(0, n, args.stride))
        boxes = []                      # boxes[k] 对应 idxs[k]
        last_box = None
        for s in range(0, len(idxs), args.batch):
            ch = idxs[s:s + args.batch]
            frames = []
            for i in ch:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ret, fr = cap.read()
                if ret:
                    frames.append(fr)
            if not frames:
                continue
            res = model(frames, verbose=False, device=device)
            for r in res:
                best = None; best_c = args.conf
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    if c >= best_c:
                        best, best_c = b, c
                if best is not None:
                    last_box = best
                boxes.append(square_box(last_box, args.margin, H, W) if last_box is not None else None)
        cap.release()
        boxes_map[key] = {"stride": args.stride, "margin": args.margin, "boxes": boxes}
        json.dump(boxes_map, open(out_path, "w"), ensure_ascii=False)
        n_ok = sum(1 for b in boxes if b is not None)
        print(f"[{vi+1}/{len(videos)}] {key}: 帧={n} 采样框={len(boxes)} 有效={n_ok} "
              f"示例框={[round(x,1) for x in boxes[len(boxes)//2]] if boxes and boxes[len(boxes)//2] else None}")

    ok = sum(1 for x in boxes_map.values() if isinstance(x, dict) and x.get("boxes"))
    print(f"-> {out_path}  ({ok}/{len(boxes_map)} 视频有逐帧裁剪框)")


if __name__ == "__main__":
    main()
