"""YOLO 裁剪预处理 (成熟做法, 参照 Swin-Panda 用 YOLO 裁到动物主体)。

对每段视频运行 YOLO 检测小鼠, 计算"稳定的裁剪框"(对检测框取分位数, 避免逐帧抖动),
把框存到 outputs/meta/yolo_boxes.json, 供 data/dataset.py 裁剪放大到 224。

成熟组件: ultralytics.YOLO (官方) + 用户预训练的 best.pt (mouse 类)。

用法:
  D:/Anaconda3/envs/deepgroomnet3/python.exe scripts/05_yolo_crop.py \
      --weights C:/Users/WIN10/PycharmProjects/mouseaction/DeepBehavior/runs/detect/train/weights/best.pt \
      --stride 30 --margin 0.15
"""
import os, sys, json, glob, argparse
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
    ap.add_argument("--stride", type=int, default=30, help="每多少帧检测一次")
    ap.add_argument("--margin", type=float, default=0.15, help="裁剪边长裕量比例")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    import warnings; warnings.filterwarnings("ignore")
    from ultralytics import YOLO
    model = YOLO(args.weights)

    videos = json.load(open(os.path.join(args.meta, "videos.json"), encoding="utf-8"))
    boxes_map = {}
    for vi, v in enumerate(videos):
        cap = cv2.VideoCapture(v["video"])
        if not cap.isOpened():
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        boxes = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            if idx % args.stride == 0:
                r = model(fr, verbose=False)[0]
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    if c >= args.conf:
                        boxes.append(b.tolist())
            idx += 1
        cap.release()
        sb = stable_box(boxes, args.margin)
        key = os.path.splitext(os.path.basename(v["video"]))[0]
        boxes_map[key] = sb
        print(f"[{vi+1}/{len(videos)}] {key}: 检测框={len(boxes)} 稳定框={sb and [round(x,1) for x in sb]}")

    out = os.path.join(args.meta, "yolo_boxes.json")
    json.dump(boxes_map, open(out, "w"), ensure_ascii=False)
    print(f"-> {out}  ({sum(1 for x in boxes_map.values() if x)}/{len(boxes_map)} 视频有稳定框)")


if __name__ == "__main__":
    main()
