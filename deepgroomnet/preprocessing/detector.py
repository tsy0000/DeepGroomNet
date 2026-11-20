import json
import cv2
import numpy as np


def load_boxes(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def box_at(boxes, key, frame_index):
    entry = boxes.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict) and "boxes" in entry:
        stride = int(entry.get("stride", 30))
        items = entry["boxes"]
        if not items:
            return None
        return items[min(max(int(round(frame_index / stride)), 0), len(items) - 1)]
    if isinstance(entry, list) and len(entry) == 4:
        return entry
    return None


def crop_to_box(frame, box, size, margin=0.2):
    if box is None:
        return cv2.resize(frame, (size, size))
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    x1 -= margin * w
    x2 += margin * w
    y1 -= margin * h
    y2 += margin * h
    height, width = frame.shape[:2]
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bx1 = int(round(cx - side / 2.0))
    by1 = int(round(cy - side / 2.0))
    bx2 = int(round(cx + side / 2.0))
    by2 = int(round(cy + side / 2.0))
    canvas = np.zeros((side and int(side) or 1, int(side) or 1, 3), np.uint8)
    sx1, sy1 = max(bx1, 0), max(by1, 0)
    sx2, sy2 = min(bx2, width), min(by2, height)
    if sx2 > sx1 and sy2 > sy1:
        canvas[sy1 - by1:sy2 - by1, sx1 - bx1:sx2 - bx1] = frame[sy1:sy2, sx1:sx2]
    return cv2.resize(canvas, (size, size))


def run_yolo_detector(video, weights, stride=30, margin=0.2, conf=0.3, batch=8):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("ultralytics is required for YOLO box extraction") from error
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, boxes = [], []
    for index in range(0, total, int(stride)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        result = model.predict(frame, conf=conf, verbose=False, device="cuda" if batch else "cpu")[0]
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            best = int(np.argmax(scores))
            boxes.append([float(v) for v in xyxy[best]])
        else:
            boxes.append([0.0, 0.0, float(frame.shape[1]), float(frame.shape[0])])
    cap.release()
    return {"stride": int(stride), "margin": float(margin), "boxes": boxes}
