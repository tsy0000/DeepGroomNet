import cv2
import numpy as np


def frame_count(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_video(path, size=224):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (size, size)))
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames)


def read_window(path, start, length, size=224):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    frames = []
    for _ in range(length):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (size, size)))
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, size, size, 3), np.uint8)
