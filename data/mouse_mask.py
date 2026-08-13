import numpy as np
import cv2

def median_background(gray):
    return np.median(gray, axis=0)

def compute_mouse_mask(frames, k_thresh=0.35, dilate_k=7, use_center_union=True, center_margin=0.12, min_coverage=0.05, max_coverage=0.9):
    gray = frames.mean(axis=3).astype(np.float32) / 255.0 if frames.max() > 1.0 else frames.mean(axis=3).astype(np.float32)
    T = gray.shape[0]
    if T < 3:
        (h, w) = gray.shape[1:]
        return _center_mask(h, w, center_margin)
    bg = median_background(gray)
    diff = np.abs(gray - bg[None]).mean(axis=0)
    th = diff.mean() + k_thresh * (diff.std() + 1e-06)
    mask = (diff > th).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    (n, lab, stats, _) = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(np.argmax(areas)) + 1
        mask = (lab == largest).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((dilate_k, dilate_k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cov = mask.mean()
    (h, w) = mask.shape
    center = _center_mask(h, w, center_margin)
    if use_center_union:
        mask = np.maximum(mask, center).astype(np.uint8)
    if cov < min_coverage or cov > max_coverage:
        mask = center
    return mask.astype(np.float32)

def _center_mask(h, w, margin=0.12):
    m = np.zeros((h, w), np.float32)
    (y0, y1) = (int(h * margin), int(h * (1 - margin)))
    (x0, x1) = (int(w * margin), int(w * (1 - margin)))
    m[y0:y1, x0:x1] = 1.0
    return m

def mask_coverage_ratio(mask):
    return float(mask.mean())
