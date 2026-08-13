from __future__ import annotations
import os
import threading
import numpy as np

def find_keypoint_csv(root: str, mouse: str):
    if not root or not mouse:
        return None
    for fn in os.listdir(root):
        if fn.lower().startswith(mouse.lower()) and fn.lower().endswith('.csv'):
            return os.path.join(root, fn)
    return None

def load_keypoints_csv(path: str):
    import csv
    with open(path, encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if len(rows) < 4:
        raise ValueError(f'keypoint CSV 过短: {path}')
    bps = [b for b in rows[1][1::3] if b]
    K = len(bps)
    data = np.array([[float(v) if v != '' else np.nan for v in r[1:]] for r in rows[3:]], dtype=np.float64)
    n = data.shape[0]
    kp = np.nan_to_num(data.reshape(n, K, 3), nan=0.0)
    return (bps, kp)

class KeypointBank:

    def __init__(self, root: str):
        self.root = root or ''
        self._cache = {}
        self._lock = threading.Lock()
        self.bodyparts = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_lock', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def _mouse_of(self, video_path: str):
        return os.path.basename(os.path.dirname(video_path))

    def _mouse_of(self, video_path: str):
        return self._mouse_from_video(video_path)

    def get(self, mouse: str):
        with self._lock:
            if mouse in self._cache:
                return self._cache[mouse]
        path = find_keypoint_csv(self.root, mouse)
        if path is None:
            with self._lock:
                self._cache[mouse] = (None, None)
            return (None, None)
        (bps, kp) = load_keypoints_csv(path)
        self.bodyparts = bps
        with self._lock:
            self._cache[mouse] = (bps, kp)
        return (bps, kp)

    def has(self, mouse: str) -> bool:
        (bps, _) = self.get(mouse)
        return bps is not None
HEAD_PARTS = {'nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint'}
BODY_PARTS = {'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end', 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip'}

def head_body_mask(bodyparts, use_parts=None):
    keep_names = use_parts or bodyparts
    hm = np.array([n in HEAD_PARTS for n in keep_names], dtype=bool)
    bm = np.array([n in BODY_PARTS for n in keep_names], dtype=bool)
    return (hm, bm)

def render_keypoint_maps(pose, head_mask, body_mask, size=224, sigma_px=12.0, scale_px=40.0):
    import cv2
    (T, K, _) = pose.shape
    maps = np.zeros((T, 2, size, size), dtype=np.float32)
    (yy, xx) = np.mgrid[0:size, 0:size].astype(np.float32)
    for t in range(T):
        for (g, mask) in enumerate((head_mask, body_mask)):
            pts = pose[t][mask]
            if len(pts) == 0:
                continue
            valid = pts[:, 2] > 0.5
            if not valid.any():
                continue
            for (x, y, v) in pts[valid]:
                px = size / 2.0 + x * scale_px
                py = size / 2.0 + y * scale_px
                d2 = (xx - px) ** 2 + (yy - py) ** 2
                maps[t, g] = np.maximum(maps[t, g], np.exp(-d2 / (2 * sigma_px ** 2)).astype(np.float32))
    return maps

def pose_features(kp, frame_indices, bodyparts, use_parts=None, scale_parts=('nose', 'tail_base'), center_part='mouse_center', min_lik=0.5):
    if kp is None:
        return None
    K = len(bodyparts)
    idx = {n: i for (i, n) in enumerate(bodyparts)}
    if use_parts is not None:
        keep = [i for (n, i) in idx.items() if n in use_parts]
        K = len(keep)
    else:
        keep = list(range(K))
    fi = np.asarray(frame_indices, dtype=np.int64)
    fi = np.clip(fi, 0, kp.shape[0] - 1)
    sel = kp[fi][:, keep, :]
    valid = sel[:, :, 2] >= min_lik
    xy = np.where(valid[..., None], sel[:, :, :2], 0.0)
    ci = idx.get(center_part)
    if ci is not None and ci in keep:
        c = sel[:, keep.index(ci), :2]
        c_valid = sel[:, keep.index(ci), 2] >= min_lik
    else:
        c = xy.mean(axis=1)
        c_valid = np.ones(len(fi), bool)
    fallback = np.divide(xy.sum(axis=1), valid.sum(axis=1, keepdims=True) + 1e-06)
    c = np.where(c_valid[:, None], c, fallback)
    xy = xy - c[:, None, :]
    xy = np.where(valid[..., None], xy, 0.0)
    s = None
    for (a, b) in (scale_parts,):
        (ia, ib) = (idx.get(a), idx.get(b))
        if ia is not None and ib is not None and (ia in keep) and (ib in keep):
            (ia, ib) = (keep.index(ia), keep.index(ib))
            s = np.linalg.norm(sel[:, ia, :2] - sel[:, ib, :2], axis=1)
            sv = (sel[:, ia, 2] >= min_lik) & (sel[:, ib, 2] >= min_lik)
            s = np.where(sv & (s > 0.001), s, np.nan)
            break
    if s is None:
        s = np.nan * np.ones(len(fi))
    s = np.nan_to_num(s, nan=np.nanmedian(s)) if np.isfinite(np.nanmedian(s)) else np.ones(len(fi))
    s = np.where(s > 0.001, s, 1.0)
    xy = xy / s[:, None, None]
    out = np.concatenate([xy, valid[:, :, None].astype(np.float32)], axis=-1)
    return out.astype(np.float32)
