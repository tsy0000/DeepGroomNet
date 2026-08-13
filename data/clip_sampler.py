from __future__ import annotations
import os
from typing import List, Tuple
import numpy as np
OPENFIELD_CLASSES = ['Locomote', 'Rear', 'FaceGroom', 'BodyGroom', 'background']

def _cv2():
    import cv2
    return cv2

def sample_frames(start: int, end: int, n_frames: int) -> List[int]:
    n = end - start
    if n <= 0:
        return []
    if n_frames <= 0:
        return []
    if n == 1:
        return [start]
    idx = np.linspace(start, end - 1, num=n_frames).round().astype(int)
    return [int(i) for i in np.unique(idx)]

def read_frames(video_path: str, frame_indices: List[int], fallback_size=(480, 480, 3)):
    cv2 = _cv2()
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in frame_indices:
        if i >= total or i < 0:
            frames.append(np.zeros(fallback_size, np.uint8))
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        (ok, fr) = cap.read()
        frames.append(fr if ok else np.zeros(fallback_size, np.uint8))
    cap.release()
    return frames

def video_frame_count(video_path: str) -> int:
    cv2 = _cv2()
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n

def normalize_label_columns(labels_df):
    df = labels_df.copy()
    rename = {}
    for c in df.columns:
        if c in ('Background', 'background') and c != 'background':
            rename[c] = 'background'
    if rename:
        df = df.rename(columns=rename)
    return df

def label_of_window(labels_df, start: int, end: int) -> str | None:
    sl = labels_df.iloc[start:end]
    for c in OPENFIELD_CLASSES:
        if c in sl.columns and int(sl[c].sum()) == len(sl):
            return c
    return None

def segment_to_windows(segments: List[Tuple[int, int]], seg: int, step: int=None) -> List[Tuple[int, int]]:
    if step is None:
        step = seg
    windows = []
    for (i, j) in segments:
        s = i
        while s + seg <= j:
            windows.append((s, s + seg))
            s += step
        if s < j:
            lo = max(i, j - seg)
            windows.append((lo, j))
    return windows

def extract_uniform_clips(video_path, labels_df, cfg, class_names=OPENFIELD_CLASSES):
    total = len(labels_df)
    seg = cfg['CLIP']['orig_segment_len']
    interval = cfg['CLIP']['clip_interval']
    clips = []
    for s in range(0, total - seg + 1, interval):
        e = s + seg
        lab = label_of_window(labels_df, s, e)
        if lab is not None:
            clips.append((s, e, lab))
    return clips

def extract_sliding_clips(video_path, labels_df, cfg, count, class_names=OPENFIELD_CLASSES, seed=None):
    total = len(labels_df)
    seg = cfg['CLIP']['orig_segment_len']
    rng = np.random.default_rng(seed)
    clips = []
    for _ in range(count):
        s = int(rng.integers(0, max(1, total - seg)))
        e = s + seg
        center = s + seg // 2
        row = labels_df.iloc[center]
        vals = [float(row[c]) if c in row.index else 0.0 for c in class_names]
        if abs(sum(vals) - 1.0) > 1e-06:
            continue
        clips.append((s, e, class_names[int(np.argmax(vals))]))
    return clips

def jitter_window(start: int, end: int, total: int, jit: int, rng) -> tuple:
    seg = end - start
    if jit <= 0 or seg <= 0:
        return (start, end)
    lo = max(0, start - jit)
    hi = min(total - seg, start + jit)
    if hi < lo:
        return (start, end)
    s = int(rng.integers(lo, hi + 1))
    return (s, s + seg)

def homecage_clips(clip_path: str, cfg, class_names):
    n = video_frame_count(clip_path)
    seg = cfg['CLIP']['orig_segment_len']
    clips = []
    s = 0
    while s + seg <= n:
        clips.append((s, s + seg))
        s += seg
    if n - s > 0 and n - s < seg:
        clips.append((max(0, n - seg), n))
    return clips

def infer_sliding_windows(total_frames: int, cfg) -> List[Tuple[int, int]]:
    seg = int(cfg['CLIP']['orig_segment_len'])
    interval = int(cfg['CLIP'].get('clip_interval', 15))
    if interval <= 0:
        interval = seg
    wins = []
    s = 0
    while s + seg <= total_frames:
        wins.append((s, s + seg))
        s += interval
    if s < total_frames:
        wins.append((max(0, total_frames - seg), total_frames))
    return wins

def infer_syllable_clips(syllable_labels: np.ndarray, cfg) -> List[Tuple[int, int]]:
    seg = cfg['CLIP']['orig_segment_len']
    syl = np.asarray(syllable_labels)
    n = len(syl)
    clips = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and syl[j] == syl[i]:
            j += 1
        s = i
        while s + seg <= j:
            clips.append((s, s + seg))
            s += seg
        if s < j:
            lo = max(i, j - seg)
            clips.append((lo, j))
        i = j
    return clips
