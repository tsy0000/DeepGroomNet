"""
关键点数据加载与几何特征（Sturman et al. 2020 方案，Neuropsychopharmacology 45:1942）。

数据：E:/tsy/keypointjson_to_csv/mouseXXX_superanimal_..._after_adapt.csv
  DLC superanimal_topviewmouse 输出（27 身体点 × (x,y,likelihood)），帧数与视频逐帧对齐。

特征（与裁剪/分辨率无关的几何归一化，Sturman 式）：
  - 平移归一化：所有点减 mouse_center（质心）
  - 尺度归一化：除 nose→tail_base 距离（小鼠自身大小）
  - 每点附 likelihood 有效性掩码（likelihood<0.5 的点坐标置 0，掩码=0）
输出 pose 张量 [T, K, 3]（x, y, valid），供 PoseBranch 融合。

判别信号（mouse021 实测）：
  nose-shoulder 距离：FG 49.9 < BG 52.1 < bg 59.7 < Locomote 63.9（前爪贴脸）
  nose 速度：FG 3.5 << BG 13.6（脸不动爪高频动 vs 身体舔舐）
  nose-tail 距离：BG 88.0 < FG 110.5 < bg 142.1 < Locomote 157.0（蜷缩 vs 伸展）
"""
from __future__ import annotations

import os
import threading

import numpy as np


def find_keypoint_csv(root: str, mouse: str):
    """按小鼠名前缀在 root 下找 keypoint CSV（DLC superanimal 命名 mouseXXX_*.csv）。"""
    if not root or not mouse:
        return None
    for fn in os.listdir(root):
        if fn.lower().startswith(mouse.lower()) and fn.lower().endswith(".csv"):
            return os.path.join(root, fn)
    return None


def load_keypoints_csv(path: str):
    """解析 DLC CSV -> (bodyparts, kp[N,K,3])；N 与视频帧数对齐。"""
    import csv
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 4:
        raise ValueError(f"keypoint CSV 过短: {path}")
    bps = [b for b in rows[1][1::3] if b]
    K = len(bps)
    data = np.array([[float(v) if v != "" else np.nan for v in r[1:]]
                     for r in rows[3:]], dtype=np.float64)
    n = data.shape[0]
    kp = np.nan_to_num(data.reshape(n, K, 3), nan=0.0)
    return bps, kp


class KeypointBank:
    """按小鼠名惰性加载 keypoint CSV 并缓存；线程安全 + 可 pickle（DataLoader worker 用）。"""

    def __init__(self, root: str):
        self.root = root or ""
        self._cache = {}
        self._lock = threading.Lock()
        self.bodyparts = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def _mouse_of(self, video_path: str):
        return os.path.basename(os.path.dirname(video_path))

    def _mouse_of(self, video_path: str):
        return self._mouse_from_video(video_path)

    def get(self, mouse: str):
        """按小鼠名返回 (bodyparts, kp[N,K,3])；无 keypoint 文件时返回 (None, None)。"""
        with self._lock:
            if mouse in self._cache:
                return self._cache[mouse]
        path = find_keypoint_csv(self.root, mouse)
        if path is None:
            with self._lock:
                self._cache[mouse] = (None, None)
            return None, None
        bps, kp = load_keypoints_csv(path)
        self.bodyparts = bps
        with self._lock:
            self._cache[mouse] = (bps, kp)
        return bps, kp

    def has(self, mouse: str) -> bool:
        bps, _ = self.get(mouse)
        return bps is not None


HEAD_PARTS = {"nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip",
              "left_eye", "right_eye", "head_midpoint"}
BODY_PARTS = {"neck", "mid_back", "mouse_center", "mid_backend", "mid_backend2",
              "mid_backend3", "tail_base", "tail1", "tail2", "tail3", "tail4",
              "tail5", "tail_end", "left_shoulder", "left_midside", "left_hip",
              "right_shoulder", "right_midside", "right_hip"}


def head_body_mask(bodyparts, use_parts=None):
    """返回 (head_mask, body_mask)：[K] bool，按 bodyparts 顺序。"""
    keep_names = use_parts or bodyparts
    hm = np.array([n in HEAD_PARTS for n in keep_names], dtype=bool)
    bm = np.array([n in BODY_PARTS for n in keep_names], dtype=bool)
    return hm, bm


def render_keypoint_maps(pose, head_mask, body_mask, size=224, sigma_px=12.0, scale_px=40.0):
    """归一化 pose [T,K,3] -> 头部/身体高斯热图 [T,2,size,size]。

    归一化坐标（中心=mouse_center，nose→tail 距离=1）渲染到输入分辨率：
      像素位置 = 中心 + 归一化坐标 × scale_px（nose→tail 距离≈40px），
      与 YOLO 裁剪后小鼠居中的设定一致；scale 固定 → 与裁剪/分辨率无关。
    输出 [T,2,H,W]：channel0=头部点（nose/耳/眼/head_midpoint），
    channel1=身体点（neck/背/尾/肩/髋）——注意力门可直接利用"头在哪/身在哪"。
    无有效关键点的帧 → 全零（该帧无先验，门不受影响）。
    """
    import cv2
    T, K, _ = pose.shape
    maps = np.zeros((T, 2, size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for t in range(T):
        for g, mask in enumerate((head_mask, body_mask)):
            pts = pose[t][mask]
            if len(pts) == 0:
                continue
            valid = pts[:, 2] > 0.5
            if not valid.any():
                continue
            for x, y, v in pts[valid]:
                px = size / 2.0 + x * scale_px
                py = size / 2.0 + y * scale_px
                d2 = (xx - px) ** 2 + (yy - py) ** 2
                maps[t, g] = np.maximum(maps[t, g],
                                        np.exp(-d2 / (2 * sigma_px ** 2)).astype(np.float32))
    return maps


def pose_features(kp, frame_indices, bodyparts, use_parts=None, scale_parts=("nose", "tail_base"),
                  center_part="mouse_center", min_lik=0.5):
    """frame_indices: 绝对视频帧号 -> pose [T, K, 3]（x, y, valid），归一化几何特征。

    kp: [N, K, 3]（来自 KeypointBank）。无关键点时调用方传 kp=None → 返回 None。
    归一化：中心=mouse_center，尺度=nose→tail_base 距离（旋转不变由网络学）。
    """
    if kp is None:
        return None
    K = len(bodyparts)
    idx = {n: i for i, n in enumerate(bodyparts)}
    if use_parts is not None:
        keep = [i for n, i in idx.items() if n in use_parts]
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
    fallback = np.divide(xy.sum(axis=1), valid.sum(axis=1, keepdims=True) + 1e-6)
    c = np.where(c_valid[:, None], c, fallback)
    xy = xy - c[:, None, :]
    xy = np.where(valid[..., None], xy, 0.0)

    s = None
    for a, b in (scale_parts,):
        ia, ib = idx.get(a), idx.get(b)
        if ia is not None and ib is not None and ia in keep and ib in keep:
            ia, ib = keep.index(ia), keep.index(ib)
            s = np.linalg.norm(sel[:, ia, :2] - sel[:, ib, :2], axis=1)
            sv = (sel[:, ia, 2] >= min_lik) & (sel[:, ib, 2] >= min_lik)
            s = np.where(sv & (s > 1e-3), s, np.nan)
            break
    if s is None:
        s = np.nan * np.ones(len(fi))
    s = np.nan_to_num(s, nan=np.nanmedian(s)) if np.isfinite(np.nanmedian(s)) else np.ones(len(fi))
    s = np.where(s > 1e-3, s, 1.0)
    xy = xy / s[:, None, None]

    out = np.concatenate([xy, valid[:, :, None].astype(np.float32)], axis=-1)
    return out.astype(np.float32)
