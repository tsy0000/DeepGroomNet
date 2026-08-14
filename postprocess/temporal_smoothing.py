"""
模块B（Solution 2）推理端：时序平滑后处理。

训练 clip 来自人工标注段（段内一致），测试 clip 来自 MoSeq 音节边界，
因此帧级预测在音节边界附近会有孤立翻转。本模块提供两种经典平滑：

  - majority_vote(preds, window)：滑动窗口多数投票
  - median_filter(preds, window)：滑动中值滤波

参考：Wilson et al. 2026（Ecol Evol，PMID 42544294）"sequence-informed post-processing"
     的一般形式；以及行为分类领域常用的时序一致性平滑。
"""
import numpy as np
from collections import deque


def majority_vote(preds, window=5):
    """preds: np.ndarray [N]，多数投票平滑（窗口内取众数）。"""
    preds = np.asarray(preds, dtype=np.int64)
    n = len(preds)
    if n == 0:
        return preds
    out = preds.copy()
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        vals, counts = np.unique(preds[lo:hi], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def median_filter(preds, window=7):
    """滑动中值滤波（对类 id 排序中值）。"""
    preds = np.asarray(preds, dtype=np.int64)
    n = len(preds)
    if n == 0:
        return preds
    out = preds.copy()
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(preds[lo:hi]).astype(np.int64)
    return out


def smooth(preds, method="vote", window=5):
    method = (method or "none").lower()
    if method == "vote":
        return majority_vote(preds, window)
    if method == "median":
        return median_filter(preds, window)
    if method in ("none", ""):
        return np.asarray(preds, dtype=np.int64)
    raise ValueError(f"unknown smoothing: {method}")


def bout_filter(preds, min_len=10, bg_idx=0):
    """bout 级时序滤波（DeepEthogram kernel 的等价实现，eLife 2021）。

    两遍扫描（对帧级预测序列）：
      1) 删除 < min_len 帧的孤立行为片段 → 转背景（抑制 FG/BG 类假阳性——调研证实
         长尾行为类的误报多为 1-3 帧孤立翻转，Aso 2022 后处理即靠此把 FG/BG 拉到
         PPV 83-88%）。
      2) 填充 < min_len 帧的短背景洞（左右邻居同类时补成该类，修复 bout 内中断）。

    30fps 下 min_len=10 ≈ 333ms。preds: [N] 类 id 序列。返回同形数组。
    """
    preds = np.asarray(preds, dtype=np.int64).copy()
    n = len(preds)
    if n == 0:
        return preds

    def _runs():
        i = 0
        while i < n:
            j = i
            while j < n and preds[j] == preds[i]:
                j += 1
            yield i, j
            i = j

    for i, j in list(_runs()):
        if j - i <= min_len and preds[i] != bg_idx:
            preds[i:j] = bg_idx
    for i, j in list(_runs()):
        if j - i <= min_len and preds[i] == bg_idx:
            left = preds[i - 1] if i > 0 else -1
            right = preds[j] if j < n else -1
            if left == right and left >= 0:
                preds[i:j] = left
    return preds
