import numpy as np
from collections import deque

def majority_vote(preds, window=5):
    preds = np.asarray(preds, dtype=np.int64)
    n = len(preds)
    if n == 0:
        return preds
    out = preds.copy()
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        (vals, counts) = np.unique(preds[lo:hi], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out

def median_filter(preds, window=7):
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

def smooth(preds, method='vote', window=5):
    method = (method or 'none').lower()
    if method == 'vote':
        return majority_vote(preds, window)
    if method == 'median':
        return median_filter(preds, window)
    if method in ('none', ''):
        return np.asarray(preds, dtype=np.int64)
    raise ValueError(f'unknown smoothing: {method}')
