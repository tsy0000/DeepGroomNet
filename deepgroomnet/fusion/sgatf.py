import numpy as np


def runs_of(syllables):
    s = np.asarray(syllables)
    changes = np.flatnonzero(np.diff(s) != 0) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(s)]
    return starts, ends


def run_means(probs, syllables):
    p = np.asarray(probs, dtype=np.float64)
    out = np.empty_like(p)
    starts, ends = runs_of(syllables)
    for a, b in zip(starts, ends):
        out[a:b] = p[a:b].mean(axis=0)
    return out


def adaptive_weights(probs, syllables, alpha_max=1.0, max_run=12,
                     consensus_floor=0.35, consensus_span=0.50):
    p = np.asarray(probs, dtype=np.float64)
    alpha = np.zeros(len(p), dtype=np.float64)
    starts, ends = runs_of(syllables)
    for a, b in zip(starts, ends):
        length = max(int(b - a), 1)
        consensus = float(p[a:b].max(axis=1).mean())
        gate = np.clip((consensus - consensus_floor) / consensus_span, 0.0, 1.0)
        alpha[a:b] = alpha_max * min(1.0, max_run / length) * gate
    return alpha


def binary_predictions(probs, thresholds, background_index=4):
    p = np.asarray(probs, dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if p.ndim != 2 or p.shape[1] != len(thr):
        raise ValueError("probs must be [n_frames, n_classes] matching thresholds")
    pred = (p > thr[None, :]).astype(np.int64)
    others = [i for i in range(p.shape[1]) if i != background_index]
    pred[:, background_index] = (~np.any(pred[:, others], axis=1)).astype(np.int64)
    return pred


def class_predictions(probs, binary, background_index=4):
    p = np.asarray(probs, dtype=np.float64)
    pred = np.asarray(binary, dtype=np.int64)
    out = np.full(len(p), background_index, dtype=np.int64)
    for i in range(len(p)):
        if pred[i, background_index] == 0:
            positive = np.flatnonzero(pred[i] > 0)
            if len(positive):
                out[i] = int(positive[np.argmax(p[i, positive])])
    return out


def sgatf(probs, syllables, thresholds, alpha_max=1.0, max_run=12,
          consensus_floor=0.35, consensus_span=0.50, background_index=4):
    p = np.asarray(probs, dtype=np.float64)
    s = np.asarray(syllables)
    if len(s) != len(p):
        raise ValueError("syllables and probs differ in length")
    alpha = adaptive_weights(p, s, alpha_max, max_run, consensus_floor, consensus_span)
    fused = (1.0 - alpha[:, None]) * p + alpha[:, None] * run_means(p, s)
    labels = class_predictions(fused, binary_predictions(fused, thresholds, background_index),
                               background_index)
    return labels, fused, alpha
