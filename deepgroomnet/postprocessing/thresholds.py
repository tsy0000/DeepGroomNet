import numpy as np

def binary_metrics(y_true, probs, class_names, threshold=0.5):
    yt = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    K = p.shape[1]
    pred = (p >= threshold).astype(int)
    eye = (yt[:, None] == np.arange(K)).astype(int)
    per_class = {}
    for (i, c) in enumerate(class_names):
        t = eye[:, i]
        pr = pred[:, i]
        tp = int(((t == 1) & (pr == 1)).sum())
        fn = int(((t == 1) & (pr == 0)).sum())
        fp = int(((t == 0) & (pr == 1)).sum())
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        pre = tp / (tp + fp) if tp + fp > 0 else 0.0
        f1 = 2 * pre * rec / (pre + rec) if pre + rec > 0 else 0.0
        per_class[c] = {'precision': pre * 100, 'recall': rec * 100, 'f1': f1 * 100}
    f1s = [per_class[c]['f1'] for c in class_names]
    return ({'f1_macro': float(np.mean(f1s))}, per_class)

def evaluate_thresholds(y_true, probs, class_names, bg_idx=None):
    yt = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    K = p.shape[1]
    eye = (yt[:, None] == np.arange(K)).astype(int)
    thresholds = np.linspace(0.0001, 1, 200)
    (best_t, best_f1s) = ({}, {})
    for (i, c) in enumerate(class_names):
        t = eye[:, i]
        (best_t_i, best_f1) = (0.5, -1.0)
        for thr in thresholds:
            pr = (p[:, i] >= thr).astype(int)
            tp = int(((t == 1) & (pr == 1)).sum())
            fn = int(((t == 1) & (pr == 0)).sum())
            fp = int(((t == 0) & (pr == 1)).sum())
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            pre = tp / (tp + fp) if tp + fp > 0 else 0.0
            f1 = 2 * pre * rec / (pre + rec) if pre + rec > 0 else 0.0
            if f1 > best_f1:
                (best_f1, best_t_i) = (f1, thr)
        if best_t_i < 0.01:
            best_t_i = 0.01
        if best_f1 < 0.05:
            best_t_i = 0.5
        (best_t[c], best_f1s[c]) = (best_t_i, best_f1)
    return (best_t, np.mean(list(best_f1s.values())))

def threshold_predict(probs, thresholds, class_names):
    p = np.asarray(probs, dtype=np.float64)
    K = p.shape[1]
    pred = np.zeros_like(p, dtype=int)
    for (i, c) in enumerate(class_names):
        pred[:, i] = (p[:, i] >= thresholds.get(c, 0.5)).astype(int)
    return pred

def compute_background(pred, bg_idx=4):
    out = pred.copy()
    if bg_idx is not None:
        others = pred[:, [i for i in range(pred.shape[1]) if i != bg_idx]].any(axis=1)
        out[:, bg_idx] = (~others).astype(int)
    return out

def get_bout_length_percentile(y_true, class_names, percentile=1):
    yt = np.asarray(y_true, dtype=np.int64)
    K = len(class_names)
    bout_len = {}
    for (i, c) in enumerate(class_names):
        seg = (yt == i).astype(int)
        bouts = []
        run = 0
        for v in seg:
            if v:
                run += 1
            else:
                if run > 0:
                    bouts.append(run)
                run = 0
        if run > 0:
            bouts.append(run)
        bout_len[c] = max(1, int(np.percentile(bouts, percentile))) if bouts else 1
    return bout_len

def remove_short_bouts(trace, bout_length):
    t = trace.astype(int).copy()
    for bl in range(1, bout_length + 1):
        t = _fill_holes(t, bl)
    for bl in range(1, bout_length + 1):
        t = _remove_short(t, bl)
    return t

def _fill_holes(trace, bl):
    t = trace.copy()
    i = 0
    n = len(t)
    while i < n:
        if t[i] == 0:
            j = i
            while j < n and t[j] == 0:
                j += 1
            if 1 <= j - i <= bl and i > 0 and (j < n) and (t[i - 1] == 1) and (t[j] == 1):
                t[i:j] = 1
            i = j
        else:
            i += 1
    return t

def _remove_short(trace, bl):
    t = trace.copy()
    i = 0
    n = len(t)
    while i < n:
        if t[i] == 1:
            j = i
            while j < n and t[j] == 1:
                j += 1
            if j - i <= bl:
                t[i:j] = 0
            i = j
        else:
            i += 1
    return t

def min_bout_per_behavior(prob_pred, y_true, class_names, percentile=1, bg_idx=4):
    bout_len = get_bout_length_percentile(y_true, class_names, percentile)
    out = prob_pred.copy()
    for (i, c) in enumerate(class_names):
        if i == bg_idx:
            continue
        out[:, i] = remove_short_bouts(out[:, i], bout_len[c])
    return out

def smooth_probabilities(P, window=3):
    P = np.asarray(P, dtype=np.float64)
    K = P.shape[1]
    n = P.shape[0]
    if window <= 1:
        return P
    w = int(window)
    out = np.zeros_like(P)
    for i in range(n):
        (lo, hi) = (max(0, i - w // 2), min(n, i + w // 2 + 1))
        out[i] = P[lo:hi].mean(axis=0)
    return out

def median_smooth_trace(trace, k=3):
    t = np.asarray(trace, dtype=int)
    if k <= 1:
        return t.copy()
    from scipy.ndimage import median_filter
    try:
        return median_filter(t, size=k, mode='nearest')
    except ImportError:
        out = t.copy()
        n = len(t)
        for i in range(n):
            (lo, hi) = (max(0, i - k // 2), min(n, i + k // 2 + 1))
            out[i] = 1 if t[lo:hi].mean() >= 0.5 else 0
        return out

def hmm_smooth(prob_pred, self_trans=0.9, n_iters=1):
    P = np.asarray(prob_pred, dtype=np.float64)
    K = P.shape[1]
    w = 3 if self_trans >= 0.9 else 1
    Ps = smooth_probabilities(P, w)
    thr = 0.5
    pred = (Ps >= thr).astype(int)
    k = 3
    for _ in range(n_iters):
        pred = np.stack([median_smooth_trace(pred[:, i], k) for i in range(K)], axis=1)
    return pred

def bout_stats(labels, class_names, bg_idx=4):
    labels = np.asarray(labels)
    (N, K) = labels.shape
    stats = {}
    for (i, c) in enumerate(class_names):
        if i == bg_idx:
            continue
        seg = labels[:, i]
        runs = []
        run = 0
        for v in seg:
            if v:
                run += 1
            else:
                if run > 0:
                    runs.append(run)
                run = 0
        if run > 0:
            runs.append(run)
        durations = np.asarray(runs, dtype=float)
        stats[c] = {'n_bouts': len(runs), 'total_frames': int(seg.sum()), 'mean_duration': float(durations.mean()) if len(durations) else 0.0, 'median_duration': float(np.median(durations)) if len(durations) else 0.0, 'durations': durations.tolist()}
    return stats

def syllable_boundaries(syllable):
    syl = np.asarray(syllable)
    n = len(syl)
    bd = np.zeros(n, dtype=np.int32)
    bd[1:] = (np.diff(syl) != 0).astype(np.int32)
    bd[0] = 1
    return bd

def moseq_boundary_refine(pred, syllable, pairs=((2, 3),)):
    boundaries = syllable_boundaries(syllable)
    pred = np.asarray(pred).copy()
    N = pred.shape[0]
    prev = None
    for i in range(N):
        if prev is not None and (not boundaries[i]):
            for (a, b) in pairs:
                if pred[i, a] and pred[i, b]:
                    if prev[a] == 1 and prev[b] == 0:
                        (pred[i, a], pred[i, b]) = (1, 0)
                    elif prev[b] == 1 and prev[a] == 0:
                        (pred[i, a], pred[i, b]) = (0, 1)
                    else:
                        (pred[i, a], pred[i, b]) = (0, 0)
        prev = pred[i]
    return pred
