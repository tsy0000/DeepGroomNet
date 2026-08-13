import numpy as np
from sklearn.metrics import f1_score

def sigmoid_probs(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -80.0, 80.0)))

def threshold_predict(probs: np.ndarray, thresholds: np.ndarray, bg_idx: int=0) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    t = np.asarray(thresholds, dtype=np.float64)
    active = p >= t[None, :]
    n_active = active.sum(axis=1)
    pred = np.full(p.shape[0], bg_idx, dtype=np.int64)
    any_active = n_active > 0
    if any_active.any():
        best = np.argmax(np.where(active, p, -np.inf), axis=1)
        pred[any_active] = best[any_active]
    return pred

def tune_thresholds(val_probs: np.ndarray, val_labels: np.ndarray, bg_idx: int=0, candidates: int=40, passes: int=2, min_f1: float=0.05, seed: int=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    label_ids = np.unique(val_labels)
    thresholds = np.full(val_probs.shape[1], 0.5, dtype=np.float64)
    for _ in range(passes):
        for c in sorted(label_ids):
            if c == bg_idx:
                continue
            grid = np.linspace(0.05, 0.95, candidates)
            rng.shuffle(grid)
            (best_t, best_f1) = (0.5, -1.0)
            for t in grid:
                th = thresholds.copy()
                th[c] = t
                pred = threshold_predict(val_probs, th, bg_idx)
                f1 = f1_score(val_labels, pred, average='macro', zero_division=0)
                if f1 > best_f1:
                    (best_t, best_f1) = (t, f1)
            thresholds[c] = best_t
    thresholds = np.clip(thresholds, 0.01, None)
    per_class = f1_score(val_labels, threshold_predict(val_probs, thresholds, bg_idx), average=None, zero_division=0, labels=np.arange(val_probs.shape[1]))
    thresholds[per_class < min_f1] = 0.5
    return thresholds
