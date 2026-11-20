import numpy as np


def macro_ovr_auroc(probs, labels_onehot):
    from sklearn.metrics import roc_auc_score
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels_onehot, dtype=int)
    scores = []
    for index in range(p.shape[1]):
        if y[:, index].min() == y[:, index].max():
            continue
        scores.append(roc_auc_score(y[:, index], p[:, index]))
    return float(np.mean(scores)), scores


def bootstrap_auroc(probs, labels_onehot, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(probs)
    values = []
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        value, _ = macro_ovr_auroc(probs[index], labels_onehot[index])
        values.append(value)
    return float(np.mean(values)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def label_shuffle_p(probs, labels_onehot, n_shuffle=200, seed=0):
    observed, _ = macro_ovr_auroc(probs, labels_onehot)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_shuffle):
        shifted = np.roll(labels_onehot, rng.integers(1, len(labels_onehot)), axis=0)
        value, _ = macro_ovr_auroc(probs, shifted)
        null.append(value)
    null = np.asarray(null)
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))
