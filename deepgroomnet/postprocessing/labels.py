import csv
import numpy as np
from deepgroomnet.fusion.sgatf import sgatf
from deepgroomnet.postprocessing.thresholds import (threshold_predict, compute_background,
                                                    smooth_probabilities)


def probs_to_labels(probs, thresholds, class_names, background_index=4,
                    smooth_window=0, syllables=None, use_sgatf=False, sgatf_cfg=None):
    p = np.asarray(probs, dtype=np.float64)
    if smooth_window and smooth_window > 1:
        p = smooth_probabilities(p, int(smooth_window))
    alpha = None
    if use_sgatf and syllables is not None:
        labels, p, alpha = sgatf(p, syllables, thresholds, background_index=background_index,
                                 **(sgatf_cfg or {}))
    else:
        pred = threshold_predict(p, {c: thresholds[i] for i, c in enumerate(class_names)},
                                 list(class_names))
        pred = compute_background(pred, bg_idx=background_index)
        labels = np.full(len(p), background_index, dtype=int)
        for row in range(len(p)):
            positive = np.flatnonzero(pred[row])
            if len(positive):
                labels[row] = int(positive[np.argmax(p[row, positive])])
    return {"labels": labels, "probs": p, "alpha": alpha}


def write_ethogram(labels, class_names, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "behavior"])
        for index, value in enumerate(np.asarray(labels, dtype=int)):
            writer.writerow([index, class_names[value]])
