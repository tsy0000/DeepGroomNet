"""
逐类 + 总体指标（accuracy / precision / recall / F1），F1 采用主流做法同时报告
macro-F1 与 weighted-F1（类别不平衡时 weighted 更能反映整体，macro 更能反映稀有类）。

同时提供边界误差计算（模块B 推理端评估用）。
"""
import csv
import os

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)


def classwise_and_overall(y_true, y_pred, class_names):
    """
    返回 (overall, per_class, cm)：
      overall: dict(acc, precision_macro, recall_macro, f1_macro, f1_weighted)
      per_class: {class_name: dict(acc, precision, recall, f1)}
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    n = len(class_names)
    cm = confusion_matrix(yt, yp, labels=list(range(n)))

    per_class = {}
    for i, c in enumerate(class_names):
        denom = cm[i].sum()
        t = (yt == i).astype(int)
        p = (yp == i).astype(int)
        per_class[c] = {
            "accuracy": float(cm[i, i] / denom * 100) if denom > 0 else 0.0,
            "recall": recall_score(t, p, zero_division=0) * 100,
            "precision": precision_score(t, p, zero_division=0) * 100,
            "f1": f1_score(t, p, zero_division=0) * 100,
            "n": int(denom),
        }

    overall = {
        "acc": accuracy_score(yt, yp) * 100,
        "precision_macro": precision_score(yt, yp, average="macro", zero_division=0) * 100,
        "recall_macro": recall_score(yt, yp, average="macro", zero_division=0) * 100,
        "f1_macro": f1_score(yt, yp, average="macro", zero_division=0) * 100,
        "f1_weighted": f1_score(yt, yp, average="weighted", zero_division=0) * 100,
    }
    return overall, per_class, cm


def binary_classwise_metrics(y_true, probs, class_names, threshold=0.5):
    """DeepEthogram 式多标签指标（eLife 2021 口径）。

    - per_class: 每类 one-vs-rest 的 recall/precision/f1（在 threshold 上二值化）
    - overall.acc: 元素级准确率（[N,C] 每个 (样本,类) 元素判对的比例，DeepEthogram
      原文 "accuracy = fraction of elements of the T,K ethogram predicted correctly"）
    - overall.f1_macro / f1_weighted
    返回 (overall, per_class, cm=None)。
    """
    yt = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    K = p.shape[1]
    pred = (p >= threshold).astype(int)
    eye = (yt[:, None] == np.arange(K)).astype(int)

    per_class = {}
    for i, c in enumerate(class_names):
        t = eye[:, i]
        pr = pred[:, i]
        tp = int(((t == 1) & (pr == 1)).sum())
        fn = int(((t == 1) & (pr == 0)).sum())
        fp = int(((t == 0) & (pr == 1)).sum())
        tn = int(((t == 0) & (pr == 0)).sum())
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
        per_class[c] = {
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1) * 100,
            "recall": rec * 100,
            "precision": pre * 100,
            "f1": f1 * 100,
            "n": int((yt == i).sum()),
        }

    counts = np.array([(yt == i).sum() for i in range(K)], dtype=np.float64)
    f1s = np.array([per_class[c]["f1"] for c in class_names])
    pres = np.array([per_class[c]["precision"] for c in class_names])
    recs = np.array([per_class[c]["recall"] for c in class_names])
    overall = {
        "acc": float((pred == eye).mean()) * 100,
        "precision_macro": float(pres.mean()),
        "recall_macro": float(recs.mean()),
        "f1_macro": float(f1s.mean()),
        "f1_weighted": float((f1s * counts).sum() / counts.sum()),
    }
    return overall, per_class, None


def save_epoch_metrics_csv(csv_path, epoch, split, class_names, overall, per_class):
    """逐 epoch 保存：每类 acc/precision/recall/f1 + 总体指标。

    列约定（scope 列区分整体/每类）：
      overall 行：acc + macro precision/recall/f1 + weighted-f1
      每类行   ：该类 acc/precision/recall/f1（f1_weighted 列留空）
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header = ["epoch", "split", "scope", "class", "acc", "precision", "recall", "f1",
              "f1_weighted", "n_samples"]
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        row = {"epoch": epoch, "split": split, "scope": "overall", "class": "-",
               "acc": f"{overall['acc']:.2f}",
               "precision": f"{overall['precision_macro']:.2f}",
               "recall": f"{overall['recall_macro']:.2f}",
               "f1": f"{overall['f1_macro']:.2f}",
               "f1_weighted": f"{overall['f1_weighted']:.2f}",
               "n_samples": ""}
        w.writerow(row)
        for c, m in per_class.items():
            row = {"epoch": epoch, "split": split, "scope": "class", "class": c,
                   "acc": f"{m['accuracy']:.2f}",
                   "precision": f"{m['precision']:.2f}",
                   "recall": f"{m['recall']:.2f}",
                   "f1": f"{m['f1']:.2f}",
                   "f1_weighted": "",
                   "n_samples": m.get("n", "")}
            w.writerow(row)


def save_confusion_matrix_csv(cm_path, cm, class_names, split, epoch):
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names)
        for i, c in enumerate(class_names):
            w.writerow([c] + [f"{v:.2f}" for v in cm[i]])
    cmn = cm.astype(np.float64)
    for i in range(cmn.shape[0]):
        s = cmn[i].sum()
        if s > 0:
            cmn[i] = cmn[i] / s * 100
    norm_path = cm_path.replace(".csv", "_normalized.csv")
    with open(norm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names)
        for i, c in enumerate(class_names):
            w.writerow([c] + [f"{v:.2f}%" for v in cmn[i]])


def boundary_error(gt_bouts, pred_bouts):
    """onset/offset 90th 分位误差（帧），bouts: [(start, end, label)]。"""
    if not gt_bouts or not pred_bouts:
        return {"onset_90th_pct": float("nan"), "offset_90th_pct": float("nan")}
    gt_onset = np.array([b[0] for b in gt_bouts])
    pr_onset = np.array([b[0] for b in pred_bouts])
    gt_off = np.array([b[1] for b in gt_bouts])
    pr_off = np.array([b[1] for b in pred_bouts])

    def nearest(a, b):
        err = np.abs(a[:, None] - b[None, :]).min(axis=1)
        return err

    return {"onset_90th_pct": float(np.percentile(nearest(gt_onset, pr_onset), 90)),
            "offset_90th_pct": float(np.percentile(nearest(gt_off, pr_off), 90))}
