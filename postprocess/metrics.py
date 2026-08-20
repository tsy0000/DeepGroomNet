"""Multilabel classification metrics (threshold search + per-class report).

Core:
  - evaluate_thresholds: per-class argmax-F1 threshold search on a grid,
    with remove_low_thresholds and valid-background handling.
  - postprocess / binary_confusion_matrix / compute_metrics_by_threshold.
  - epoch metrics: accuracy_overall (pooled binary acc), macro-F1 (all classes),
    macro-AUROC, macro-mAP; background handled via an explicit bg_idx (supports
    OFT where background=0 and openfield where background is the last class).

Reported metrics (eLife Evaluation convention):
  - accuracy_overall: (tn+tp)/total over the summed per-class confusion matrices
  - f1_class_mean: macro F1 over all classes (incl. background)
  - f1_class_mean_nobg: macro F1 excluding background (HPO/selection target)
  - auroc_class_mean: 逐类 AUROC 均值 (阈值网格梯形积分)
  - mAP_class_mean: 逐类 mAP 均值
"""
import numpy as np
from sklearn.metrics import auc

EPS = 1e-7


def index_to_onehot(index: np.ndarray, n_classes: int) -> np.ndarray:
    onehot = np.zeros((index.shape[0], n_classes), dtype=np.uint16)
    onehot[np.arange(onehot.shape[0]), index] = 1
    return onehot


def hardmax(probabilities: np.ndarray) -> np.ndarray:
    array = np.zeros(probabilities.shape, dtype=np.uint16)
    array[np.arange(array.shape[0]), np.argmax(probabilities, axis=1)] = 1
    return array


def onehot_to_index(onehot: np.ndarray) -> np.ndarray:
    return np.argmax(onehot, axis=1)


def remove_low_thresholds(thresholds: np.ndarray, minimum: float = 0.01,
                          f1s: np.ndarray = None, minimum_f1: float = 0.05) -> np.ndarray:
    """官方 postprocessing.remove_low_thresholds: 阈值<minimum 置 minimum; 对应 F1<minimum_f1 置 0.5。"""
    thresholds = thresholds.astype(float).copy()
    thresholds[thresholds < minimum] = minimum
    if f1s is not None:
        thresholds[np.asarray(f1s) < minimum_f1] = 0.5
    return thresholds


def postprocess(predictions: np.ndarray, thresholds: np.ndarray, bg_idx: int) -> np.ndarray:
    """概率 -> 逐类二值预测, background(bg_idx) = NOT(any other)。官方 valid_bg=True 语义。"""
    N, n_classes = predictions.shape
    assert len(thresholds) == n_classes
    estimates = np.zeros((N, n_classes), dtype=np.int64)
    for i in range(n_classes):
        estimates[:, i] = (predictions[:, i] > thresholds[i]).astype(int)
    others = [i for i in range(n_classes) if i != bg_idx]
    estimates[:, bg_idx] = np.logical_not(np.any(estimates[:, others], axis=1)).astype(int)
    return estimates


def binary_confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """逐类二元混淆矩阵, 返回 (2,2,K): [actual, predicted]。官方 binary_confusion_matrix。"""
    K = predictions.shape[1]
    cms = np.zeros((2, 2, K), dtype=int)
    neg_lab = np.logical_not(labels)
    neg_pred = np.logical_not(predictions)
    cms[0, 0] = (neg_lab * neg_pred).sum(axis=0)
    cms[0, 1] = (neg_lab * predictions).sum(axis=0)
    cms[1, 0] = (labels * neg_pred).sum(axis=0)
    cms[1, 1] = (labels * predictions).sum(axis=0)
    return cms


def get_denominator(expression):
    if isinstance(expression, (int, np.integer, float, np.floating)):
        return max(EPS, expression)
    expression = expression.astype(np.float32)
    expression[expression < EPS] = EPS
    return expression


def compute_precision_recall(cm):
    tp = cm[1, 1]; fp = cm[0, 1]; fn = cm[1, 0]
    precision = tp / get_denominator(tp + fp)
    recall = tp / get_denominator(tp + fn)
    return precision, recall


def compute_f1(precision, recall, beta=1.0):
    num = (1 + beta ** 2) * (precision * recall)
    denom = get_denominator((beta ** 2) * precision + recall)
    return num / denom


def compute_informedness(cm):
    tn = cm[0, 0]; tp = cm[1, 1]; fp = cm[0, 1]; fn = cm[1, 0]
    sensitivity = tp / get_denominator(tp + fn)
    specificity = tn / get_denominator(fp + tn)
    return sensitivity + specificity - 1


def compute_tpr_fpr(cm):
    cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + EPS)
    fp = cm_normalized[0, 1]
    tp = cm_normalized[1, 1]
    return tp, fp


def auc_on_array(x, y):
    """官方 auc_on_array 的鲁棒版: 官方用 sklearn.auc (要求 x 单调), 阈值网格下 FPR 可能重复/非单调而报错。
    改为按 x 排序后梯形积分 (np.trapz), 语义仍是 ROC/AUC 曲线下面积, 数值与官方单调情形一致。"""
    if x.ndim < 2 or y.ndim < 2:
        return np.nan
    K = x.shape[1]
    area = np.zeros((K,), dtype=np.float32)
    for i in range(K):
        xi, yi = np.asarray(x[:, i], float), np.asarray(y[:, i], float)
        order = np.argsort(xi)
        area[i] = np.trapz(yi[order], xi[order])
    return area


def compute_metrics_by_threshold(probabilities, labels, thresholds, bg_idx, cm=None):
    if cm is None:
        # grid: 每个阈值下生成逐类预测 (含 valid_bg), 返回 (2,2,M,K)
        cm = _grid_confusion(probabilities, labels, thresholds, bg_idx)
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum(axis=0).sum(axis=0)
    p, r = compute_precision_recall(cm)
    tp, fp = compute_tpr_fpr(cm)
    info = compute_informedness(cm)
    f1 = compute_f1(p, r)
    fbeta_2 = compute_f1(p, r, beta=2.0)
    auroc = auc_on_array(fp, tp)
    mAP = auc_on_array(r, p)
    return {
        "accuracy": acc, "f1": f1, "precision": p, "recall": r,
        "fbeta_2": fbeta_2, "informedness": info, "tpr": tp, "fpr": fp,
        "auroc": auroc, "mAP": mAP, "confusion": cm,
    }


def _grid_confusion(probabilities, labels, thresholds, bg_idx):
    """官方 binary_confusion_matrix_multiple_thresholds + valid_bg: 返回 (2,2,K,M)。"""
    K = probabilities.shape[1]
    M = len(thresholds)
    pred = np.greater(probabilities[:, :, None], thresholds[None, None, :])  # (N,K,M)
    lab = labels[:, :, None].repeat(M, axis=2)                               # (N,K,M)
    others = [i for i in range(K) if i != bg_idx]
    bg = np.logical_not(np.any(pred[:, others, :], axis=1))                  # (N,M)
    pred[:, bg_idx, :] = bg
    cms = np.zeros((2, 2, K, M), dtype=int)
    neg_lab = np.logical_not(lab); neg_pred = np.logical_not(pred)
    cms[0, 0] = (neg_lab * neg_pred).sum(axis=0)
    cms[0, 1] = (neg_lab * pred).sum(axis=0)
    cms[1, 0] = (lab * neg_pred).sum(axis=0)
    cms[1, 1] = (lab * pred).sum(axis=0)
    # 官方 binary_confusion_matrix_multiple_thresholds: transpose to (2,2,M,K)
    return cms.transpose(0, 1, 3, 2)  # (2,2,M,K)


def evaluate_thresholds(probabilities, labels, bg_idx, thresholds=None):
    """官方 evaluate_thresholds 忠实迁移 (单进程)。

    返回 (metrics_by_threshold, epoch_metrics)。
    阈值选取: 每类在网格上 argmax 二元 F1, remove_low_thresholds; ALWAYS valid_bg。
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if probabilities.ndim == 1:
        raise ValueError("To calc threshold, predictions must be probabilities, not classes")
    K = probabilities.shape[1]
    if labels.ndim == 1:
        labels = index_to_onehot(labels, K)
    labels = labels.astype(float)

    if thresholds is None:
        thresholds = np.linspace(1e-4, 1, 200)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    metrics_by_threshold = compute_metrics_by_threshold(probabilities, labels, thresholds, bg_idx)

    # 每类 argmax F1 -> 最优阈值
    optimum_indices = np.argmax(metrics_by_threshold["f1"], axis=0)          # (K,)
    optimum_thresholds = thresholds[optimum_indices]
    optimum_f1s = metrics_by_threshold["f1"][optimum_indices, np.arange(K)]
    optimum_thresholds = remove_low_thresholds(optimum_thresholds, f1s=optimum_f1s)

    # informedness 最优 (备用)
    opt_info_idx = np.argmax(metrics_by_threshold["informedness"], axis=0)
    opt_thresholds_info = thresholds[opt_info_idx]
    opt_info = metrics_by_threshold["informedness"][opt_info_idx, np.arange(K)]
    opt_thresholds_info = remove_low_thresholds(opt_thresholds_info, f1s=opt_info)

    metrics_by_threshold["optimum"] = optimum_thresholds
    metrics_by_threshold["optimum_info"] = opt_thresholds_info

    # 最终预测 (valid_bg)
    predictions = (probabilities > optimum_thresholds).astype(int)
    others = [i for i in range(K) if i != bg_idx]
    predictions[:, bg_idx] = np.logical_not(np.any(predictions[:, others], axis=1)).astype(int)

    # 最终逐类指标 (二元混淆矩阵, 单阈值)
    cm_by_class = binary_confusion_matrix(predictions, labels)
    metrics_by_class = compute_metrics_by_threshold(None, None, None, bg_idx, cm=cm_by_class)
    # overall: 所有类混淆矩阵求和 -> 单一二元问题
    overall_cm = cm_by_class.sum(axis=2)
    overall_metrics = compute_metrics_by_threshold(None, None, None, bg_idx, cm=overall_cm)

    nobg = [i for i in range(K) if i != bg_idx]
    epoch_metrics = {
        "accuracy_overall": overall_metrics["accuracy"],
        "accuracy_by_class": metrics_by_class["accuracy"],
        "f1_overall": overall_metrics["f1"],
        "f1_class_mean": float(metrics_by_class["f1"].mean()),
        "f1_class_mean_nobg": float(metrics_by_class["f1"][nobg].mean()),
        "f1_by_class": metrics_by_class["f1"],
        "precision_by_class": metrics_by_class["precision"],
        "recall_by_class": metrics_by_class["recall"],
        "auroc_by_class": metrics_by_threshold["auroc"],
        "auroc_class_mean": float(np.nanmean(metrics_by_threshold["auroc"])),
        "mAP_by_class": metrics_by_threshold["mAP"],
        "mAP_class_mean": float(np.nanmean(metrics_by_threshold["mAP"])),
        "binary_confusion": cm_by_class.transpose(2, 0, 1),
    }
    return metrics_by_threshold, epoch_metrics


def compute_metrics(P, Y, class_names, bg_idx, thresholds=None, fixed_thresholds=None):
    """便捷封装: 返回 (epoch_metrics, per_class_table, used_thresholds)。

    per_class_table: {class: {accuracy, precision, recall, f1, auroc}} (百分数 for acc/pre/rec/f1)。
    thresholds: 阈值搜索网格 (None=默认200点), 仅在 fixed_thresholds 为 None 时生效。
    fixed_thresholds: 逐类固定阈值 (K,) numpy array。传入时跳过搜索, 直接用该阈值预测
                      (用于验证/测试集, 避免在验证集自调阈值导致过拟合)。
    """
    P = np.asarray(P, dtype=np.float64)
    K = P.shape[1]
    if fixed_thresholds is not None:
        # ---- 固定阈值模式: 不搜索, 直接预测 ----
        thr = np.asarray(fixed_thresholds, dtype=float).reshape(-1)
        assert len(thr) == K, f"fixed_thresholds length {len(thr)} != {K} classes"
        pred = postprocess(P, thr, bg_idx)
        lab = index_to_onehot(Y, K) if np.asarray(Y).ndim == 1 else np.asarray(Y, dtype=float)
        cm = binary_confusion_matrix(pred, lab)
        m = compute_metrics_by_threshold(None, None, None, bg_idx, cm=cm)
        overall_cm = cm.sum(axis=2)
        om = compute_metrics_by_threshold(None, None, None, bg_idx, cm=overall_cm)
        nobg = [i for i in range(K) if i != bg_idx]
        # AUROC/mAP 仍用全网格计算 (阈值无关, 不受 fixed_thresholds 影响)
        full_mt, _ = evaluate_thresholds(P, Y, bg_idx=bg_idx, thresholds=thresholds)
        em = {
            "accuracy_overall": om["accuracy"],
            "accuracy_by_class": m["accuracy"],
            "f1_overall": om["f1"],
            "f1_class_mean": float(m["f1"].mean()),
            "f1_class_mean_nobg": float(m["f1"][nobg].mean()) if nobg else float(m["f1"].mean()),
            "f1_by_class": m["f1"],
            "precision_by_class": m["precision"],
            "recall_by_class": m["recall"],
            "auroc_by_class": full_mt["auroc"],
            "auroc_class_mean": float(np.nanmean(full_mt["auroc"])),
            "mAP_by_class": full_mt["mAP"],
            "mAP_class_mean": float(np.nanmean(full_mt["mAP"])),
            "binary_confusion": cm.transpose(2, 0, 1),
        }
        per_class = {}
        for i, c in enumerate(class_names):
            per_class[c] = {
                "accuracy": float(em["accuracy_by_class"][i]) * 100,
                "precision": float(em["precision_by_class"][i]) * 100,
                "recall": float(em["recall_by_class"][i]) * 100,
                "f1": float(em["f1_by_class"][i]) * 100,
                "auroc": float(em["auroc_by_class"][i]) if np.isfinite(em["auroc_by_class"][i]) else 0.0,
            }
        return em, per_class, thr

    # ---- 默认: 在给定数据上按官方 argmax-F1 调优 ----
    mt, em = evaluate_thresholds(P, Y, bg_idx=bg_idx, thresholds=thresholds)
    per_class = {}
    for i, c in enumerate(class_names):
        per_class[c] = {
            "accuracy": float(em["accuracy_by_class"][i]) * 100,
            "precision": float(em["precision_by_class"][i]) * 100,
            "recall": float(em["recall_by_class"][i]) * 100,
            "f1": float(em["f1_by_class"][i]) * 100,
            "auroc": float(em["auroc_by_class"][i]) if np.isfinite(em["auroc_by_class"][i]) else 0.0,
        }
    return em, per_class, mt["optimum"]