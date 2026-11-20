import numpy as np
EPS = 1e-07

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

def remove_low_thresholds(thresholds: np.ndarray, minimum: float=0.01, f1s: np.ndarray=None, minimum_f1: float=0.05) -> np.ndarray:
    thresholds = thresholds.astype(float).copy()
    thresholds[thresholds < minimum] = minimum
    if f1s is not None:
        thresholds[np.asarray(f1s) < minimum_f1] = 0.5
    return thresholds

def postprocess(predictions: np.ndarray, thresholds: np.ndarray, bg_idx: int) -> np.ndarray:
    (N, n_classes) = predictions.shape
    assert len(thresholds) == n_classes
    estimates = np.zeros((N, n_classes), dtype=np.int64)
    for i in range(n_classes):
        estimates[:, i] = (predictions[:, i] > thresholds[i]).astype(int)
    others = [i for i in range(n_classes) if i != bg_idx]
    estimates[:, bg_idx] = np.logical_not(np.any(estimates[:, others], axis=1)).astype(int)
    return estimates

def binary_confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
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
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    precision = tp / get_denominator(tp + fp)
    recall = tp / get_denominator(tp + fn)
    return (precision, recall)

def compute_f1(precision, recall, beta=1.0):
    num = (1 + beta ** 2) * (precision * recall)
    denom = get_denominator(beta ** 2 * precision + recall)
    return num / denom

def compute_informedness(cm):
    tn = cm[0, 0]
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    sensitivity = tp / get_denominator(tp + fn)
    specificity = tn / get_denominator(fp + tn)
    return sensitivity + specificity - 1

def compute_tpr_fpr(cm):
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + EPS)
    fp = cm_normalized[0, 1]
    tp = cm_normalized[1, 1]
    return (tp, fp)

def auc_on_array(x, y):
    if x.ndim < 2 or y.ndim < 2:
        return np.nan
    K = x.shape[1]
    area = np.zeros((K,), dtype=np.float32)
    for i in range(K):
        (xi, yi) = (np.asarray(x[:, i], float), np.asarray(y[:, i], float))
        order = np.argsort(xi)
        area[i] = np.trapz(yi[order], xi[order])
    return area

def compute_metrics_by_threshold(probabilities, labels, thresholds, bg_idx, cm=None):
    if cm is None:
        cm = _grid_confusion(probabilities, labels, thresholds, bg_idx)
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum(axis=0).sum(axis=0)
    (p, r) = compute_precision_recall(cm)
    (tp, fp) = compute_tpr_fpr(cm)
    info = compute_informedness(cm)
    f1 = compute_f1(p, r)
    fbeta_2 = compute_f1(p, r, beta=2.0)
    auroc = auc_on_array(fp, tp)
    mAP = auc_on_array(r, p)
    return {'accuracy': acc, 'f1': f1, 'precision': p, 'recall': r, 'fbeta_2': fbeta_2, 'informedness': info, 'tpr': tp, 'fpr': fp, 'auroc': auroc, 'mAP': mAP, 'confusion': cm}

def _grid_confusion(probabilities, labels, thresholds, bg_idx):
    K = probabilities.shape[1]
    M = len(thresholds)
    pred = np.greater(probabilities[:, :, None], thresholds[None, None, :])
    lab = labels[:, :, None].repeat(M, axis=2)
    others = [i for i in range(K) if i != bg_idx]
    bg = np.logical_not(np.any(pred[:, others, :], axis=1))
    pred[:, bg_idx, :] = bg
    cms = np.zeros((2, 2, K, M), dtype=int)
    neg_lab = np.logical_not(lab)
    neg_pred = np.logical_not(pred)
    cms[0, 0] = (neg_lab * neg_pred).sum(axis=0)
    cms[0, 1] = (neg_lab * pred).sum(axis=0)
    cms[1, 0] = (lab * neg_pred).sum(axis=0)
    cms[1, 1] = (lab * pred).sum(axis=0)
    return cms.transpose(0, 1, 3, 2)

def evaluate_thresholds(probabilities, labels, bg_idx, thresholds=None):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if probabilities.ndim == 1:
        raise ValueError('To calc threshold, predictions must be probabilities, not classes')
    K = probabilities.shape[1]
    if labels.ndim == 1:
        labels = index_to_onehot(labels, K)
    labels = labels.astype(float)
    if thresholds is None:
        thresholds = np.linspace(0.0001, 1, 200)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    metrics_by_threshold = compute_metrics_by_threshold(probabilities, labels, thresholds, bg_idx)
    optimum_indices = np.argmax(metrics_by_threshold['f1'], axis=0)
    optimum_thresholds = thresholds[optimum_indices]
    optimum_f1s = metrics_by_threshold['f1'][optimum_indices, np.arange(K)]
    optimum_thresholds = remove_low_thresholds(optimum_thresholds, f1s=optimum_f1s)
    opt_info_idx = np.argmax(metrics_by_threshold['informedness'], axis=0)
    opt_thresholds_info = thresholds[opt_info_idx]
    opt_info = metrics_by_threshold['informedness'][opt_info_idx, np.arange(K)]
    opt_thresholds_info = remove_low_thresholds(opt_thresholds_info, f1s=opt_info)
    metrics_by_threshold['optimum'] = optimum_thresholds
    metrics_by_threshold['optimum_info'] = opt_thresholds_info
    predictions = (probabilities > optimum_thresholds).astype(int)
    others = [i for i in range(K) if i != bg_idx]
    predictions[:, bg_idx] = np.logical_not(np.any(predictions[:, others], axis=1)).astype(int)
    cm_by_class = binary_confusion_matrix(predictions, labels)
    metrics_by_class = compute_metrics_by_threshold(None, None, None, bg_idx, cm=cm_by_class)
    overall_cm = cm_by_class.sum(axis=2)
    overall_metrics = compute_metrics_by_threshold(None, None, None, bg_idx, cm=overall_cm)
    nobg = [i for i in range(K) if i != bg_idx]
    epoch_metrics = {'accuracy_overall': overall_metrics['accuracy'], 'accuracy_by_class': metrics_by_class['accuracy'], 'f1_overall': overall_metrics['f1'], 'f1_class_mean': float(metrics_by_class['f1'].mean()), 'f1_class_mean_nobg': float(metrics_by_class['f1'][nobg].mean()), 'f1_by_class': metrics_by_class['f1'], 'precision_by_class': metrics_by_class['precision'], 'recall_by_class': metrics_by_class['recall'], 'auroc_by_class': metrics_by_threshold['auroc'], 'auroc_class_mean': float(np.nanmean(metrics_by_threshold['auroc'])), 'mAP_by_class': metrics_by_threshold['mAP'], 'mAP_class_mean': float(np.nanmean(metrics_by_threshold['mAP'])), 'binary_confusion': cm_by_class.transpose(2, 0, 1)}
    return (metrics_by_threshold, epoch_metrics)

def compute_metrics(P, Y, class_names, bg_idx, thresholds=None, fixed_thresholds=None):
    P = np.asarray(P, dtype=np.float64)
    K = P.shape[1]
    if fixed_thresholds is not None:
        thr = np.asarray(fixed_thresholds, dtype=float).reshape(-1)
        assert len(thr) == K, f'fixed_thresholds length {len(thr)} != {K} classes'
        pred = postprocess(P, thr, bg_idx)
        lab = index_to_onehot(Y, K) if np.asarray(Y).ndim == 1 else np.asarray(Y, dtype=float)
        cm = binary_confusion_matrix(pred, lab)
        m = compute_metrics_by_threshold(None, None, None, bg_idx, cm=cm)
        overall_cm = cm.sum(axis=2)
        om = compute_metrics_by_threshold(None, None, None, bg_idx, cm=overall_cm)
        nobg = [i for i in range(K) if i != bg_idx]
        (full_mt, _) = evaluate_thresholds(P, Y, bg_idx=bg_idx, thresholds=thresholds)
        em = {'accuracy_overall': om['accuracy'], 'accuracy_by_class': m['accuracy'], 'f1_overall': om['f1'], 'f1_class_mean': float(m['f1'].mean()), 'f1_class_mean_nobg': float(m['f1'][nobg].mean()) if nobg else float(m['f1'].mean()), 'f1_by_class': m['f1'], 'precision_by_class': m['precision'], 'recall_by_class': m['recall'], 'auroc_by_class': full_mt['auroc'], 'auroc_class_mean': float(np.nanmean(full_mt['auroc'])), 'mAP_by_class': full_mt['mAP'], 'mAP_class_mean': float(np.nanmean(full_mt['mAP'])), 'binary_confusion': cm.transpose(2, 0, 1)}
        per_class = {}
        for (i, c) in enumerate(class_names):
            per_class[c] = {'accuracy': float(em['accuracy_by_class'][i]) * 100, 'precision': float(em['precision_by_class'][i]) * 100, 'recall': float(em['recall_by_class'][i]) * 100, 'f1': float(em['f1_by_class'][i]) * 100, 'auroc': float(em['auroc_by_class'][i]) if np.isfinite(em['auroc_by_class'][i]) else 0.0}
        return (em, per_class, thr)
    (mt, em) = evaluate_thresholds(P, Y, bg_idx=bg_idx, thresholds=thresholds)
    per_class = {}
    for (i, c) in enumerate(class_names):
        per_class[c] = {'accuracy': float(em['accuracy_by_class'][i]) * 100, 'precision': float(em['precision_by_class'][i]) * 100, 'recall': float(em['recall_by_class'][i]) * 100, 'f1': float(em['f1_by_class'][i]) * 100, 'auroc': float(em['auroc_by_class'][i]) if np.isfinite(em['auroc_by_class'][i]) else 0.0}
    return (em, per_class, mt['optimum'])
