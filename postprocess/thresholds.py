"""
DeepEthogram 式阈值推断后处理（借鉴 Bohnslav et al., eLife 2021）。

针对"background 占比 56.7%"的根因解法：
  1. 逐类阈值调优（在验证集上网格搜索，不用 argmax）——argmax 对稀有类过于激进，
     阈值化允许"什么都不预测 → 背景补集"，直接提升 FG/BG 与背景的可信度。
  2. 背景补集：没有任何非背景类超过阈值时 = background（类 0），
     而非靠 5 类 softmax 硬分（background 先验 56.7% 会把 FG 挤进背景）。
  3. 预测流程 = threshold -> (多类同时超过阈值时取最高分) -> 空集取背景。
"""
import numpy as np
from sklearn.metrics import f1_score


def sigmoid_probs(logits: np.ndarray) -> np.ndarray:
    """logits [N,C] -> sigmoid 概率 [N,C]（数值稳定，torch 同款实现）。"""
    z = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -80.0, 80.0)))


def threshold_predict(probs: np.ndarray, thresholds: np.ndarray, bg_idx: int = 0) -> np.ndarray:
    """DeepEthogram 式决策：s_k >= t_k 的类进入候选集；空集 -> 背景；多类 -> 最高分。

    probs:     [N,C] sigmoid 概率（多类之和不必为 1，逐类二值化）
    thresholds: [C] 逐类阈值（背景类的阈值理论上不用，传 1.0 即可）
    返回:      [N] 预测类索引
    """
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


def tune_thresholds(val_probs: np.ndarray, val_labels: np.ndarray, bg_idx: int = 0,
                    candidates: int = 40, passes: int = 2, min_f1: float = 0.05,
                    seed: int = 0) -> np.ndarray:
    """验证集上逐类网格搜索阈值，最大化 f1_macro（坐标下降，2 轮收敛）。

    val_probs:  [N,C] sigmoid 概率；val_labels: [N] 标签（0=background）
    返回:       [C] 逐类阈值
    守卫: 与 DeepEthogram 一致——F1 < min_f1 的类复位阈值 0.5（防退化/切到全 1），
          阈值 < 0.01 一律提到 0.01（防全预测 1）。
    """
    rng = np.random.default_rng(seed)
    label_ids = np.unique(val_labels)
    thresholds = np.full(val_probs.shape[1], 0.5, dtype=np.float64)
    for _ in range(passes):
        for c in sorted(label_ids):
            if c == bg_idx:
                continue
            grid = np.linspace(0.05, 0.95, candidates)
            rng.shuffle(grid)
            best_t, best_f1 = 0.5, -1.0
            for t in grid:
                th = thresholds.copy()
                th[c] = t
                pred = threshold_predict(val_probs, th, bg_idx)
                f1 = f1_score(val_labels, pred, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_t, best_f1 = t, f1
            thresholds[c] = best_t
    thresholds = np.clip(thresholds, 0.01, None)
    per_class = f1_score(
        val_labels,
        threshold_predict(val_probs, thresholds, bg_idx),
        average=None, zero_division=0, labels=np.arange(val_probs.shape[1]))
    thresholds[per_class < min_f1] = 0.5
    return thresholds
