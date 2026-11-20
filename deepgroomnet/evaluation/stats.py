import itertools
import numpy as np
from scipy import stats


def exact_permutation_p(a, b, max_exact=12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    n_a = len(a)
    if len(pool) <= max_exact:
        count = total = 0
        for index in itertools.combinations(range(len(pool)), n_a):
            mask = np.zeros(len(pool), dtype=bool)
            mask[list(index)] = True
            total += 1
            if abs(pool[mask].mean() - pool[~mask].mean()) >= observed - 1e-12:
                count += 1
        return count / total
    rng = np.random.default_rng(0)
    count = 0
    for _ in range(100000):
        rng.shuffle(pool)
        if abs(pool[:n_a].mean() - pool[n_a:].mean()) >= observed - 1e-12:
            count += 1
    return (count + 1) / 100001


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    previous = 1.0
    for rank in range(n - 1, -1, -1):
        index = order[rank]
        previous = min(previous, p[index] * n / (rank + 1))
        q[index] = previous
    return q


def bootstrap_ci(values, statistic=np.mean, n_boot=4000, seed=0, alpha=0.05):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = [statistic(rng.choice(values, len(values), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(draws, 100 * alpha / 2)), float(np.percentile(draws, 100 * (1 - alpha / 2)))


def cliff_delta(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    greater = sum((x > y) for x in a for y in b)
    less = sum((x < y) for x in a for y in b)
    return (greater - less) / (len(a) * len(b))


def paired_test(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    difference = a - b
    if len(a) >= 3 and stats.shapiro(difference).pvalue > 0.05:
        stat, p = stats.ttest_rel(a, b)
        return {"test": "paired_t", "stat": float(stat), "p": float(p)}
    stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    return {"test": "wilcoxon", "stat": float(stat), "p": float(p)}


def bonferroni(p_values, n_tests=None):
    p = np.asarray(p_values, dtype=float)
    n = n_tests or len(p)
    return np.minimum(p * n, 1.0)
