import numpy as np

from deepgroomnet.evaluation.stats import (bh_fdr, bootstrap_ci, cliff_delta,
                                           exact_permutation_p, paired_test)


def test_exact_permutation_six_vs_six():
    a = np.arange(6, dtype=float)
    b = np.arange(6, 12, dtype=float)
    p = exact_permutation_p(a, b)
    assert abs(p - 2 / 924) < 1e-12


def test_exact_permutation_identical_groups():
    a = np.arange(6, dtype=float)
    assert exact_permutation_p(a, a.copy()) == 1.0


def test_bh_fdr_monotone_and_bounded():
    p = np.array([0.001, 0.02, 0.04, 0.2])
    q = bh_fdr(p)
    assert np.all(q >= p)
    assert np.all(q <= 1.0)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)


def test_bootstrap_and_cliff():
    values = np.arange(20, dtype=float)
    low, high = bootstrap_ci(values, n_boot=500)
    assert low < values.mean() < high
    assert cliff_delta(values, values + 100) == -1.0


def test_paired_test_selects_normality():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 30)
    b = a + rng.normal(0, 0.1, 30)
    assert "test" in paired_test(a, b)
