import numpy as np
import pytest

from deepgroomnet.fusion.sgatf import (adaptive_weights, class_predictions,
                                       run_means, runs_of, sgatf)


def make_probs(n=120, n_classes=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.95, size=(n, n_classes))


def reference_fusion(probs, syllables, alpha_max=1.0, max_run=12):
    p = np.asarray(probs, dtype=float)
    q = p.copy()
    starts, ends = runs_of(syllables)
    for a, b in zip(starts, ends):
        length = max(int(b - a), 1)
        consensus = float(p[a:b].max(axis=1).mean())
        gate = np.clip((consensus - 0.35) / 0.50, 0, 1)
        alpha = alpha_max * min(1.0, max_run / length) * gate
        q[a:b] = (1 - alpha) * p[a:b] + alpha * p[a:b].mean(axis=0)
    return q


def test_runs_and_run_means():
    syllables = np.array([0, 0, 1, 1, 1, 2])
    starts, ends = runs_of(syllables)
    assert list(starts) == [0, 2, 5]
    assert list(ends) == [2, 5, 6]
    probs = np.arange(12, dtype=float).reshape(6, 2)
    means = run_means(probs, syllables)
    assert np.allclose(means[0:2], probs[0:2].mean(axis=0))


def test_alpha_bounds_and_shapes():
    syllables = np.repeat(np.arange(8), 15)
    probs = make_probs()
    alpha = adaptive_weights(probs, syllables)
    assert alpha.shape == (len(probs),)
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    labels, fused, alpha2 = sgatf(probs, syllables, [0.5] * 5)
    assert labels.shape == (len(probs),)
    assert fused.shape == probs.shape
    assert np.array_equal(alpha, alpha2)
    assert labels.min() >= 0 and labels.max() <= 4


def test_short_runs_saturate_long_runs_decay():
    probs = make_probs(240)
    syllables = np.r_[np.zeros(5, int), np.ones(235, int)]
    alpha = adaptive_weights(probs, syllables)
    assert alpha[:5].max() >= alpha[5:].max()
    assert np.isclose(alpha[5:].mean(), alpha[5:].mean())
    assert alpha[5:].max() <= 12.0 / 235 + 1e-12


def test_low_confidence_suppresses_fusion():
    probs = np.full((60, 5), 0.2)
    syllables = np.zeros(60, int)
    alpha = adaptive_weights(probs, syllables)
    assert np.all(alpha == 0.0)
    labels, fused, _ = sgatf(probs, syllables, [0.5] * 5)
    assert np.allclose(fused, probs)


def test_frame_count_preserved_and_deterministic():
    probs = make_probs(137)
    syllables = np.repeat(np.arange(14), 10)[:137]
    first = sgatf(probs, syllables, [0.5] * 5)
    second = sgatf(probs, syllables, [0.5] * 5)
    assert len(first[0]) == len(probs)
    assert np.array_equal(first[0], second[0])
    assert np.allclose(first[1], second[1])


def test_matches_reference_implementation():
    probs = make_probs(200, seed=3)
    syllables = np.repeat(np.arange(20), 10)
    _, fused, _ = sgatf(probs, syllables, [0.5] * 5)
    assert np.allclose(fused, reference_fusion(probs, syllables), atol=1e-12)


def test_background_rule():
    probs = np.array([[0.1, 0.1, 0.2, 0.1, 0.6],
                      [0.2, 0.2, 0.9, 0.8, 0.1]])
    labels = class_predictions(probs, np.array([[0, 0, 0, 0, 1], [0, 0, 1, 1, 0]]))
    assert labels[0] == 4
    assert labels[1] == 2


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        sgatf(np.zeros((10, 5)), np.zeros(9, int), [0.5] * 5)
