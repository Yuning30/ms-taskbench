"""Unit tests for the 3-way selective-classification helpers."""

from __future__ import annotations

import numpy as np

from taskbench.verifier.selective import (
    apply_three_way,
    find_lambda_delta,
    selective_metrics,
)


def test_apply_three_way_basic():
    probs = np.array([0.05, 0.30, 0.50, 0.70, 0.95])
    out = apply_three_way(probs, lam=0.5, delta=0.2)
    # > 0.7 -> success(1), < 0.3 -> failure(0), else -1 uncertain.
    # 0.05 < 0.3 → 0;  0.30 NOT < 0.3 (strict) → -1; 0.50 → -1; 0.70 NOT > 0.7 → -1; 0.95 → 1.
    assert out.tolist() == [0, -1, -1, -1, 1]


def test_selective_metrics_perfect_classifier_with_band():
    probs = np.concatenate([np.full(100, 0.05), np.full(100, 0.95)])
    labels = np.concatenate([np.zeros(100), np.ones(100)]).astype(np.int32)
    m = selective_metrics(probs, labels, lam=0.5, delta=0.2)
    assert m["coverage"] == 1.0  # all predictions cleared the band
    assert m["selective_accuracy"] == 1.0
    assert m["uncertain"] == 0


def test_selective_metrics_uncertain_band_eats_some():
    """All probs at 0.5 -> all uncertain regardless of label."""
    probs = np.full(200, 0.5)
    labels = (np.arange(200) % 2).astype(np.int32)
    m = selective_metrics(probs, labels, lam=0.5, delta=0.1)
    assert m["coverage"] == 0.0
    assert m["uncertain"] == 200
    assert m["tp"] == m["fp"] == m["fn"] == m["tn"] == 0


def test_find_lambda_delta_recovers_clean_boundary():
    """With a clear bimodal distribution and a feasible coverage floor, the
    optimizer should choose a (lambda, delta) that covers almost everything
    with near-perfect selective accuracy."""
    rng = np.random.default_rng(0)
    n = 4000
    labels = (rng.random(n) < 0.5).astype(np.int32)
    probs = np.where(labels == 1,
                     rng.normal(0.85, 0.05, n),
                     rng.normal(0.15, 0.05, n)).clip(0.0, 1.0)
    fit = find_lambda_delta(probs, labels, min_coverage=0.80,
                            n_lambda=40, n_delta=20)
    assert fit["coverage"] >= 0.80
    assert fit["selective_accuracy"] > 0.97


def test_find_lambda_delta_falls_back_when_floor_infeasible():
    """A 100% uncertain band can't satisfy coverage>=0.99. Optimizer should
    return *something* (best by coverage), not crash."""
    probs = np.full(50, 0.5)
    labels = np.zeros(50, dtype=np.int32)
    # No (lambda, delta) can hit 99% coverage when every prob is 0.5.
    fit = find_lambda_delta(probs, labels, min_coverage=0.99,
                            n_lambda=20, n_delta=10)
    assert "lambda" in fit and "delta" in fit
