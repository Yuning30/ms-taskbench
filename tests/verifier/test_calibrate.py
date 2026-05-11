"""Unit tests for calibration helpers (no model/IO required)."""

from __future__ import annotations

import numpy as np

from taskbench.verifier.calibrate import (
    best_f1_threshold,
    expected_calibration_error,
    fit_temperature,
)


def test_fit_temperature_already_calibrated_returns_near_one():
    """If logits already produce well-calibrated probs, T should land near 1."""
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2.0, size=5000)
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(5000) < p).astype(np.float64)
    T = fit_temperature(z, y)
    assert 0.7 < T < 1.5, T


def test_fit_temperature_overconfident_returns_T_above_one():
    """Overconfident logits should get scaled up (T > 1)."""
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2.0, size=5000)
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(5000) < p).astype(np.float64)
    z_sharp = z * 4.0  # exaggerate confidence
    T = fit_temperature(z_sharp, y)
    assert T > 2.0, T


def test_ece_zero_for_perfect_calibration():
    """If predicted probs equal empirical frequencies, ECE should be ~0."""
    rng = np.random.default_rng(2)
    probs = rng.uniform(0, 1, size=20000)
    y = (rng.random(20000) < probs).astype(np.int32)
    ece = expected_calibration_error(probs, y, n_bins=10)
    assert ece < 0.02, ece


def test_best_f1_threshold_separable():
    """Perfectly separable score should yield F1=1.0."""
    probs = np.concatenate([np.full(100, 0.2), np.full(100, 0.8)])
    labels = np.concatenate([np.zeros(100), np.ones(100)]).astype(np.int32)
    out = best_f1_threshold(probs, labels)
    assert out["f1"] >= 0.99
    assert 0.21 <= out["threshold"] <= 0.79


def test_best_f1_threshold_no_positives_returns_nan():
    probs = np.linspace(0, 1, 50)
    labels = np.zeros(50, dtype=np.int32)
    out = best_f1_threshold(probs, labels)
    assert np.isnan(out["f1"])
