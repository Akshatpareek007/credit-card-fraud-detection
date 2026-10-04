"""
Tests for Metrics Calculation, Cost Minimization, Bootstrap CIs, and Reproducibility.
"""

import numpy as np
import pytest
from src.evaluate import (
    calculate_metrics,
    calculate_expected_cost,
    find_optimal_cost_threshold,
    bootstrap_confidence_intervals
)


def test_calculate_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confusion_matrix"]["tn"] == 2


def test_cost_threshold_minimization():
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.4, 0.6, 0.9])

    best_thresh, min_cost = find_optimal_cost_threshold(y_true, y_prob, cost_FN=500.0, cost_FP=10.0)
    assert 0.0 < best_thresh < 1.0
    assert min_cost >= 0.0


def test_bootstrap_ci_reproducibility():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=100)
    y_prob = rng.uniform(0, 1, size=100)

    ci1 = bootstrap_confidence_intervals(y_true, y_prob, threshold=0.5, n_bootstraps=100, seed=42)
    ci2 = bootstrap_confidence_intervals(y_true, y_prob, threshold=0.5, n_bootstraps=100, seed=42)

    assert ci1["roc_auc"]["mean"] == pytest.approx(ci2["roc_auc"]["mean"], abs=1e-5)
    assert ci1["f1"]["ci_lower"] == pytest.approx(ci2["f1"]["ci_lower"], abs=1e-5)
