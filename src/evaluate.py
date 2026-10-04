"""
Metrics, Cost Function Threshold Tuning, Platt Calibration, and Bootstrap Confidence Intervals Module.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc
)
from sklearn.calibration import calibration_curve


class PlattCalibrator:
    """Platt Scaling probability calibrator fitted on validation set logits/probabilities."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=self.random_state)

    def fit(self, val_raw_probs: np.ndarray, y_val: np.ndarray):
        eps = 1e-7
        probs_clamped = np.clip(val_raw_probs, eps, 1.0 - eps)
        logits = np.log(probs_clamped / (1.0 - probs_clamped)).reshape(-1, 1)
        self.calibrator.fit(logits, y_val)
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        eps = 1e-7
        probs_clamped = np.clip(raw_probs, eps, 1.0 - eps)
        logits = np.log(probs_clamped / (1.0 - probs_clamped)).reshape(-1, 1)
        return self.calibrator.predict_proba(logits)[:, 1]



def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Precision-Recall Area Under Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Calculate confusion matrix and classification metrics for a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = compute_pr_auc(y_true, y_prob)

    return {
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


def calculate_expected_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_FN: float = 500.0,
    cost_FP: float = 10.0
) -> float:
    """Compute total financial cost based on False Negatives and False Positives."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_cost = (cost_FN * fn) + (cost_FP * fp)
    return float(total_cost)


def find_optimal_cost_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_FN: float = 500.0,
    cost_FP: float = 10.0,
    num_steps: int = 200
) -> Tuple[float, float]:
    """
    Find optimal decision threshold on VALIDATION set minimizing total expected cost.
    Returns (optimal_threshold, min_cost).
    """
    thresholds = np.linspace(0.01, 0.99, num_steps)
    costs = []

    for t in thresholds:
        c = calculate_expected_cost(y_true, y_prob, threshold=t, cost_FN=cost_FN, cost_FP=cost_FP)
        costs.append(c)

    best_idx = int(np.argmin(costs))
    best_threshold = float(thresholds[best_idx])
    min_cost = float(costs[best_idx])

    return best_threshold, min_cost


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Calculate 95% Confidence Intervals via non-parametric bootstrapping (1000 resamples).
    Returns lower (2.5%) and upper (97.5%) bounds for each metric.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    boot_prec = []
    boot_rec = []
    boot_f1 = []
    boot_roc_auc = []
    boot_pr_auc = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, size=n)
        y_true_b = y_true[indices]
        y_prob_b = y_prob[indices]

        if len(np.unique(y_true_b)) < 2:
            continue

        metrics = calculate_metrics(y_true_b, y_prob_b, threshold=threshold)
        boot_prec.append(metrics["precision"])
        boot_rec.append(metrics["recall"])
        boot_f1.append(metrics["f1"])
        boot_roc_auc.append(metrics["roc_auc"])
        boot_pr_auc.append(metrics["pr_auc"])

    def get_ci(arr: list) -> Dict[str, float]:
        if not arr:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5))
        }

    return {
        "precision": get_ci(boot_prec),
        "recall": get_ci(boot_rec),
        "f1": get_ci(boot_f1),
        "roc_auc": get_ci(boot_roc_auc),
        "pr_auc": get_ci(boot_pr_auc)
    }


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, list]:
    """Compute calibration curve reliability points."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return {
        "prob_true": [float(x) for x in prob_true],
        "prob_pred": [float(x) for x in prob_pred]
    }
