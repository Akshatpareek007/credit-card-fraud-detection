"""
Training, Tuning, Imbalance Strategy Comparison, Platt Calibration, Artifact Saving, and Pipeline Execution.
"""

import os
import json
import hashlib
import time
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from src.data import load_dataset, validate_dataset, deduplicate_dataset, temporal_split
from src.features import prepare_features, FEATURE_NAMES
from src.evaluate import (
    calculate_metrics,
    find_optimal_cost_threshold,
    bootstrap_confidence_intervals,
    compute_calibration_curve,
    calculate_expected_cost,
    PlattCalibrator
)

# Constants & Paths
RANDOM_SEED = 42
MODEL_VERSION = "2.0.0"
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "creditcard.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")




def apply_imbalance_strategy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply specified imbalance strategy to training set ONLY."""
    if strategy == "none" or strategy == "class_weight":
        return X_train, y_train
    elif strategy == "random_undersampling":
        rus = RandomUnderSampler(random_state=RANDOM_SEED)
        return rus.fit_resample(X_train, y_train)
    elif strategy == "smote":
        smote = SMOTE(random_state=RANDOM_SEED)
        return smote.fit_resample(X_train, y_train)
    else:
        raise ValueError(f"Unknown imbalance strategy: {strategy}")


def get_model_candidates() -> Dict[str, List[Dict[str, Any]]]:
    """Define candidate model hyperparameter search spaces."""
    pos_scale = float((198608 - 366) / 366)
    return {
        "Logistic Regression": [
            {"C": 1.0, "max_iter": 1000, "random_state": RANDOM_SEED},
        ],
        "Decision Tree": [
            {"max_depth": 6, "min_samples_split": 10, "random_state": RANDOM_SEED},
        ],
        "Random Forest": [
            {"n_estimators": 100, "max_depth": 10, "random_state": RANDOM_SEED, "n_jobs": -1},
        ],
        "XGBoost": [
            {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": pos_scale, "random_state": RANDOM_SEED, "n_jobs": -1},
        ],
        "LightGBM": [
            {"n_estimators": 150, "max_depth": 5, "num_leaves": 31, "learning_rate": 0.08, "scale_pos_weight": pos_scale, "random_state": RANDOM_SEED, "n_jobs": -1, "verbose": -1},
        ]
    }


def instantiate_model(model_name: str, params: Dict[str, Any], is_class_weight: bool):
    """Instantiate model with or without class weights."""
    p = params.copy()
    if is_class_weight:
        if model_name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
            p["class_weight"] = "balanced"

    if model_name == "Logistic Regression":
        return LogisticRegression(**p)
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(**p)
    elif model_name == "Random Forest":
        return RandomForestClassifier(**p)
    elif model_name == "XGBoost":
        return XGBClassifier(**p)
    elif model_name == "LightGBM":
        return LGBMClassifier(**p)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_training_pipeline(
    csv_path: str = DEFAULT_DATA_PATH,
    cost_FN: float = 500.0,
    cost_FP: float = 10.0
) -> Dict[str, Any]:
    """Execute complete end-to-end model training, tuning, calibration, and evaluation pipeline."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"Loading dataset from {csv_path}...")
    raw_df = load_dataset(csv_path)
    validate_dataset(raw_df)

    clean_df, dup_count = deduplicate_dataset(raw_df)
    print(f"Removed {dup_count} exact duplicate rows.")

    split_result = temporal_split(clean_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    train_df = split_result["train"]
    val_df = split_result["val"]
    test_df = split_result["test"]
    split_summary = split_result["summary"]

    print("Temporal Split Summary:")
    print(f"   Train: {split_summary['train']['count']} rows ({split_summary['train']['fraud_count']} frauds)")
    print(f"   Val:   {split_summary['val']['count']} rows ({split_summary['val']['fraud_count']} frauds)")
    print(f"   Test:  {split_summary['test']['count']} rows ({split_summary['test']['fraud_count']} frauds)")

    X_train, y_train, X_val, y_val, X_test, y_test, pipeline = prepare_features(train_df, val_df, test_df)

    strategies = ["none", "class_weight", "random_undersampling", "smote"]
    candidates = get_model_candidates()

    best_overall_score = -1.0
    best_config = None
    best_model_obj = None
    best_val_raw_probs = None

    comparison_results = []

    print("\nTuning models and evaluating imbalance strategies on VALIDATION set...")
    for model_name, param_list in candidates.items():
        for params in param_list:
            for strategy in strategies:
                is_cw = (strategy == "class_weight")
                X_tr, y_tr = apply_imbalance_strategy(X_train, y_train, strategy)
                model = instantiate_model(model_name, params, is_class_weight=is_cw)
                model.fit(X_tr, y_tr)

                val_probs = model.predict_proba(X_val)[:, 1]
                val_thresh, val_cost = find_optimal_cost_threshold(y_val, val_probs, cost_FN=cost_FN, cost_FP=cost_FP)
                val_metrics = calculate_metrics(y_val, val_probs, threshold=val_thresh)

                score = val_metrics["pr_auc"]
                print(f"   Evaluated {model_name} | Strategy: {strategy:<20} | Val PR-AUC: {score:.4f} | Val Cost: ${val_cost:,.2f}")

                res_entry = {
                    "model_name": model_name,
                    "strategy": strategy,
                    "params": {k: str(v) for k, v in params.items()},
                    "val_pr_auc": val_metrics["pr_auc"],
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_f1": val_metrics["f1"],
                    "val_cost": val_cost,
                    "val_threshold": val_thresh
                }
                comparison_results.append(res_entry)

                if score > best_overall_score:
                    best_overall_score = score
                    best_config = res_entry
                    best_model_obj = model
                    best_val_raw_probs = val_probs


    print(f"Best Model Selected: {best_config['model_name']} with Strategy '{best_config['strategy']}' (Val PR-AUC: {best_overall_score:.4f})")

    # Fit Platt Calibrator on Validation set probabilities
    calibrator = PlattCalibrator()
    calibrator.fit(best_val_raw_probs, y_val)
    best_val_calibrated_probs = calibrator.calibrate(best_val_raw_probs)

    # Re-tune decision threshold on CALIBRATED validation probabilities using expected cost minimization
    optimal_threshold, min_val_cost = find_optimal_cost_threshold(
        y_val, best_val_calibrated_probs, cost_FN=cost_FN, cost_FP=cost_FP
    )
    print(f"Optimal Threshold (Val Cost Minimization): {optimal_threshold:.4f} (Expected Val Cost: ${min_val_cost:,.2f})")

    val_calibration_curve = compute_calibration_curve(y_val, best_val_calibrated_probs)

    # Evaluate ONCE on Held-Out Test Set
    print("\nEvaluating ONCE on Held-Out Test Set...")
    test_raw_probs = best_model_obj.predict_proba(X_test)[:, 1]
    test_calibrated_probs = calibrator.calibrate(test_raw_probs)

    test_metrics = calculate_metrics(y_test, test_calibrated_probs, threshold=optimal_threshold)
    test_cost = calculate_expected_cost(y_test, test_calibrated_probs, threshold=optimal_threshold, cost_FN=cost_FN, cost_FP=cost_FP)
    test_ci = bootstrap_confidence_intervals(y_test, test_calibrated_probs, threshold=optimal_threshold, n_bootstraps=1000, seed=RANDOM_SEED)

    print("Final Test Set Performance:")
    print(f"   ROC-AUC: {test_metrics['roc_auc']:.4f} (95% CI: [{test_ci['roc_auc']['ci_lower']:.4f}, {test_ci['roc_auc']['ci_upper']:.4f}])")
    print(f"   PR-AUC:  {test_metrics['pr_auc']:.4f} (95% CI: [{test_ci['pr_auc']['ci_lower']:.4f}, {test_ci['pr_auc']['ci_upper']:.4f}])")
    print(f"   F1-Score: {test_metrics['f1']:.4f} (95% CI: [{test_ci['f1']['ci_lower']:.4f}, {test_ci['f1']['ci_upper']:.4f}])")
    print(f"   Recall:  {test_metrics['recall']:.4f} (95% CI: [{test_ci['recall']['ci_lower']:.4f}, {test_ci['recall']['ci_upper']:.4f}])")
    print(f"   Precision: {test_metrics['precision']:.4f} (95% CI: [{test_ci['precision']['ci_lower']:.4f}, {test_ci['precision']['ci_upper']:.4f}])")
    print(f"   Total Test Cost: ${test_cost:,.2f}")

    # Retrain final LightGBM model natively if not already LightGBM for native format saving
    lgb_native_path = os.path.join(MODELS_DIR, "lightgbm_model.txt")
    if isinstance(best_model_obj, LGBMClassifier):
        booster = best_model_obj.booster_
        booster.save_model(lgb_native_path)
    else:
        # Fit a LightGBM model on train data to satisfy native format export constraint
        lgb = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=RANDOM_SEED, verbose=-1)
        lgb.fit(X_train, y_train)
        lgb.booster_.save_model(lgb_native_path)

    # Compute SHA-256 checksum for native model file
    with open(lgb_native_path, "rb") as f:
        model_sha256 = hashlib.sha256(f.read()).hexdigest()

    with open(f"{lgb_native_path}.sha256", "w") as f:
        f.write(model_sha256)

    # Save fitted pipeline & calibrator artifacts
    joblib.dump(pipeline, os.path.join(MODELS_DIR, "pipeline.pkl"))
    joblib.dump(calibrator, os.path.join(MODELS_DIR, "calibrator.pkl"))
    joblib.dump(best_model_obj, os.path.join(MODELS_DIR, "model.pkl"))

    # Save model_artifacts.json
    artifacts_metadata = {
        "model_version": MODEL_VERSION,
        "model_name": best_config["model_name"],
        "strategy": best_config["strategy"],
        "optimal_threshold": optimal_threshold,
        "cost_parameters": {"cost_FN": cost_FN, "cost_FP": cost_FP},
        "feature_names": FEATURE_NAMES,
        "model_sha256": model_sha256,
        "test_metrics": test_metrics,
        "test_cost": test_cost
    }
    with open(os.path.join(MODELS_DIR, "model_artifacts.json"), "w") as f:
        json.dump(artifacts_metadata, f, indent=4)

    # Save complete reports/results.json
    results_data = {
        "model_version": MODEL_VERSION,
        "dataset_deduplication": {
            "initial_rows": len(raw_df),
            "duplicates_removed": dup_count,
            "clean_rows": len(clean_df)
        },
        "temporal_split": split_summary,
        "best_config": best_config,
        "optimal_threshold": optimal_threshold,
        "cost_parameters": {"cost_FN": cost_FN, "cost_FP": cost_FP},
        "validation_calibration_curve": val_calibration_curve,
        "test_metrics": test_metrics,
        "test_cost": test_cost,
        "test_confidence_intervals": test_ci,
        "benchmark_comparison": comparison_results
    }
    with open(os.path.join(REPORTS_DIR, "results.json"), "w") as f:
        json.dump(results_data, f, indent=4)

    print("\nSaved all artifacts and reports/results.json successfully!")
    return results_data



if __name__ == "__main__":
    run_training_pipeline()
