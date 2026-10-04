"""
Tests for Data Processing, Validation, Deduplication, and Temporal Splitting.
"""

import pytest
import numpy as np
import pandas as pd
from src.data import validate_dataset, deduplicate_dataset, temporal_split
from src.features import prepare_features


def test_deduplicate_dataset():
    data = {
        "Time": [1.0, 2.0, 2.0, 3.0],
        "Amount": [10.0, 20.0, 20.0, 30.0],
        "Class": [0, 1, 1, 0]
    }
    for i in range(1, 29):
        data[f"V{i}"] = [0.1] * 4

    df = pd.DataFrame(data)
    clean_df, dups_removed = deduplicate_dataset(df)
    assert dups_removed == 1
    assert len(clean_df) == 3


def test_temporal_split():
    records = []
    for i in range(100):
        row = {"Time": float(i * 10), "Amount": float(i * 5), "Class": 1 if i % 10 == 0 else 0}
        for v in range(1, 29):
            row[f"V{v}"] = 0.0
        records.append(row)

    df = pd.DataFrame(records)
    res = temporal_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    train_df = res["train"]
    val_df = res["val"]
    test_df = res["test"]

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Test temporal ordering
    assert train_df["Time"].max() <= val_df["Time"].min()
    assert val_df["Time"].max() <= test_df["Time"].min()


def test_scaler_fitted_on_train_only():
    train_records = []
    for i in range(50):
        row = {"Time": float(i), "Amount": 100.0, "Class": 0}
        for v in range(1, 29):
            row[f"V{v}"] = 1.0
        train_records.append(row)

    val_records = []
    for i in range(50, 70):
        row = {"Time": float(i), "Amount": 1000.0, "Class": 1}
        for v in range(1, 29):
            row[f"V{v}"] = 10.0
        val_records.append(row)

    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)
    test_df = val_df.copy()

    X_train, y_train, X_val, y_val, X_test, y_test, pipeline = prepare_features(train_df, val_df, test_df)

    scaler = pipeline.named_steps["scaler"]

    # Verify mean is fitted on train data (where Amount=100 -> LogAmount ~ 4.615)
    # and NOT influenced by validation data (where Amount=1000 -> LogAmount ~ 6.908)
    assert X_train.shape[0] == 50
    assert abs(X_train.mean(axis=0)[0]) < 1e-5  # Standardized mean close to 0 on train
