"""
Data Loading, Validation, Deduplication, and Temporal Splitting Module.
"""

import os
from typing import Dict, Any, Tuple
import pandas as pd


REQUIRED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load raw credit card transaction dataset from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset CSV not found at path: {file_path}")
    
    df = pd.read_csv(file_path)
    return df


def validate_dataset(df: pd.DataFrame) -> bool:
    """Validate dataset columns, null values, and datatypes."""
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required dataset columns: {sorted(list(missing_cols))}")
    
    null_counts = df[REQUIRED_COLUMNS].isnull().sum().sum()
    if null_counts > 0:
        raise ValueError(f"Dataset contains {null_counts} null values. Nulls are not allowed.")
    
    return True


def deduplicate_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows before data splitting."""
    initial_count = len(df)
    clean_df = df.drop_duplicates(keep="first").copy()
    duplicates_removed = initial_count - len(clean_df)
    return clean_df, duplicates_removed


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, Any]:
    """
    Perform strict temporal split ordered by Time column:
    First 70% Train, Next 15% Validation, Last 15% Test.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # Sort chronologically by Time
    df_sorted = df.sort_values(by="Time", ascending=True).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    split_summary = {
        "total_rows": n,
        "train": {
            "count": len(train_df),
            "fraud_count": int(train_df["Class"].sum()),
            "legit_count": int((train_df["Class"] == 0).sum()),
            "fraud_percentage": float((train_df["Class"].sum() / len(train_df)) * 100),
            "min_time": float(train_df["Time"].min()),
            "max_time": float(train_df["Time"].max()),
        },
        "val": {
            "count": len(val_df),
            "fraud_count": int(val_df["Class"].sum()),
            "legit_count": int((val_df["Class"] == 0).sum()),
            "fraud_percentage": float((val_df["Class"].sum() / len(val_df)) * 100),
            "min_time": float(val_df["Time"].min()),
            "max_time": float(val_df["Time"].max()),
        },
        "test": {
            "count": len(test_df),
            "fraud_count": int(test_df["Class"].sum()),
            "legit_count": int((test_df["Class"] == 0).sum()),
            "fraud_percentage": float((test_df["Class"].sum() / len(test_df)) * 100),
            "min_time": float(test_df["Time"].min()),
            "max_time": float(test_df["Time"].max()),
        },
    }

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "summary": split_summary,
    }
