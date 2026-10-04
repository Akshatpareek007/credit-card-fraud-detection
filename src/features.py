"""
Feature Engineering and Scaler Preprocessing Pipeline Module.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["LogAmount", "Hour"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer to compute cyclic Hour and log-transformed Amount."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_out = X.copy()

        # Compute cyclic Hour of the day from Time
        if "Time" in df_out.columns:
            df_out["Hour"] = (df_out["Time"] // 3600) % 24
        elif "Hour" not in df_out.columns:
            df_out["Hour"] = 12.0

        # Compute LogAmount from Amount
        if "Amount" in df_out.columns:
            df_out["LogAmount"] = np.log1p(np.maximum(0.0, df_out["Amount"].astype(float)))
        elif "LogAmount" not in df_out.columns:
            df_out["LogAmount"] = 0.0

        # Ensure all required V1..V28 exist
        for i in range(1, 29):
            col = f"V{i}"
            if col not in df_out.columns:
                df_out[col] = 0.0

        return df_out[FEATURE_NAMES]


def build_preprocessing_pipeline() -> Pipeline:
    """Build sklearn Pipeline containing FeatureEngineer and StandardScaler."""
    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("scaler", StandardScaler())
    ])


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    """
    Prepare feature matrices by fitting pipeline STRICTLY on train_df only.
    Returns scaled feature arrays and fitted preprocessing pipeline.
    """
    pipeline = build_preprocessing_pipeline()

    # Fit pipeline ONLY on training data
    X_train_scaled = pipeline.fit_transform(train_df)
    y_train = train_df["Class"].values.astype(int)

    # Transform val and test using fitted pipeline
    X_val_scaled = pipeline.transform(val_df)
    y_val = val_df["Class"].values.astype(int)

    X_test_scaled = pipeline.transform(test_df)
    y_test = test_df["Class"].values.astype(int)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, pipeline
