"""
DineIQ Analytics - ML Pipeline Preprocessing & Feature Transformation (SRS Step 12)
Provides vector assembly, standard scaling, and categorical encoding
for PySpark MLlib and Scikit-Learn distributed training workflows.
"""
import os
import sys
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PARQUET_FEATURES_DIR = os.path.join(PROJECT_ROOT, "parquet_data", "features")

FEATURE_COLUMNS = [
    "customer_recency",
    "customer_frequency",
    "customer_monetary_value",
    "average_order_value",
    "basket_size",
    "discount_percentage",
    "promotion_dependency",
    "peak_hour_frequency",
    "weekend_order_ratio",
    "location_performance"
]

TARGET_COLUMN = "is_churned"

def load_preprocessed_dataset():
    """
    Loads customer_master_features.parquet, engineers the binary churn target,
    imputes missing values, and returns clean X, y, and feature metadata.
    """
    parquet_path = os.path.join(PARQUET_FEATURES_DIR, "customer_master_features.parquet")
    df = pd.read_parquet(parquet_path)

    df[TARGET_COLUMN] = (
        (df["customer_recency"] >= 180) | 
        (df["churn_risk_score"] >= 0.50)
    ).astype(int)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    for col in FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())

    return X, y, df
