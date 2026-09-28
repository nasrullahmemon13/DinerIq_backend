"""
DineIQ Analytics - Spark Feature Engineering Transformation Test Suite (SRS Step 7)
Verifies:
- PySpark feature generation outputs in parquet_data/features/
- Menu feature calculations
- Customer behavioral features
- RFM scoring features
"""
import os
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARQUET_FEAT_DIR = os.path.join(PROJECT_ROOT, "parquet_data", "features")


def test_spark_menu_features_file_and_schema():
    """Verify Spark-generated menu features parquet file and schema."""
    menu_pq = os.path.join(PARQUET_FEAT_DIR, "menu_features.parquet")
    assert os.path.exists(menu_pq), f"Missing {menu_pq}"
    df = pd.read_parquet(menu_pq)
    assert len(df) == 150, f"Expected 150 menu items, found {len(df)}"
    expected_cols = [
        "item_id", "item_name", "item_revenue", "cost", "contribution_margin",
        "profit_percentage", "order_frequency", "item_popularity",
        "repeat_purchase_rate", "average_rating", "rating_trend", "wastage_percentage"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing feature column: {col}"


def test_spark_customer_features_file_and_schema():
    """Verify Spark-generated customer features parquet file and schema."""
    cust_pq = os.path.join(PARQUET_FEAT_DIR, "customer_features.parquet")
    assert os.path.exists(cust_pq), f"Missing {cust_pq}"
    df = pd.read_parquet(cust_pq)
    assert len(df) == 50000, f"Expected 50,000 customers, found {len(df)}"
    expected_cols = [
        "customer_id", "basket_size", "discount_percentage", "promotion_dependency",
        "peak_hour_frequency", "weekend_order_ratio", "channel_preference", "location_performance"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing customer feature column: {col}"


def test_spark_rfm_features_file_and_schema():
    """Verify Spark-generated RFM features parquet file and score ranges."""
    rfm_pq = os.path.join(PARQUET_FEAT_DIR, "rfm_features.parquet")
    assert os.path.exists(rfm_pq), f"Missing {rfm_pq}"
    df = pd.read_parquet(rfm_pq)
    assert len(df) == 50000, f"Expected 50,000 customers in RFM, found {len(df)}"
    assert {"customer_recency", "customer_frequency", "customer_monetary_value", "average_order_value", "r_score", "f_score", "m_score", "rfm_segment"}.issubset(df.columns)
    assert df["r_score"].between(1, 5).all(), "r_score outside [1, 5] bounds"
    assert df["f_score"].between(1, 5).all(), "f_score outside [1, 5] bounds"
    assert df["m_score"].between(1, 5).all(), "m_score outside [1, 5] bounds"
