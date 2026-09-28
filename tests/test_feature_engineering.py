"""
DineIQ Analytics - Feature Engineering Tests (SRS Step 7)
Verifies:
- All 21+ required SRS features exist in Parquet outputs
- Menu features: item revenue, cost, contribution margin, profit percentage, order frequency,
                 item popularity, repeat-purchase rate, average rating, rating trend,
                 wastage percentage, price-change percentage
- Customer features: basket size, discount percentage, promotion dependency, peak-hour frequency,
                     weekend-order ratio, channel preference, location performance
- RFM features: customer recency, customer frequency, customer monetary value, average order value
- Mathematical invariants and range bounds
"""
import os
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARQUET_FEATURES_DIR = os.path.join(PROJECT_ROOT, "parquet_data", "features")

@pytest.fixture(scope="module")
def menu_features_df():
    path = os.path.join(PARQUET_FEATURES_DIR, "menu_features.parquet")
    assert os.path.exists(path), f"Missing menu_features.parquet at {path}"
    return pd.read_parquet(path)

@pytest.fixture(scope="module")
def customer_features_df():
    path = os.path.join(PARQUET_FEATURES_DIR, "customer_features.parquet")
    assert os.path.exists(path), f"Missing customer_features.parquet at {path}"
    return pd.read_parquet(path)

@pytest.fixture(scope="module")
def rfm_features_df():
    path = os.path.join(PARQUET_FEATURES_DIR, "rfm_features.parquet")
    assert os.path.exists(path), f"Missing rfm_features.parquet at {path}"
    return pd.read_parquet(path)

@pytest.fixture(scope="module")
def customer_master_df():
    path = os.path.join(PARQUET_FEATURES_DIR, "customer_master_features.parquet")
    assert os.path.exists(path), f"Missing customer_master_features.parquet at {path}"
    return pd.read_parquet(path)

def test_menu_features_presence(menu_features_df):
    """Verify all 11 menu features are present and non-null."""
    expected_cols = [
        "item_revenue", "cost", "contribution_margin", "profit_percentage",
        "order_frequency", "item_popularity", "repeat_purchase_rate",
        "average_rating", "rating_trend", "wastage_percentage", "price_change_percentage"
    ]
    for col in expected_cols:
        assert col in menu_features_df.columns, f"Missing menu feature '{col}'"
        assert menu_features_df[col].isnull().sum() == 0, f"Null values found in menu feature '{col}'"

def test_menu_features_math_invariants(menu_features_df):
    """Verify mathematical consistency of menu financial metrics."""
                                               
    diff = (menu_features_df["item_revenue"] - menu_features_df["cost"]) - menu_features_df["contribution_margin"]
    assert (diff.abs() < 0.05).all(), "Contribution margin must equal item_revenue - cost"
    
                                 
    assert (menu_features_df["average_rating"] >= 1.0).all()
    assert (menu_features_df["average_rating"] <= 5.0).all()

                                          
    assert (menu_features_df["repeat_purchase_rate"] >= 0.0).all()
    assert (menu_features_df["repeat_purchase_rate"] <= 1.0).all()

def test_customer_features_presence(customer_features_df):
    """Verify customer behavioral features are present and non-null."""
    expected_cols = [
        "basket_size", "discount_percentage", "promotion_dependency",
        "peak_hour_frequency", "weekend_order_ratio", "channel_preference",
        "location_performance"
    ]
    for col in expected_cols:
        assert col in customer_features_df.columns, f"Missing customer feature '{col}'"
        assert customer_features_df[col].isnull().sum() == 0, f"Null values found in customer feature '{col}'"

def test_rfm_features_presence(rfm_features_df):
    """Verify RFM customer value features are present and non-null."""
    expected_cols = [
        "customer_recency", "customer_frequency", "customer_monetary_value",
        "average_order_value", "rfm_segment"
    ]
    for col in expected_cols:
        assert col in rfm_features_df.columns, f"Missing RFM feature '{col}'"
        assert rfm_features_df[col].isnull().sum() == 0, f"Null values found in RFM feature '{col}'"

def test_customer_master_features_completeness(customer_master_df):
    """Verify consolidated master feature store joins customer behaviors and RFM metrics."""
    assert len(customer_master_df) == 50000, "Master customer feature store must contain 50,000 customers"
    assert "rfm_segment" in customer_master_df.columns
    assert "channel_preference" in customer_master_df.columns
    assert "basket_size" in customer_master_df.columns
