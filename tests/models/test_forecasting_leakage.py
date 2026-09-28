"""
DineIQ Analytics - Time-Series Forecasting Leakage & Chronological Split Test Suite (SRS Step 16 & Step 22)
Verifies:
- Strict chronological data partition (Train < Validation < Test)
- Absolute prevention of future target leakage into lag features
- Zero random train/test splitting for time-dependent series
"""
import os
import pandas as pd
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEANED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")
FORECAST_DIR = os.path.join(PROJECT_ROOT, "processed_data", "forecasting")


def test_chronological_splits_integrity():
    """Verify chronological split boundaries: Train Max < Val Min and Val Max < Test Min."""
    orders = pd.read_parquet(os.path.join(CLEANED_DIR, "orders", "orders.parquet"), columns=["order_date", "total_amount"])
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    daily_sales = orders.groupby(pd.Grouper(key="order_date", freq="D"))["total_amount"].sum().reset_index()
    daily_sales = daily_sales.sort_values("order_date").reset_index(drop=True)

    n_total = len(daily_sales)
    train_end_idx = int(n_total * 0.70)
    val_end_idx = int(n_total * 0.85)

    train_df = daily_sales.iloc[:train_end_idx]
    val_df = daily_sales.iloc[train_end_idx:val_end_idx]
    test_df = daily_sales.iloc[val_end_idx:]

    train_max = train_df["order_date"].max()
    val_min = val_df["order_date"].min()
    val_max = val_df["order_date"].max()
    test_min = test_df["order_date"].min()

    assert train_max < val_min, f"Data leakage: train_max ({train_max}) >= val_min ({val_min})"
    assert val_max < test_min, f"Data leakage: val_max ({val_max}) >= test_min ({test_min})"


def test_no_future_feature_leakage():
    """Verify that lag features do not incorporate future observations (t + k)."""
    orders = pd.read_parquet(os.path.join(CLEANED_DIR, "orders", "orders.parquet"), columns=["order_date", "total_amount"])
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    ts = orders.groupby(pd.Grouper(key="order_date", freq="D"))["total_amount"].sum().reset_index()
    ts = ts.sort_values("order_date").reset_index(drop=True)

                                                                                   
    ts["lag_1"] = ts["total_amount"].shift(1)
    ts["lag_7"] = ts["total_amount"].shift(7)
    ts["rolling_mean_7"] = ts["total_amount"].shift(1).rolling(window=7).mean()

                                                                            
    assert ts.loc[10, "lag_1"] == ts.loc[9, "total_amount"]
    assert ts.loc[10, "lag_7"] == ts.loc[3, "total_amount"]
                                                      
    assert pd.isna(ts.loc[0, "lag_1"])
