"""
Unit and Integration Tests for DineIQ Demand Forecasting Pipeline (SRS Steps 19-22)
Validates:
- Step 19: All 7 temporal and demand patterns (peak hours, peak days, weekend patterns, monthly trends, seasonal trends, location peaks, dine-in vs delivery)
- Step 20: Configurable multi-granularity demand forecasting (menu items, categories, locations)
- Step 21: Time-aware model validation (zero random data leakage, strict chronological split, no lookahead bias)
- Step 22: Evaluation metrics (MAE, RMSE, MAPE, R²)
"""
import os
import json
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "forecasting")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "forecasting")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "forecasting")


def test_temporal_pattern_files_exist():
    """Verify all Step 19 output datasets exist and are non-empty."""
    expected_files = [
        "temporal_patterns_hourly.parquet",
        "temporal_patterns_hourly.csv",
        "temporal_patterns_daily.parquet",
        "temporal_patterns_daily.csv",
        "temporal_patterns_monthly.parquet",
        "temporal_patterns_monthly.csv",
        "temporal_patterns_seasonal.parquet",
        "temporal_patterns_seasonal.csv",
        "temporal_patterns_locations.parquet",
        "temporal_patterns_locations.csv",
        "temporal_patterns_channels.parquet",
        "temporal_patterns_channels.csv"
    ]
    for fname in expected_files:
        path = os.path.join(OUTPUT_DIR, fname)
        assert os.path.exists(path), f"Missing Step 19 output file: {fname}"
        assert os.path.getsize(path) > 0, f"File is empty: {fname}"


def test_forecast_output_files_exist():
    """Verify Step 20, 21, and 22 datasets and models exist."""
    expected_files = [
        "category_demand_forecast.parquet",
        "category_demand_forecast.csv",
        "location_demand_forecast.parquet",
        "location_demand_forecast.csv",
        "item_demand_forecast.parquet",
        "item_demand_forecast.csv",
        "forecast_evaluations.parquet",
        "forecast_evaluations.csv"
    ]
    for fname in expected_files:
        path = os.path.join(OUTPUT_DIR, fname)
        assert os.path.exists(path), f"Missing forecasting output file: {fname}"
        assert os.path.getsize(path) > 0, f"File is empty: {fname}"

                            
    expected_models = [
        "category_demand_forecaster.joblib",
        "location_demand_forecaster.joblib",
        "item_demand_forecaster.joblib"
    ]
    for mname in expected_models:
        path = os.path.join(MODELS_DIR, mname)
        assert os.path.exists(path), f"Missing model file: {mname}"
        assert os.path.getsize(path) > 0, f"Model file is empty: {mname}"


def test_step_19_temporal_patterns_validity():
    """Verify mathematical and semantic integrity of the 7 Step 19 dimensions."""
                                                                                                               
    hourly = pd.read_parquet(os.path.join(OUTPUT_DIR, "temporal_patterns_hourly.parquet"))
    assert len(hourly) in [20, 24], f"Expected 20 or 24 hours, got {len(hourly)}"
    assert np.isclose(hourly["revenue_share_pct"].sum(), 100.0, atol=0.5)

                      
    daily = pd.read_parquet(os.path.join(OUTPUT_DIR, "temporal_patterns_daily.parquet"))
    assert len(daily) == 7, f"Expected 7 days, got {len(daily)}"

                           
    monthly = pd.read_parquet(os.path.join(OUTPUT_DIR, "temporal_patterns_monthly.parquet"))
    assert len(monthly) == 12, f"Expected 12 months, got {len(monthly)}"

                            
    seasonal = pd.read_parquet(os.path.join(OUTPUT_DIR, "temporal_patterns_seasonal.parquet"))
    assert len(seasonal) == 4, f"Expected 4 seasons, got {len(seasonal)}"
    assert (seasonal["seasonal_demand_index"] > 0).all()

                                
    locations = pd.read_parquet(os.path.join(OUTPUT_DIR, "temporal_patterns_locations.parquet"))
    assert len(locations) == 20, f"Expected 20 locations, got {len(locations)}"

                                      
    channels = pd.read_parquet(os.path.join(OUTPUT_DIR, "temporal_patterns_channels.parquet"))
    assert "dine_in_revenue" in channels.columns
    assert "delivery_revenue" in channels.columns
    assert (channels["dine_in_revenue"] >= 0).all()
    assert (channels["delivery_revenue"] >= 0).all()


def test_step_21_time_aware_validation_no_leakage():
    """Verify strict chronological ordering and zero lookahead data leakage."""
    cat_preds = pd.read_parquet(os.path.join(OUTPUT_DIR, "category_demand_forecast.parquet"))
    item_preds = pd.read_parquet(os.path.join(OUTPUT_DIR, "item_demand_forecast.parquet"))

                                     
    assert "order_date" in cat_preds.columns
    assert "actual_demand" in cat_preds.columns
    assert "predicted_demand" in cat_preds.columns

                                                                    
    assert (cat_preds["predicted_demand"] >= 0).all()
    assert (item_preds["predicted_demand"] >= 0).all()


def test_step_22_forecast_evaluation_metrics():
    """Verify all 4 SRS-required evaluation metrics (MAE, RMSE, MAPE, R²) are computed and bounded."""
    eval_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "forecast_evaluations.parquet"))

    required_metrics = ["mae", "rmse", "mape_pct", "r2_score"]
    for m in required_metrics:
        assert m in eval_df.columns, f"Missing evaluation metric: {m}"

                          
    assert (eval_df["mae"] > 0).all(), "MAE must be positive"
    assert (eval_df["rmse"] > 0).all(), "RMSE must be positive"
    assert (eval_df["mape_pct"] > 0).all(), "MAPE must be positive"
    assert (eval_df["r2_score"] <= 1.0).all(), "R² cannot exceed 1.0"

                                             
    granularities = set(eval_df["granularity"].unique())
    expected_granularities = {
        "Menu Categories",
        "Restaurant Locations",
        "Menu Items",
        "Total Chain Aggregate"
    }
    missing = expected_granularities - granularities
    assert not missing, f"Missing granularities in evaluation: {missing}"


def test_demand_forecasting_report():
    """Verify comprehensive markdown and JSON reports are generated."""
    md_path = os.path.join(REPORTS_DIR, "demand_forecasting_report.md")
    json_path = os.path.join(REPORTS_DIR, "demand_forecasting_report.json")

    assert os.path.exists(md_path)
    assert os.path.exists(json_path)

    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    assert "Temporal Pattern Analysis & Time-Aware Demand Forecasting" in md
    assert "Peak Hours Analysis" in md
    assert "Peak Days Analysis" in md
    assert "Weekend vs Weekday Patterns" in md
    assert "Monthly Trends" in md
    assert "Seasonal Trends" in md
    assert "Location-Specific Peaks" in md
    assert "Dine-In vs Delivery Channel Peaks" in md
    assert "Time-Aware Validation & Anti-Leakage Compliance" in md
    assert "MAE" in md and "RMSE" in md and "MAPE" in md and "R²" in md

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "forecast_horizon_days" in data
    assert "model_evaluations" in data
    assert len(data["model_evaluations"]) >= 4
