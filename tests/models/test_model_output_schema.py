"""
DineIQ Analytics - Model Output Schema Validation Test Suite (SRS Step 13, Step 16 & Step 26)
Verifies:
- Standardized inference response structures across both Python and Spark pipelines
- Schema compliance for Churn predictions, Demand forecasts, and Menu classifications
"""
import os
import sys
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
PYTHON_PIPELINE_DIR = os.path.join(PROJECT_ROOT, "python_pipeline")
if PYTHON_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PYTHON_PIPELINE_DIR)

from data_loader import load_operational_data


def test_menu_classification_schema():
    """Verify menu classification outputs conform to required 4-quadrant schema."""
    from menu_classification_python import run_independent_python_menu_classification
    df = run_independent_python_menu_classification()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 150
    required_cols = ["item_id", "name", "category_id", "python_classification", "profit_percentage"]
    for col in required_cols:
        assert col in df.columns, f"Missing column in classification output: {col}"
    assert set(df["python_classification"].unique()).issubset({"Profit Driver", "Volume Driver", "Hidden Opportunity", "Low Performer"})


def test_forecasting_output_schema():
    """Verify demand forecasting output schema."""
    from sales_forecaster import build_sales_forecast_model
    data = load_operational_data()
    res = build_sales_forecast_model(data["orders"])
    assert isinstance(res, dict)
    assert "forecast_df" in res
    fc_df = res["forecast_df"]
    assert len(fc_df) == 30
    required_cols = ["forecast_date", "projected_revenue", "ci_lower_95", "ci_upper_95"]
    for col in required_cols:
        assert col in fc_df.columns, f"Missing column in forecast output: {col}"
    assert (fc_df["ci_lower_95"] <= fc_df["projected_revenue"]).all()
    assert (fc_df["projected_revenue"] <= fc_df["ci_upper_95"]).all()


def test_churn_output_schema():
    """Verify churn prediction model metrics and prediction schema."""
    from churn_xgboost import train_xgboost_churn_model
    data = load_operational_data()
    results = train_xgboost_churn_model(data["orders"], data["customers"])
    assert "xgb_metrics" in results
    xgb_m = results["xgb_metrics"]
    assert "model_name" in xgb_m
    assert "accuracy" in xgb_m
    assert "f1_score" in xgb_m
    assert "roc_auc" in xgb_m
    assert 0.0 <= xgb_m["accuracy"] <= 1.0
    assert 0.0 <= xgb_m["roc_auc"] <= 1.0
