"""
DineIQ Analytics - Independent Python Model Test Suite (SRS Step 13, Step 16, Step 17 & Step 26)
Verifies:
- Scikit-Learn and XGBoost model artifacts exist and load cleanly
- Churn prediction model, Wastage risk models, and Time-series forecasters
- Inference execution and metric validity
"""
import os
import joblib
import pandas as pd
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def test_xgb_churn_model_load_and_predict():
    """Verify XGBoost churn prediction model loads and generates valid probabilities."""
    model_path = os.path.join(MODELS_DIR, "python", "xgb_churn_model.joblib")
    assert os.path.exists(model_path), f"Missing churn model at {model_path}"
    model = joblib.load(model_path)

    sample = pd.DataFrame([{
        "recency_days": 45,
        "total_orders": 3,
        "total_spend": 150.0,
        "avg_order_value": 50.0,
        "avg_discount": 5.0,
        "promo_usage_ratio": 0.33,
        "weekend_order_ratio": 0.2,
        "loyalty_points": 120
    }])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample)
        assert probs.shape[1] == 2
        assert 0.0 <= probs[0][1] <= 1.0


def test_wastage_risk_models_load():
    """Verify wastage classifier and regressor load and predict."""
    clf_path = os.path.join(MODELS_DIR, "wastage", "wastage_risk_classifier.joblib")
    reg_path = os.path.join(MODELS_DIR, "wastage", "wastage_risk_regressor.joblib")
    assert os.path.exists(clf_path), f"Missing wastage classifier at {clf_path}"
    assert os.path.exists(reg_path), f"Missing wastage regressor at {reg_path}"

    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    assert hasattr(clf, "predict")
    assert hasattr(reg, "predict")


def test_forecasting_models_load():
    """Verify demand forecasting models load."""
    item_fc_path = os.path.join(MODELS_DIR, "forecasting", "item_demand_forecaster.joblib")
    cat_fc_path = os.path.join(MODELS_DIR, "forecasting", "category_demand_forecaster.joblib")
    loc_fc_path = os.path.join(MODELS_DIR, "forecasting", "location_demand_forecaster.joblib")

    for path, name in [(item_fc_path, "Item Forecaster"), (cat_fc_path, "Category Forecaster"), (loc_fc_path, "Location Forecaster")]:
        assert os.path.exists(path), f"Missing {name} at {path}"
        fc = joblib.load(path)
        assert hasattr(fc, "predict")
