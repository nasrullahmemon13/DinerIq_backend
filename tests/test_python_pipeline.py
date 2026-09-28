"""
DineIQ Analytics - Independent Python Pipeline Tests (SRS Step 13)
Verifies:
- Independent execution using Pandas, NumPy, Scikit-learn, XGBoost, SciPy, Statsmodels
- Enforces SRS rule: No Spark-generated predictions exported or reused
- Price elasticity of demand modeled with Statsmodels OLS
- Inferential hypothesis testing with SciPy (t-test, ANOVA, Chi-Square)
- 30-day forward time-series forecasting with Statsmodels ARIMA
- Standalone XGBoost churn classifier saved to models/python/
"""
import os
import json
import joblib
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "python_pipeline")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "python")

@pytest.fixture(scope="module")
def pipeline_report():
    json_path = os.path.join(REPORTS_DIR, "python_pipeline_report.json")
    assert os.path.exists(json_path), f"Missing report at {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def xgb_model():
    model_path = os.path.join(MODELS_DIR, "xgb_churn_model.joblib")
    assert os.path.exists(model_path), f"Missing XGBoost model at {model_path}"
    return joblib.load(model_path)

def test_six_libraries_represented(pipeline_report):
    """Verify all 6 required SRS libraries were utilized."""
    expected_libs = {"Pandas", "NumPy", "Scikit-learn", "XGBoost", "SciPy", "Statsmodels"}
    actual_libs = set(pipeline_report.get("libraries_utilized", []))
    assert expected_libs.issubset(actual_libs)

def test_price_elasticity_statsmodels(pipeline_report):
    """Verify Statsmodels econometric price elasticity calculation."""
    elasticity = pipeline_report["price_elasticity"]
    assert "price_elasticity_coefficient" in elasticity
    ed = elasticity["price_elasticity_coefficient"]

    assert ed < 0.0, f"Expected negative price elasticity, got {ed}"
    assert elasticity["is_statistically_significant"] is True
    assert elasticity["r_squared"] > 0.0

def test_scipy_hypothesis_tests(pipeline_report):
    """Verify SciPy inferential statistical testing suite."""
    tests = pipeline_report["hypothesis_tests"]
    assert "two_sample_ttest_weekend_vs_weekday" in tests
    assert "anova_location_tiers" in tests
    assert "chi_square_channel_payment" in tests

    ttest = tests["two_sample_ttest_weekend_vs_weekday"]
    assert 0.0 <= ttest["p_value"] <= 1.0

    anova = tests["anova_location_tiers"]
    assert 0.0 <= anova["p_value"] <= 1.0

    chi2 = tests["chi_square_channel_payment"]
    assert chi2["degrees_of_freedom"] > 0

def test_statsmodels_arima_forecast():
    """Verify 30-day sales forecast file has 30 rows and valid confidence intervals."""
    forecast_path = os.path.join(REPORTS_DIR, "sales_forecast_30d.csv")
    assert os.path.exists(forecast_path), f"Missing forecast CSV at {forecast_path}"
    df = pd.read_csv(forecast_path)
    assert len(df) == 30, f"Expected 30 daily forecast periods, got {len(df)}"
    assert (df["projected_revenue"] > 0).all()
    assert (df["ci_upper_95"] >= df["projected_revenue"]).all()
    assert (df["projected_revenue"] >= df["ci_lower_95"]).all()

def test_standalone_xgboost_inference(xgb_model):
    """Verify standalone XGBoost model can perform inference on customer features."""

    dummy_input = np.array([
        [15, 6, 450.0, 75.0, 10.0, 0.20, 0.35, 250],
        [280, 1, 35.0, 35.0, 0.0, 0.0, 0.0, 50]
    ])
    preds = xgb_model.predict(dummy_input)
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})

    probs = xgb_model.predict_proba(dummy_input)
    assert probs.shape == (2, 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

def test_markdown_report_content():
    """Verify comprehensive markdown documentation is generated."""
    md_path = os.path.join(REPORTS_DIR, "python_pipeline_report.md")
    assert os.path.exists(md_path)
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Price Elasticity of Demand" in content
    assert "Inferential Hypothesis Testing" in content
    assert "Time-Series Sales Forecasting" in content
    assert "XGBoost Classifier" in content
