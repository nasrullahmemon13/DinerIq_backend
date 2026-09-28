"""
Unit and Integration Tests for DineIQ Wastage Analysis & Risk Modeling (SRS Steps 23-24)
Validates:
- Step 23: Multi-dimensional wastage analysis across all 9 SRS-defined dimensions:
  * menu item, category, location, day, time period, demand, inventory consumption, promotion, preparation quantity
- Step 24: Predictive wastage risk model utilizing all 9 potential variables SRS lists:
  * historical demand, historical wastage, day of week, season, location, promotion, menu popularity, forecast demand, preparation quantity
"""
import os
import json
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "wastage")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "wastage")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "wastage")


def test_wastage_output_files_exist():
    """Verify all Step 23 & 24 parquet and csv datasets exist and are non-empty."""
    expected_datasets = [
        "wastage_by_item",
        "wastage_by_category",
        "wastage_by_location",
        "wastage_by_day",
        "wastage_by_time_period",
        "wastage_by_demand",
        "wastage_by_inventory",
        "wastage_by_promotion",
        "wastage_by_preparation",
        "wastage_risk_predictions"
    ]
    for dname in expected_datasets:
        pq_path = os.path.join(OUTPUT_DIR, f"{dname}.parquet")
        csv_path = os.path.join(OUTPUT_DIR, f"{dname}.csv")
        assert os.path.exists(pq_path), f"Missing parquet file: {pq_path}"
        assert os.path.getsize(pq_path) > 0, f"Parquet file is empty: {pq_path}"
        assert os.path.exists(csv_path), f"Missing csv file: {csv_path}"
        assert os.path.getsize(csv_path) > 0, f"CSV file is empty: {csv_path}"


    assert os.path.exists(os.path.join(MODELS_DIR, "wastage_risk_regressor.joblib"))
    assert os.path.exists(os.path.join(MODELS_DIR, "wastage_risk_classifier.joblib"))


def test_step_23_all_9_dimensions_validity():
    """Verify data integrity across all 9 SRS wastage analysis dimensions."""

    item_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "wastage_by_item.parquet"))
    assert len(item_df) > 100, f"Expected >100 menu items, got {len(item_df)}"
    assert "wasted_quantity" in item_df.columns and "total_loss_amount" in item_df.columns
    assert (item_df["total_loss_amount"] >= 0).all()


    cat_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "wastage_by_category.parquet"))
    assert len(cat_df) == 10, f"Expected 10 categories, got {len(cat_df)}"
    assert np.isclose(cat_df["loss_share_pct"].sum(), 100.0, atol=0.5)


    loc_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "wastage_by_location.parquet"))
    assert len(loc_df) == 20, f"Expected 20 locations, got {len(loc_df)}"
    assert "restaurant_name" in loc_df.columns and "wastage_rate_pct" in loc_df.columns


    day_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "wastage_by_day.parquet"))
    assert len(day_df) == 7, f"Expected 7 days of the week, got {len(day_df)}"
    assert np.isclose(day_df["loss_share_pct"].sum(), 100.0, atol=0.5)


    shift_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "wastage_by_time_period.parquet"))
    assert len(shift_df) >= 3, f"Expected at least 3 kitchen shifts, got {len(shift_df)}"
    assert "shift_period" in shift_df.columns


    demand_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "wastage_by_demand.parquet"))
    assert len(demand_df) == 4, f"Expected 4 demand quartiles, got {len(demand_df)}"
    assert "demand_to_wastage_ratio" in demand_df.columns


    inv_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "wastage_by_inventory.parquet"))
    assert "stock_consumption_ratio_pct" in inv_df.columns
    assert "inventory_wastage_rate_pct" in inv_df.columns


    promo_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "wastage_by_promotion.parquet"))
    assert len(promo_df) >= 2, "Expected Active Promotion and Standard Non-Promo"
    assert "overproduction_factor" in promo_df.columns


    prep_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "wastage_by_preparation.parquet"))
    assert len(prep_df) == 4, f"Expected 4 prep batch tiers, got {len(prep_df)}"
    assert "over_prep_index_pct" in prep_df.columns


def test_step_24_predictive_model_performance():
    """Verify Step 24 model metrics and feature importance coverage."""
    preds_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "wastage_risk_predictions.parquet"))


    assert "predicted_quantity_wasted" in preds_df.columns
    assert "predicted_high_risk" in preds_df.columns
    assert "predicted_risk_probability" in preds_df.columns
    assert "actionable_mitigation_strategy" in preds_df.columns


    assert (preds_df["predicted_risk_probability"] >= 0.0).all()
    assert (preds_df["predicted_risk_probability"] <= 1.0).all()


    json_path = os.path.join(REPORTS_DIR, "wastage_risk_report.json")
    assert os.path.exists(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    eval_summary = data["evaluation_summary"]
    reg = eval_summary["regression_metrics"]
    cls_m = eval_summary["classification_metrics"]

    assert reg["mae"] < 2.0, f"MAE too high: {reg['mae']}"
    assert cls_m["accuracy"] >= 0.80, f"Accuracy too low: {cls_m['accuracy']}"
    assert cls_m["roc_auc"] >= 0.85, f"ROC-AUC too low: {cls_m['roc_auc']}"


    features_evaluated = [fi["feature_name"] for fi in eval_summary["feature_importance"]]
    expected_vars = [
        "historical_demand_7d",
        "historical_wastage_7d",
        "dow_code",
        "season_code",
        "location_code",
        "is_promo_active",
        "popularity_weight",
        "forecast_demand",
        "preparation_quantity"
    ]
    for var in expected_vars:
        assert var in features_evaluated, f"Missing SRS predictor variable: {var}"


def test_wastage_risk_report_content():
    """Verify comprehensive markdown report includes all required sections and tables."""
    md_path = os.path.join(REPORTS_DIR, "wastage_risk_report.md")
    assert os.path.exists(md_path)

    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    assert "Food Wastage Analysis & Predictive Risk Modeling" in md
    assert "Step 24 Model Performance" in md
    assert "Predictor Feature Importance Ranking" in md
    assert "1. By Menu Item" in md
    assert "2. By Menu Category" in md
    assert "3. By Restaurant Location" in md
    assert "4. By Day of Week" in md
    assert "5. By Time Period" in md
    assert "6. By Demand Levels" in md
    assert "7. By Inventory Consumption" in md
    assert "8. By Promotion Activity" in md
    assert "9. By Preparation Quantity" in md
    assert "High-Risk Operational Early-Warning Alerts" in md
