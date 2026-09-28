"""
Unit and Integration Tests for DineIQ Price Sensitivity & Elasticity Pipeline (SRS Steps 25-26)
Validates:
- Step 25: Multi-variable relationship analysis across all 7 dimensions:
  * price, demand, revenue, contribution margin, discount, rating, repeat purchase
  * identification of items with statistically significant demand changes after price changes
- Step 26: Classification into EXACTLY the 3 SRS-mandated tiers:
  * Highly Price Sensitive, Moderately Price Sensitive, Low Price Sensitivity
"""
import os
import json
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "pricing")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "pricing")


def test_pricing_output_files_exist():
    """Verify all Step 25 & 26 datasets and reports exist and are non-empty."""
    expected_files = [
        "price_sensitivity_analysis.parquet",
        "price_sensitivity_analysis.csv",
        "multivariate_correlations.parquet",
        "multivariate_correlations.csv",
        "price_change_event_impacts.parquet",
        "price_change_event_impacts.csv",
        "category_price_sensitivity.parquet",
        "category_price_sensitivity.csv"
    ]
    for fname in expected_files:
        path = os.path.join(OUTPUT_DIR, fname)
        assert os.path.exists(path), f"Missing output dataset: {fname}"
        assert os.path.getsize(path) > 0, f"Dataset is empty: {fname}"

    assert os.path.exists(os.path.join(REPORTS_DIR, "price_sensitivity_report.md"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "price_sensitivity_report.json"))


def test_step_25_all_7_variables_analyzed():
    """Verify all 7 SRS-required variables are integrated and analyzed."""
    df = pd.read_parquet(os.path.join(OUTPUT_DIR, "price_sensitivity_analysis.parquet"))
    assert len(df) == 150, f"Expected 150 menu items, got {len(df)}"

    required_vars = [
        "price",
        "demand",
        "revenue",
        "contribution_margin",
        "discount",
        "rating",
        "repeat_purchase"
    ]
    for var in required_vars:
        assert var in df.columns, f"Missing required variable: {var}"
        assert df[var].notna().all(), f"Found NaN values in {var}"


    corr_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "multivariate_correlations.parquet"))
    assert corr_df.shape == (7, 7), f"Expected 7x7 correlation matrix, got {corr_df.shape}"
    for col in required_vars:
        assert np.isclose(corr_df.loc[col, col], 1.0, atol=1e-3), f"Diagonal not 1.0 for {col}"


def test_step_25_price_change_event_impacts():
    """Verify empirical price change impact analysis and significant demand shift identification."""
    impact_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "price_change_event_impacts.parquet"))
    assert len(impact_df) > 50, f"Expected >50 price change events, got {len(impact_df)}"

    required_cols = [
        "pct_price_change",
        "pct_demand_change",
        "empirical_elasticity",
        "t_statistic",
        "p_value",
        "is_statistically_significant"
    ]
    for col in required_cols:
        assert col in impact_df.columns, f"Missing event impact column: {col}"


    sig_count = impact_df["is_statistically_significant"].sum()
    assert sig_count > 0, "No statistically significant demand changes identified"


def test_step_26_exact_3_tier_classification():
    """Verify all items are classified into EXACTLY the 3 SRS categories."""
    df = pd.read_parquet(os.path.join(OUTPUT_DIR, "price_sensitivity_analysis.parquet"))
    assert "price_sensitivity_tier" in df.columns

    assigned_tiers = set(df["price_sensitivity_tier"].unique())
    expected_tiers = {
        "Highly Price Sensitive",
        "Moderately Price Sensitive",
        "Low Price Sensitivity"
    }
    assert assigned_tiers == expected_tiers, f"Tiers mismatch: {assigned_tiers} vs {expected_tiers}"


    counts = df["price_sensitivity_tier"].value_counts().to_dict()
    for tier in expected_tiers:
        assert counts.get(tier, 0) > 0, f"Tier {tier} is empty"

    assert sum(counts.values()) == 150, "Total classified items must equal 150"


def test_price_sensitivity_report_content():
    """Verify report contains all required metrics, OLS results, and category tables."""
    md_path = os.path.join(REPORTS_DIR, "price_sensitivity_report.md")
    json_path = os.path.join(REPORTS_DIR, "price_sensitivity_report.json")

    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    assert "Price Sensitivity & Econometric Demand Elasticity" in md
    assert "Executive Summary & Step 26 Classification Tiers" in md
    assert "7-Variable Pearson Correlation Matrix" in md
    assert "Econometric Log-Log Regression Model" in md
    assert "Items with Significant Demand Shifts After Price Changes" in md
    assert "Category-Level Price Sensitivity Summary" in md
    assert "Highly Price Sensitive" in md
    assert "Moderately Price Sensitive" in md
    assert "Low Price Sensitivity" in md

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "regression_summary" in data
    assert "tier_distribution" in data
    assert "category_summary" in data
    assert "significant_price_change_items" in data
    assert data["regression_summary"]["r_squared"] > 0.5
