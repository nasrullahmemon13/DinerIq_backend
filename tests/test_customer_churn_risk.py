"""
Unit and Integration Tests for SRS Step 36: Customer Churn-Risk Identification
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from python_pipeline.churn.customer_churn_risk import (
    CustomerChurnRiskAnalyzer,
    run_customer_churn_risk_pipeline
)

CUSTOMERS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "customers", "customers.parquet")
ORDERS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "orders", "orders.parquet")
CUBE_PATH = os.path.join(PROJECT_ROOT, "processed_data", "joined", "master_analytical_cube", "master_analytical_cube.parquet")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "churn")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "churn")


@pytest.fixture(scope="module")
def input_data():
    customers_df = pd.read_parquet(CUSTOMERS_PATH)
    orders_df = pd.read_parquet(ORDERS_PATH)
    cube_df = pd.read_parquet(CUBE_PATH)
    return customers_df, orders_df, cube_df


@pytest.fixture(scope="module")
def churn_results(input_data):
    customers_df, orders_df, cube_df = input_data
    analyzer = CustomerChurnRiskAnalyzer(customers_df, orders_df, cube_df)
    churn_df = analyzer.evaluate_churn_risk()
    factor_summary = analyzer.summarize_risk_factors()
    tier_summary = analyzer.summarize_risk_tiers()
    hv_at_risk = analyzer.get_high_value_at_risk_cohort(top_n=100)
    return analyzer, churn_df, factor_summary, tier_summary, hv_at_risk


def test_all_5_srs_factors_present(churn_results):
    """
    SRS Step 36 requirement:
    Factors may include:
      Increasing recency
      Declining frequency
      Declining monetary value
      Reduced category diversity
      Lower visit frequency
    """
    _, churn_df, factor_summary, _, _ = churn_results

                                               
    expected_flags = [
        "flag_increasing_recency",
        "flag_declining_frequency",
        "flag_declining_monetary",
        "flag_reduced_category_diversity",
        "flag_lower_visit_frequency"
    ]
    for flag_col in expected_flags:
        assert flag_col in churn_df.columns, f"Missing flag column: {flag_col}"
        assert churn_df[flag_col].dtype == bool

                                                   
    expected_scores = [
        "score_recency",
        "score_declining_frequency",
        "score_declining_monetary",
        "score_reduced_category_diversity",
        "score_lower_visit_frequency"
    ]
    for score_col in expected_scores:
        assert score_col in churn_df.columns, f"Missing score column: {score_col}"
        assert ((churn_df[score_col] >= 0.0) & (churn_df[score_col] <= 1.0)).all()

                                                    
    factor_names = set(factor_summary["factor_name"].unique())
    expected_names = {
        "Increasing Recency",
        "Declining Frequency",
        "Declining Monetary Value",
        "Reduced Category Diversity",
        "Lower Visit Frequency"
    }
    assert factor_names == expected_names
    assert len(factor_summary) == 5


def test_all_50000_customers_evaluated(churn_results):
    """Assert all 50,000 customers from the master customer base are evaluated."""
    _, churn_df, _, _, _ = churn_results
    assert len(churn_df) == 50000
    assert churn_df["customer_id"].nunique() == 50000
    assert churn_df["churn_risk_score"].notna().all()


def test_churn_risk_score_and_tier_properties(churn_results):
    """Assert composite score is bounded in [0, 1] and risk tiers are properly structured."""
    _, churn_df, _, tier_summary, _ = churn_results

    assert ((churn_df["churn_risk_score"] >= 0.0) & (churn_df["churn_risk_score"] <= 1.0)).all()

    expected_tiers = {"High Churn Risk", "Medium Churn Risk", "Low Churn Risk"}
    actual_tiers = set(churn_df["churn_risk_tier"].unique())
    assert actual_tiers == expected_tiers

                                            
    high_tier = tier_summary[tier_summary["churn_risk_tier"] == "High Churn Risk"].iloc[0]
    low_tier = tier_summary[tier_summary["churn_risk_tier"] == "Low Churn Risk"].iloc[0]

    assert high_tier["mean_churn_risk_score"] > low_tier["mean_churn_risk_score"]
    assert high_tier["mean_recency_days"] > low_tier["mean_recency_days"]
    assert high_tier["avg_risk_factors_count"] > low_tier["avg_risk_factors_count"]


def test_primary_driver_and_retention_actions(churn_results):
    """Assert primary risk driver identification and actionable prescriptions."""
    _, churn_df, _, _, _ = churn_results

    assert "primary_risk_driver" in churn_df.columns
    assert "recommended_retention_action" in churn_df.columns
    assert (churn_df["recommended_retention_action"].str.len() > 10).all()

                                                        
    high_risk_actions = churn_df[churn_df["churn_risk_tier"] == "High Churn Risk"]["recommended_retention_action"]
    assert not high_risk_actions.str.contains("Standard Loyalty Nurturing").any()


def test_high_value_at_risk_cohort(churn_results):
    """Assert identification of high-value / VIP customers at churn risk."""
    _, _, _, _, hv_at_risk = churn_results

    assert len(hv_at_risk) > 0
                                                             
    valid_segments = {"HIGH_VALUE"}
    valid_loyalty = {"PLATINUM", "GOLD"}
    assert (
        hv_at_risk["customer_segment"].isin(valid_segments) |
        hv_at_risk["loyalty_tier"].isin(valid_loyalty)
    ).all()

                                          
    assert hv_at_risk["churn_risk_tier"].isin(["High Churn Risk", "Medium Churn Risk"]).all()
    assert "revenue_at_risk" in hv_at_risk.columns
    assert (hv_at_risk["revenue_at_risk"] >= 0).all()


def test_pipeline_execution_and_file_artifacts():
    """Assert pipeline runs and persists all parquet, CSV, JSON, and Markdown artifacts."""
    churn_df, factor_summary, tier_summary, stats = run_customer_churn_risk_pipeline()

    assert stats["total_customers_evaluated"] == 50000
    assert stats["high_churn_risk_count"] > 0
    assert stats["total_revenue_evaluated"] > 0

                                 
    assert os.path.exists(os.path.join(OUTPUT_DIR, "customer_churn_risk.parquet"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "customer_churn_risk.csv"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "churn_risk_factor_summary.parquet"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "churn_risk_factor_summary.csv"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "high_value_at_risk.parquet"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "high_value_at_risk.csv"))

                        
    assert os.path.exists(os.path.join(REPORTS_DIR, "customer_churn_risk_report.md"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "customer_churn_risk_summary.json"))

                                
    report_size = os.path.getsize(os.path.join(REPORTS_DIR, "customer_churn_risk_report.md"))
    assert report_size > 1000
