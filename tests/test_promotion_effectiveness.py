"""
Unit and Integration Tests for DineIQ Promotion Effectiveness & Trap Detection (SRS Steps 27-28)
Validates:
- Step 27: Multi-dimensional promotion evaluation across all 8 dimensions:
  * order volume, revenue, contribution margin, customer acquisition, repeat purchases, AOV, wastage, post-promo behavior
  * enforcement of SRS explicit rule: 'An increase in sales alone must not automatically classify a promotion as successful.'
- Step 28: Promotion Trap Detection identifying EXACTLY the 5 SRS-listed misleading patterns:
  1. sales increase but profit decreases
  2. customer count increases but average margin collapses
  3. promotion increases wastage
  4. customers purchase only while discounts are active
  5. promotion shifts sales away from a more profitable product
"""
import os
import json
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "promotion")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "promotion")


def test_promotion_output_files_exist():
    """Verify all Step 27 & 28 datasets and reports exist and are non-empty."""
    expected_files = [
        "promotion_evaluations.parquet",
        "promotion_evaluations.csv",
        "promotion_trap_detection.parquet",
        "promotion_trap_detection.csv",
        "promotion_post_behavior.parquet",
        "promotion_post_behavior.csv",
        "promotion_cannibalization_analysis.parquet",
        "promotion_cannibalization_analysis.csv"
    ]
    for fname in expected_files:
        path = os.path.join(OUTPUT_DIR, fname)
        assert os.path.exists(path), f"Missing promotion dataset: {fname}"
        assert os.path.getsize(path) > 0, f"Dataset is empty: {fname}"

    assert os.path.exists(os.path.join(REPORTS_DIR, "promotion_effectiveness_report.md"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "promotion_effectiveness_report.json"))


def test_step_27_all_8_dimensions_evaluated():
    """Verify every promotion is evaluated across all 8 SRS dimensions."""
    eval_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "promotion_evaluations.parquet"))
    assert len(eval_df) == 12, f"Expected 12 promotions, got {len(eval_df)}"

    required_dimensions = [
        "order_volume",
        "promotional_revenue",
        "contribution_margin_dollars",
        "profit_margin_pct",
        "acquired_customers",
        "repeat_purchase_rate_pct",
        "average_order_value",
        "wasted_loss_amount",
        "post_promo_retention_rate_pct"
    ]
    for dim in required_dimensions:
        assert dim in eval_df.columns, f"Missing required dimension: {dim}"
        assert eval_df[dim].notna().all(), f"Found NaN values in dimension: {dim}"


    statuses = set(eval_df["evaluation_status"].unique())
    assert any("Promotion Trap" in s or "FAILED" in s for s in statuses), (
        "SRS Explicit Rule Violation: All promotions were marked successful despite traps"
    )


def test_step_28_all_5_misleading_trap_patterns():
    """Verify EXACTLY the 5 SRS-listed misleading patterns are detected with empirical evidence."""
    traps_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "promotion_trap_detection.parquet"))
    assert len(traps_df) == 12, f"Expected 12 promotions, got {len(traps_df)}"

    expected_trap_columns = [
        "trap_1_sales_up_profit_down",
        "trap_2_customers_up_margin_collapse",
        "trap_3_increases_wastage",
        "trap_4_discount_only_buyers",
        "trap_5_cannibalization"
    ]
    for trap_col in expected_trap_columns:
        assert trap_col in traps_df.columns, f"Missing trap flag: {trap_col}"

        evidence_col = trap_col.replace("trap_", "trap_").split("_")[0] + "_" + trap_col.split("_")[1] + "_evidence"
        matching_evidence = [c for c in traps_df.columns if c.startswith(trap_col.split("_")[0] + "_" + trap_col.split("_")[1]) and "evidence" in c]
        assert len(matching_evidence) > 0, f"Missing evidence for: {trap_col}"


    for trap_col in expected_trap_columns:
        trigger_count = traps_df[trap_col].sum()
        assert trigger_count > 0, f"Trap pattern {trap_col} was never triggered across portfolio"


def test_step_28_cannibalization_analysis():
    """Verify category product substitute cannibalization analysis."""
    cannibal_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "promotion_cannibalization_analysis.parquet"))
    assert len(cannibal_df) > 0, "No category cannibalization records generated"

    required_cols = [
        "promotion_id",
        "category_id",
        "promotional_category_revenue",
        "non_promotional_category_revenue",
        "cannibalization_revenue_share_pct",
        "is_cannibalistic"
    ]
    for col in required_cols:
        assert col in cannibal_df.columns, f"Missing cannibalization column: {col}"

    assert (cannibal_df["cannibalization_revenue_share_pct"] >= 0).all()
    assert (cannibal_df["cannibalization_revenue_share_pct"] <= 100).all()


def test_promotion_effectiveness_report():
    """Verify markdown and JSON reports contain all 8 dimensions and 5 trap patterns."""
    md_path = os.path.join(REPORTS_DIR, "promotion_effectiveness_report.md")
    json_path = os.path.join(REPORTS_DIR, "promotion_effectiveness_report.json")

    assert os.path.exists(md_path)
    assert os.path.exists(json_path)

    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    assert "Promotion Effectiveness & Trap Detection Report" in md
    assert "Multi-Dimensional Promotion Evaluation Matrix" in md
    assert "Promotion Trap Detection (Step 28 - All 5 Exact Misleading Patterns)" in md
    assert "Product Cannibalization Analysis" in md
    assert "Post-Promotion Customer Retention & Cohort Behavior" in md
    assert "Sales Up, Profit Down" in md
    assert "Margin Collapse" in md
    assert "Increases Wastage" in md
    assert "Discount Churners" in md
    assert "Cannibalization" in md

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "baseline_metrics" in data
    assert "promotion_evaluations" in data
    assert "promotion_traps" in data
    assert len(data["promotion_evaluations"]) == 12
