"""
Unit and Integration Tests for SRS Step 32: Slow-Moving Dish Detection
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
CUBE_PATH = os.path.join(PROJECT_ROOT, "processed_data", "joined", "master_analytical_cube", "master_analytical_cube.parquet")
WASTAGE_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "wastage", "wastage.parquet")

from python_pipeline.menu.slow_moving_dishes import (
    SlowMovingDishDetector,
    run_slow_moving_pipeline
)


@pytest.fixture(scope="module")
def menu_data():
    cube_df = pd.read_parquet(CUBE_PATH)
    wastage_df = pd.read_parquet(WASTAGE_PATH)
    return cube_df, wastage_df


def test_slow_moving_all_7_dimensions(menu_data):
    cube_df, wastage_df = menu_data
    detector = SlowMovingDishDetector(cube_df, wastage_df)
    df_dims = detector.compute_all_7_dimensions()

    assert len(df_dims) == 150


    assert "total_quantity_sold" in df_dims.columns
    assert (df_dims["total_quantity_sold"] > 0).all()


    assert "unique_order_count" in df_dims.columns
    assert (df_dims["unique_order_count"] > 0).all()


    assert "mean_gap_days" in df_dims.columns
    assert "max_gap_days" in df_dims.columns
    assert (df_dims["mean_gap_days"] >= 0).all()


    assert "repeat_purchase_pct" in df_dims.columns
    assert (df_dims["repeat_purchase_pct"] >= 0).all()
    assert (df_dims["repeat_purchase_pct"] <= 100).all()


    assert "wasted_quantity" in df_dims.columns
    assert "wastage_percentage" in df_dims.columns
    assert (df_dims["wastage_percentage"] >= 0).all()


    assert "contribution_margin_pct" in df_dims.columns
    assert "profit_per_unit" in df_dims.columns


    assert "normalized_trend_slope" in df_dims.columns
    assert "q4_vs_q1_growth_pct" in df_dims.columns


def test_slow_moving_scorecard_and_classes(menu_data):
    cube_df, wastage_df = menu_data
    detector = SlowMovingDishDetector(cube_df, wastage_df)
    scorecard = detector.compute_slow_moving_scorecard()

    assert len(scorecard) == 150
    assert "slow_moving_index" in scorecard.columns
    assert "movement_class" in scorecard.columns
    assert "recommended_action" in scorecard.columns


    assert scorecard["slow_moving_index"].min() >= 0.0
    assert scorecard["slow_moving_index"].max() <= 1.0


    classes = set(scorecard["movement_class"].unique())
    assert "Critical Slow-Moving" in classes
    assert "Moderate Slow-Moving" in classes
    assert "Active Mover" in classes


def test_exact_7_srs_dimensions_combination(menu_data):
    cube_df, wastage_df = menu_data
    detector = SlowMovingDishDetector(cube_df, wastage_df)
    scorecard = detector.compute_slow_moving_scorecard()


    srs_metrics = [
        "low_sales_volume_val",
        "low_purchase_frequency_val",
        "long_gaps_between_purchases_val",
        "low_repeat_purchase_val",
        "high_wastage_val",
        "weak_profitability_val",
        "poor_trend_val"
    ]
    for m in srs_metrics:
        assert m in scorecard.columns


    srs_scores = [
        "low_sales_volume_score",
        "low_purchase_frequency_score",
        "long_gaps_between_purchases_score",
        "low_repeat_purchase_score",
        "high_wastage_score",
        "weak_profitability_score",
        "poor_trend_score"
    ]
    for s in srs_scores:
        assert s in scorecard.columns
        assert (scorecard[s] >= 0.0).all() and (scorecard[s] <= 1.0).all()

    srs_flags = [
        "flag_low_sales_volume",
        "flag_low_purchase_frequency",
        "flag_long_gaps_between_purchases",
        "flag_low_repeat_purchase",
        "flag_high_wastage",
        "flag_weak_profitability",
        "flag_poor_trend"
    ]
    for f in srs_flags:
        assert f in scorecard.columns
        assert scorecard[f].dtype == bool


    assert "srs_dimensions_triggered_count" in scorecard.columns
    assert (scorecard["srs_dimensions_triggered_count"] >= 0).all()
    assert (scorecard["srs_dimensions_triggered_count"] <= 7).all()


    critical = scorecard[scorecard["movement_class"] == "Critical Slow-Moving"]
    assert len(critical) > 0
    assert (critical["slow_moving_index"] >= 0.50).all()


def test_slow_moving_pipeline_artifacts():
    scorecard_df, summary_stats = run_slow_moving_pipeline()

    assert summary_stats["total_dishes_evaluated"] == 150
    assert summary_stats["critical_slow_moving_count"] > 0
    assert summary_stats["total_wastage_loss_slow_moving"] > 0

    out_dir = os.path.join(PROJECT_ROOT, "processed_data", "slow_moving")
    rep_dir = os.path.join(PROJECT_ROOT, "reports", "slow_moving")

    assert os.path.exists(os.path.join(out_dir, "slow_moving_dishes.parquet"))
    assert os.path.exists(os.path.join(out_dir, "all_menu_items_movement_scorecard.parquet"))
    assert os.path.exists(os.path.join(rep_dir, "slow_moving_dishes_report.md"))
    assert os.path.exists(os.path.join(rep_dir, "slow_moving_dishes_report.json"))

