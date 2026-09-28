"""
Unit and Integration Tests for SRS Step 35: Ordering Channel Analysis
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

from python_pipeline.channels.channel_analysis import (
    OrderingChannelAnalyzer,
    run_channel_pipeline
)

ORDERS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "orders", "orders.parquet")
CUBE_PATH = os.path.join(PROJECT_ROOT, "processed_data", "joined", "master_analytical_cube", "master_analytical_cube.parquet")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "channels")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "channels")


@pytest.fixture(scope="module")
def channel_data():
    orders_df = pd.read_parquet(ORDERS_PATH)
    cube_df = pd.read_parquet(CUBE_PATH)
    return orders_df, cube_df


def test_all_5_channels_present(channel_data):
    orders_df, cube_df = channel_data
    analyzer = OrderingChannelAnalyzer(orders_df, cube_df)
    matrix = analyzer.generate_consolidated_channel_matrix()

    channels = set(matrix["ordering_channel"].unique())
    expected = {
        "Dine-in",
        "Takeaway",
        "Restaurant Website or App",
        "Third-party delivery platforms",
        "Other supported channels (Drive-thru)"
    }
    assert channels == expected
    assert len(matrix) == 5


def test_all_7_comparative_dimensions(channel_data):
    orders_df, cube_df = channel_data
    analyzer = OrderingChannelAnalyzer(orders_df, cube_df)
    matrix = analyzer.generate_consolidated_channel_matrix()
    menu_df = analyzer.compare_menu_preferences()


    assert "avg_units_per_order" in matrix.columns
    assert "avg_distinct_items_per_order" in matrix.columns
    assert (matrix["avg_units_per_order"] > 0).all()


    assert "mean_aov" in matrix.columns
    assert (matrix["mean_aov"] > 0).all()


    assert "category_name" in menu_df.columns
    assert "channel_unit_share_pct" in menu_df.columns
    assert len(menu_df) > 0


    assert "discount_penetration_pct" in matrix.columns
    assert "avg_discount_when_applied" in matrix.columns
    assert (matrix["discount_penetration_pct"] >= 0).all()


    assert "promo_order_penetration_pct" in matrix.columns
    assert "promoted_revenue_share_pct" in matrix.columns
    assert (matrix["promo_order_penetration_pct"] > 0).all()


    assert "peak_ordering_hour" in matrix.columns
    assert "peak_day_of_week" in matrix.columns
    assert "weekend_share_pct" in matrix.columns


    assert "gross_profit" in matrix.columns
    assert "contribution_margin_pct" in matrix.columns
    assert "profit_per_order" in matrix.columns
    assert (matrix["contribution_margin_pct"] > 0).all()
    assert (matrix["profit_per_order"] > 0).all()


def test_channel_pipeline_artifacts():
    matrix_df, menu_df, stats = run_channel_pipeline()

    assert stats["channels_evaluated"] == 5
    assert stats["total_revenue_all_channels"] > 0

    assert os.path.exists(os.path.join(OUTPUT_DIR, "ordering_channel_comparison.parquet"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "channel_menu_preferences.parquet"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "channel_hourly_patterns.parquet"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "ordering_channel_report.md"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "ordering_channel_report.json"))
