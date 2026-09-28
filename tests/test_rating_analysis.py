"""
Unit and Integration Tests for SRS Steps 29 & 30:
Rating and Satisfaction Analysis & Rating Anomaly Detection
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
RATINGS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "ratings", "ratings.parquet")
ORDERS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "orders", "orders.parquet")
CUBE_PATH = os.path.join(PROJECT_ROOT, "processed_data", "joined", "master_analytical_cube", "master_analytical_cube.parquet")

from python_pipeline.ratings.rating_analysis import (
    RatingSatisfactionAnalyzer,
    RatingAnomalyDetector,
    run_rating_pipeline
)


@pytest.fixture(scope="module")
def pipeline_data():
    ratings_df = pd.read_parquet(RATINGS_PATH)
    orders_df = pd.read_parquet(ORDERS_PATH)
    cube_df = pd.read_parquet(CUBE_PATH)
    return ratings_df, orders_df, cube_df


def test_rating_satisfaction_all_7_dimensions(pipeline_data):
    ratings_df, orders_df, cube_df = pipeline_data
    analyzer = RatingSatisfactionAnalyzer(ratings_df, orders_df, cube_df)


    items_df = analyzer.analyze_menu_items()
    assert len(items_df) == 150
    assert "avg_overall_rating" in items_df.columns
    assert "csat_pct" in items_df.columns
    assert "net_satisfaction_score" in items_df.columns
    assert items_df["avg_overall_rating"].min() >= 1.0
    assert items_df["avg_overall_rating"].max() <= 5.0


    loc_df = analyzer.analyze_locations()
    assert len(loc_df) == 20
    assert "satisfaction_rank" in loc_df.columns
    assert loc_df["satisfaction_rank"].min() == 1
    assert loc_df["satisfaction_rank"].max() <= 20


    profit_res = analyzer.analyze_profitability()
    assert "pearson_correlation" in profit_res
    assert "profit_tier_summary" in profit_res
    assert len(profit_res["profit_tier_summary"]) == 4


    sales_res = analyzer.analyze_sales()
    assert "sales_rating_correlation" in sales_res
    assert "quadrant_distribution" in sales_res
    assert sum(sales_res["quadrant_distribution"].values()) == 150


    repeat_res = analyzer.analyze_repeat_purchase()
    assert "customer_type_satisfaction" in repeat_res
    assert len(repeat_res["customer_type_satisfaction"]) >= 2


    time_res = analyzer.analyze_time_period()
    assert "monthly_satisfaction_trend" in time_res
    assert len(time_res["monthly_satisfaction_trend"]) >= 12
    assert "weekend_vs_weekday" in time_res


    promo_res = analyzer.analyze_promotion_status()
    assert "promoted_vs_non_promoted" in promo_res
    assert len(promo_res["promoted_vs_non_promoted"]) == 2
    assert "misleading_backlash_by_campaign" in promo_res
    assert "PROMO-008" in promo_res["misleading_backlash_by_campaign"]


def test_rating_anomaly_detection_all_5_patterns(pipeline_data):
    ratings_df, orders_df, cube_df = pipeline_data
    detector = RatingAnomalyDetector(ratings_df, orders_df, cube_df)
    anomalies = detector.detect_all_anomalies()

    assert "sudden_rating_spikes" in anomalies
    assert "sudden_rating_drops" in anomalies
    assert "excessive_identical_ratings" in anomalies
    assert "high_volume_bursts" in anomalies
    assert "inconsistent_ratings" in anomalies


    assert len(anomalies["sudden_rating_spikes"]) > 0
    assert len(anomalies["sudden_rating_drops"]) > 0
    assert len(anomalies["excessive_identical_ratings"]) > 0
    assert len(anomalies["high_volume_bursts"]) > 0
    assert len(anomalies["inconsistent_ratings"]) > 0


def test_rating_pipeline_artifacts():
    sat_summary, anomalies = run_rating_pipeline()
    assert sat_summary["menu_items_evaluated"] == 150
    assert sat_summary["locations_evaluated"] == 20

    out_dir = os.path.join(PROJECT_ROOT, "processed_data", "ratings")
    rep_dir = os.path.join(PROJECT_ROOT, "reports", "ratings")

    assert os.path.exists(os.path.join(out_dir, "rating_item_satisfaction.parquet"))
    assert os.path.exists(os.path.join(out_dir, "rating_location_satisfaction.parquet"))
    assert os.path.exists(os.path.join(out_dir, "rating_anomalies.parquet"))
    assert os.path.exists(os.path.join(rep_dir, "rating_and_satisfaction_report.md"))
    assert os.path.exists(os.path.join(rep_dir, "rating_and_satisfaction_report.json"))
