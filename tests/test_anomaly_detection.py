"""
Unit and Integration Tests for DineIQ Unified Anomaly Detection Pipeline (SRS Steps 29-31)
Target file: python_pipeline/anomaly/anomaly_detection.py
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

from python_pipeline.anomaly.anomaly_detection import (
    RatingSatisfactionAnalyzer,
    RatingAnomalyDetector,
    SalesAnomalyDetector,
    DineIQAnomalyPipeline,
    run_anomaly_detection_pipeline
)

RATINGS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "ratings", "ratings.parquet")
ORDERS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "orders", "orders.parquet")
CUBE_PATH = os.path.join(PROJECT_ROOT, "processed_data", "joined", "master_analytical_cube", "master_analytical_cube.parquet")
QUARANTINE_ORDERS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "quarantine", "orders_rule_02.parquet")


@pytest.fixture(scope="module")
def pipeline_data():
    ratings_df = pd.read_parquet(RATINGS_PATH)
    orders_df = pd.read_parquet(ORDERS_PATH)
    cube_df = pd.read_parquet(CUBE_PATH)
    quarantined_df = pd.read_parquet(QUARANTINE_ORDERS_PATH) if os.path.exists(QUARANTINE_ORDERS_PATH) else pd.DataFrame()
    return ratings_df, orders_df, cube_df, quarantined_df


                                                                               
                                                 
                                                                               

def test_step_29_all_7_dimensions(pipeline_data):
    ratings_df, orders_df, cube_df, _ = pipeline_data
    analyzer = RatingSatisfactionAnalyzer(ratings_df, orders_df, cube_df)
    results = analyzer.run_full_satisfaction_analysis()

                   
    menu_items = results["menu_items"]
    assert len(menu_items) == 150
    assert "avg_overall_rating" in menu_items.columns
    assert "csat_pct" in menu_items.columns
    assert "net_satisfaction_score" in menu_items.columns

                             
    locations = results["locations"]
    assert len(locations) == 20
    assert "satisfaction_rank" in locations.columns
    assert "avg_overall_rating" in locations.columns

                      
    prof = results["profitability"]
    assert "pearson_correlation" in prof
    assert "profit_tier_summary" in prof

              
    sales = results["sales"]
    assert "sales_rating_correlation" in sales
    assert "quadrant_distribution" in sales

                        
    repeat = results["repeat_purchase"]
    assert "repeat_rate_vs_rating_correlation" in repeat
    assert "customer_type_satisfaction" in repeat

                    
    time_p = results["time_period"]
    assert "monthly_satisfaction_trend" in time_p
    assert "day_of_week_satisfaction" in time_p
    assert "weekend_vs_weekday" in time_p

                         
    promo = results["promotion_status"]
    assert "promoted_vs_non_promoted" in promo
    assert "campaign_breakdown" in promo
    assert "misleading_backlash_by_campaign" in promo


                                                                               
                                         
                                                                               

def test_step_30_all_5_rating_anomalies(pipeline_data):
    ratings_df, orders_df, cube_df, _ = pipeline_data
    detector = RatingAnomalyDetector(ratings_df, orders_df, cube_df)
    anomalies = detector.detect_all_anomalies()

                             
    assert "sudden_rating_spikes" in anomalies
    assert len(anomalies["sudden_rating_spikes"]) > 0

                            
    assert "sudden_rating_drops" in anomalies
    assert len(anomalies["sudden_rating_drops"]) > 0

                                    
    assert "excessive_identical_ratings" in anomalies
    assert len(anomalies["excessive_identical_ratings"]) > 0

                                                 
    assert "high_volume_bursts" in anomalies
    assert len(anomalies["high_volume_bursts"]) > 0

                                                      
    assert "inconsistent_ratings" in anomalies
    assert len(anomalies["inconsistent_ratings"]) > 0


                                                                               
                                        
                                                                               

def test_step_31_all_6_sales_anomalies(pipeline_data):
    _, orders_df, cube_df, quarantined_df = pipeline_data
    detector = SalesAnomalyDetector(orders_df, cube_df, quarantined_df)
    anomalies = detector.detect_all_anomalies()

                            
    assert "sudden_sales_spikes" in anomalies
    assert len(anomalies["sudden_sales_spikes"]) > 0

                           
    assert "sudden_sales_drops" in anomalies
    assert len(anomalies["sudden_sales_drops"]) > 0

                                     
    assert "abnormally_high_order_values" in anomalies
    assert len(anomalies["abnormally_high_order_values"]) > 0

                          
    assert "unusual_discounts" in anomalies
    assert len(anomalies["unusual_discounts"]) > 0

                          
    assert "unexpected_demand" in anomalies
    assert len(anomalies["unexpected_demand"]) > 0

                               
    assert "duplicate_transactions" in anomalies
    assert len(anomalies["duplicate_transactions"]) >= 200


                                                                               
                                     
                                                                               

def test_anomaly_pipeline_execution_and_artifacts():
    result = run_anomaly_detection_pipeline()

    counts = result["summary_counts"]
    assert counts["ratings_evaluated"] == 100300
    assert counts["orders_evaluated"] == 90471
    assert counts["total_rating_anomalies"] > 0
    assert counts["total_sales_anomalies"] > 0
    assert counts["total_anomalies_flagged"] > 0

    out_dir = os.path.join(PROJECT_ROOT, "processed_data", "anomaly")
    rep_dir = os.path.join(PROJECT_ROOT, "reports", "anomaly")

    assert os.path.exists(os.path.join(out_dir, "rating_anomalies.parquet"))
    assert os.path.exists(os.path.join(out_dir, "sales_anomalies.parquet"))
    assert os.path.exists(os.path.join(out_dir, "anomaly_master_summary.parquet"))
    assert os.path.exists(os.path.join(out_dir, "rating_item_satisfaction.parquet"))
    assert os.path.exists(os.path.join(out_dir, "rating_location_satisfaction.parquet"))
    assert os.path.exists(os.path.join(rep_dir, "anomaly_detection_report.md"))
    assert os.path.exists(os.path.join(rep_dir, "anomaly_detection_report.json"))
