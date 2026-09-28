"""
Unit and Integration Tests for SRS Steps 33 & 34:
Multi-Location Intelligence & Location-Specific Menu Performance
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

from python_pipeline.locations.location_intelligence import (
    MultiLocationComparator,
    LocationMenuClassifier,
    run_location_pipeline
)

ORDERS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "orders", "orders.parquet")
CUBE_PATH = os.path.join(PROJECT_ROOT, "processed_data", "joined", "master_analytical_cube", "master_analytical_cube.parquet")
RATINGS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "ratings", "ratings.parquet")
WASTAGE_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "wastage", "wastage.parquet")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "locations")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "locations")


@pytest.fixture(scope="module")
def pipeline_data():
    orders_df = pd.read_parquet(ORDERS_PATH)
    cube_df = pd.read_parquet(CUBE_PATH)
    ratings_df = pd.read_parquet(RATINGS_PATH)
    wastage_df = pd.read_parquet(WASTAGE_PATH)
    return orders_df, cube_df, ratings_df, wastage_df


def test_location_comparison_all_9_dimensions(pipeline_data):
    orders_df, cube_df, ratings_df, wastage_df = pipeline_data
    comparator = MultiLocationComparator(orders_df, cube_df, ratings_df, wastage_df)
    loc_matrix = comparator.compute_all_9_dimensions()

    assert len(loc_matrix) == 20

                          
    assert "total_revenue" in loc_matrix.columns
    assert "revenue_share_pct" in loc_matrix.columns
    assert (loc_matrix["total_revenue"] > 0).all()

                                
    assert "gross_profit" in loc_matrix.columns
    assert "contribution_margin_pct" in loc_matrix.columns
    assert (loc_matrix["contribution_margin_pct"] > 0).all()

                                            
    assert "average_order_value" in loc_matrix.columns
    assert (loc_matrix["average_order_value"] > 0).all()

                                 
    assert "unique_customer_count" in loc_matrix.columns
    assert (loc_matrix["unique_customer_count"] > 0).all()

                                  
    assert "repeat_purchase_rate" in loc_matrix.columns
    assert (loc_matrix["repeat_purchase_rate"] > 0).all()

                          
    assert "quantity_wasted" in loc_matrix.columns
    assert "total_wastage_loss_amount" in loc_matrix.columns
    assert "wastage_rate_pct" in loc_matrix.columns

                          
    assert "avg_overall_rating" in loc_matrix.columns
    assert "csat_pct" in loc_matrix.columns
    assert (loc_matrix["avg_overall_rating"] >= 1.0).all()
    assert (loc_matrix["avg_overall_rating"] <= 5.0).all()

                                          
    assert "promoted_order_count" in loc_matrix.columns
    assert "promoted_revenue" in loc_matrix.columns
    assert "promoted_order_share_pct" in loc_matrix.columns

              
    assert "revenue_rank" in loc_matrix.columns
    assert "satisfaction_rank" in loc_matrix.columns


def test_location_specific_menu_performance_4_classes(pipeline_data):
    _, cube_df, ratings_df, wastage_df = pipeline_data
    classifier = LocationMenuClassifier(cube_df, wastage_df, ratings_df)
    loc_menu_df = classifier.classify_all_location_items()

                                                                 
    assert len(loc_menu_df) == 3000

                                             
    classes = set(loc_menu_df["location_menu_classification"].unique())
    assert classes == {"Profit Driver", "Volume Driver", "Hidden Opportunity", "Low Performer"}

                
    assert (loc_menu_df["quantity_sold"] >= 0).all()
    assert (loc_menu_df["revenue"] >= 0).all()
    assert (loc_menu_df["contribution_margin_pct"].notna()).all()


def test_dish_performing_differently_across_locations(pipeline_data):
    """Verifies SRS difficult case: same dish performing differently across locations."""
    _, cube_df, ratings_df, wastage_df = pipeline_data
    classifier = LocationMenuClassifier(cube_df, wastage_df, ratings_df)
    loc_menu_df = classifier.classify_all_location_items()
    divergent_dishes = classifier.identify_cross_location_divergent_dishes(loc_menu_df)

                                                     
    assert len(divergent_dishes) >= 50
    assert (divergent_dishes["distinct_class_count"] > 1).all()

                                                                                    
    item_11 = loc_menu_df[loc_menu_df["item_id"] == "ITEM-011"]
    assert item_11["location_menu_classification"].nunique() > 1


def test_location_pipeline_artifacts():
    loc_matrix, loc_menu_df, divergent_dishes, summary_stats = run_location_pipeline()

    assert summary_stats["locations_compared"] == 20
    assert summary_stats["total_location_menu_pairs"] == 3000
    assert summary_stats["dishes_with_multiple_classes_across_locations"] >= 50

                           
    assert os.path.exists(os.path.join(OUTPUT_DIR, "location_comparison_matrix.parquet"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "location_menu_performance.parquet"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "cross_location_divergent_dishes.parquet"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "multi_location_intelligence_report.md"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "multi_location_intelligence_report.json"))
