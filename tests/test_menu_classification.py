"""
DineIQ Analytics - Menu Performance Classification Tests (SRS Steps 9, 10, 11)
Verifies:
- Step 9: Menu profitability analysis covers ALL 10 SRS dimensions
- Step 10: Exact 4 SRS categories (Profit Driver, Volume Driver, Hidden Opportunity, Low Performer)
- Step 10 constraint: High volume alone does not make an item successful (loss-making items never Profit Drivers)
- Step 11: All 10 tricky performance scenarios identified and handled
"""
import os
import json
import pytest
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLASSIFICATION_DIR = os.path.join(PROJECT_ROOT, "processed_data", "menu_classification")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "menu_classification")

@pytest.fixture(scope="module")
def menu_class_df():
    parquet_path = os.path.join(CLASSIFICATION_DIR, "menu_classification.parquet")
    assert os.path.exists(parquet_path), f"Missing menu_classification.parquet at {parquet_path}"
    return pd.read_parquet(parquet_path)

@pytest.fixture(scope="module")
def report_metadata():
    json_path = os.path.join(REPORTS_DIR, "menu_classification_report.json")
    assert os.path.exists(json_path), f"Missing menu_classification_report.json at {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_step9_all_10_profitability_dimensions(menu_class_df):
    """Step 9: Verify ALL 10 dimensions SRS lists are present and non-null."""
    srs_10_dimensions = [
        "quantity_sold",
        "revenue",
        "cost",
        "contribution_margin",
        "profit_percentage",
        "customer_rating",
        "repeat_purchase_rate",
        "wastage_percentage",
        "promotion_dependency",
        "sales_trend"
    ]
    for dim in srs_10_dimensions:
        assert dim in menu_class_df.columns, f"Missing Step 9 dimension '{dim}'"
        assert menu_class_df[dim].isnull().sum() == 0, f"Null values in dimension '{dim}'"

def test_step10_exact_4_categories(menu_class_df, report_metadata):
    """Step 10: Verify every item is classified into exactly the 4 SRS categories."""
    expected_categories = {"Profit Driver", "Volume Driver", "Hidden Opportunity", "Low Performer"}
    actual_categories = set(menu_class_df["menu_classification"].unique())
    
    assert actual_categories == expected_categories, f"Expected {expected_categories}, got {actual_categories}"
    assert len(menu_class_df) == 150, "All 150 menu items must be classified"
    
                                               
    for cat in expected_categories:
        count = (menu_class_df["menu_classification"] == cat).sum()
        assert count > 0, f"Category '{cat}' has 0 items"

def test_step10_volume_alone_not_successful_rule(menu_class_df):
    """Step 10 Rule: 'High sales volume alone must not make a menu item successful'."""
                                                                                               
    unhealthy_items = menu_class_df[
        (menu_class_df["contribution_margin"] <= 0) | (menu_class_df["wastage_percentage"] >= 20.0)
    ]
    for _, row in unhealthy_items.iterrows():
        assert row["menu_classification"] != "Profit Driver", (
            f"Item {row['item_id']} ({row['item_name']}) has negative margin or high wastage "
            f"and cannot be classified as a Profit Driver"
        )

def test_step11_all_10_tricky_scenarios_handled(report_metadata, menu_class_df):
    """Step 11: Verify EXACTLY the 10 tricky scenarios are tracked and identified."""
    srs_10_tricky_scenarios = [
        "High-Selling Loss-Making Dish",
        "Highly Profitable but Rarely Purchased Dish",
        "Popular Dish with Excessive Wastage",
        "Highly Rated Dish with Poor Profitability",
        "Low-Rated Dish with High Sales",
        "Promotion-Dependent Dish",
        "Dish Performing Differently Across Locations",
        "Weekend-Only Performer",
        "Seasonal Item",
        "New Item with Insufficient History"
    ]
    
    tricky_counts = report_metadata["tricky_scenario_counts"]
    for sc in srs_10_tricky_scenarios:
        assert sc in tricky_counts, f"Missing Step 11 tricky scenario: '{sc}'"
        assert tricky_counts[sc] > 0, f"Scenario '{sc}' identified 0 items"

def test_reports_and_csv_persistence():
    """Verify markdown report, json report, and CSV files exist."""
    assert os.path.exists(os.path.join(REPORTS_DIR, "menu_classification_report.md"))
    assert os.path.exists(os.path.join(CLASSIFICATION_DIR, "menu_classification.csv"))
