"""
DineIQ Analytics - Customer Segmentation & RFM Tests (SRS Steps 15 & 16)
Verifies:
- Step 16: RFM metrics (Recency, Frequency, Monetary Value) and quintile scoring for all 50,000 customers
- Step 15: Exact 10 factors present and non-null
- Step 15: Exact 6 SRS segments represented and populated
- Business logic invariants (monetary spend, recency, and promo sensitivity across segments)
- Parquet, CSV, and report persistence
"""
import os
import json
import pytest
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEG_DIR = os.path.join(PROJECT_ROOT, "processed_data", "customer_segmentation")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "customer_segmentation")

@pytest.fixture(scope="module")
def segments_df():
    parquet_path = os.path.join(SEG_DIR, "customer_segments.parquet")
    assert os.path.exists(parquet_path), f"Missing customer_segments.parquet at {parquet_path}"
    return pd.read_parquet(parquet_path)

@pytest.fixture(scope="module")
def rfm_df():
    parquet_path = os.path.join(SEG_DIR, "rfm_analysis.parquet")
    assert os.path.exists(parquet_path), f"Missing rfm_analysis.parquet at {parquet_path}"
    return pd.read_parquet(parquet_path)

@pytest.fixture(scope="module")
def report_json():
    json_path = os.path.join(REPORTS_DIR, "customer_segmentation_report.json")
    assert os.path.exists(json_path), f"Missing report json at {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_step16_rfm_analysis_completeness(rfm_df):
    """Step 16: Verify Recency, Frequency, and Monetary metrics and quintiles."""
    assert len(rfm_df) == 50000, "RFM analysis must cover all 50,000 customers"
    
    rfm_cols = ["customer_recency", "customer_frequency", "customer_monetary_value", "r_score", "f_score", "m_score", "rfm_cell"]
    for col in rfm_cols:
        assert col in rfm_df.columns, f"Missing RFM column: '{col}'"
        assert rfm_df[col].isnull().sum() == 0, f"Null values in RFM column: '{col}'"
    
                           
    for score in ["r_score", "f_score", "m_score"]:
        assert rfm_df[score].min() >= 1 and rfm_df[score].max() <= 5

def test_step15_exact_10_factors_present(segments_df):
    """Step 15: Verify the EXACT 10 factors SRS lists are present and non-null."""
    the_10_factors = [
        "recency",
        "frequency",
        "monetary_value",
        "average_order_value",
        "visit_frequency",
        "favorite_menu_categories",
        "promotion_sensitivity",
        "ordering_channel",
        "time_of_day_preference",
        "repeat_behavior"
    ]
    for factor in the_10_factors:
        assert factor in segments_df.columns, f"Missing Step 15 factor: '{factor}'"
        assert segments_df[factor].isnull().sum() == 0, f"Null values found in factor: '{factor}'"

def test_step15_exact_6_srs_segments(segments_df, report_json):
    """Step 15: Verify all 6 SRS suggested segments are present and populated."""
    expected_segments = {
        "High-Value Loyal Customers",
        "Frequent Customers",
        "Promotion-Driven Customers",
        "At-Risk Customers",
        "New Customers",
        "Occasional Customers"
    }
    actual_segments = set(segments_df["customer_segment"].unique())
    assert actual_segments == expected_segments, f"Expected {expected_segments}, got {actual_segments}"

                                            
    for seg in expected_segments:
        count = (segments_df["customer_segment"] == seg).sum()
        assert count > 0, f"Segment '{seg}' has 0 customers"

def test_segment_business_logic_invariants(segments_df):
    """Verify logical consistency across customer segments."""
    loyal = segments_df[segments_df["customer_segment"] == "High-Value Loyal Customers"]
    occasional = segments_df[segments_df["customer_segment"] == "Occasional Customers"]
    at_risk = segments_df[segments_df["customer_segment"] == "At-Risk Customers"]
    new_cust = segments_df[segments_df["customer_segment"] == "New Customers"]
    promo_driven = segments_df[segments_df["customer_segment"] == "Promotion-Driven Customers"]

                                               
    assert loyal["monetary_value"].mean() > occasional["monetary_value"].mean()

                                            
    assert at_risk["recency"].mean() > new_cust["recency"].mean()

                                                                      
    assert promo_driven["promotion_sensitivity"].mean() > loyal["promotion_sensitivity"].mean()

def test_persistence_artifacts_exist():
    """Verify Parquet, CSV, and markdown reports exist."""
    assert os.path.exists(os.path.join(SEG_DIR, "customer_segments.parquet"))
    assert os.path.exists(os.path.join(SEG_DIR, "customer_segments.csv"))
    assert os.path.exists(os.path.join(SEG_DIR, "rfm_analysis.parquet"))
    assert os.path.exists(os.path.join(SEG_DIR, "rfm_analysis.csv"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "customer_segmentation_report.md"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "customer_segmentation_report.json"))
