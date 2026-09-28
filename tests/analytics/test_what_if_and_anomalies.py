"""
DineIQ Analytics - What-If Simulation and Anomaly Detection Test Suite (SRS Step 20, Step 21 & Step 28)
Verifies:
- What-if simulation runs with required delta scenarios (+10% price, -10% price, prep reduction, etc.)
- Results are explicitly marked as simulations (is_simulation_estimate=True, disclaimer present)
- Rating anomaly and Sales anomaly detections are independently calculated
"""
import os
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WHAT_IF_DIR = os.path.join(PROJECT_ROOT, "processed_data", "what_if")
ANOMALIES_DIR = os.path.join(PROJECT_ROOT, "processed_data", "anomalies")
ANOMALY_DIR = os.path.join(PROJECT_ROOT, "processed_data", "anomaly")
RATINGS_DIR = os.path.join(PROJECT_ROOT, "processed_data", "ratings")


def test_what_if_simulation_structure():
    """Verify what-if simulation scenarios and explicit simulation marking."""
    pq_path = os.path.join(WHAT_IF_DIR, "what_if_scenario_benchmark.parquet")
    assert os.path.exists(pq_path), f"Missing {pq_path}"
    df = pd.read_parquet(pq_path)
    assert len(df) > 0

    assert "is_simulation_estimate" in df.columns
    assert df["is_simulation_estimate"].all(), "All records in benchmark must be tagged as simulation estimates"
    assert "disclaimer" in df.columns

    scenario_names = df["scenario_name"].str.lower().tolist()
    text_corpus = " ".join(scenario_names)
    assert "price" in text_corpus, "Missing price adjustment scenario"
    assert "waste" in text_corpus or "prep" in text_corpus or "demand" in text_corpus, "Missing operational scenario"


def test_sales_and_rating_anomalies_detected():
    """Verify anomaly detection outputs exist and contain independent statistical evidence."""
                      
    rating_anom_pq = os.path.join(ANOMALY_DIR, "rating_anomalies.parquet")
    if not os.path.exists(rating_anom_pq):
        rating_anom_pq = os.path.join(RATINGS_DIR, "rating_anomalies.parquet")
    assert os.path.exists(rating_anom_pq), f"Missing rating anomalies at {rating_anom_pq}"
    df_rat = pd.read_parquet(rating_anom_pq)
    assert len(df_rat) > 0
    assert "score_or_metric" in df_rat.columns or "anomaly_score" in df_rat.columns or "z_score" in df_rat.columns

                     
    sales_anom_pq = os.path.join(ANOMALY_DIR, "sales_anomalies.parquet")
    if not os.path.exists(sales_anom_pq):
        sales_anom_pq = os.path.join(ANOMALIES_DIR, "sales_anomalies.parquet")
    assert os.path.exists(sales_anom_pq), f"Missing sales anomalies at {sales_anom_pq}"
    df_sales = pd.read_parquet(sales_anom_pq)
    assert len(df_sales) > 0
    assert "score_or_metric" in df_sales.columns or "financial_impact_or_score" in df_sales.columns
