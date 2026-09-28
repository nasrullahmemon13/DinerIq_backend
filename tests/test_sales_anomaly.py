"""
Unit and Integration Tests for SRS Step 31: Sales Anomaly Detection
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
ORDERS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "orders", "orders.parquet")
CUBE_PATH = os.path.join(PROJECT_ROOT, "processed_data", "joined", "master_analytical_cube", "master_analytical_cube.parquet")
QUARANTINE_ORDERS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "quarantine", "orders_rule_02.parquet")

from python_pipeline.anomalies.sales_anomaly import (
    SalesAnomalyDetector,
    run_sales_anomaly_pipeline
)


@pytest.fixture(scope="module")
def sales_data():
    orders_df = pd.read_parquet(ORDERS_PATH)
    cube_df = pd.read_parquet(CUBE_PATH)
    quarantined_df = pd.read_parquet(QUARANTINE_ORDERS_PATH) if os.path.exists(QUARANTINE_ORDERS_PATH) else pd.DataFrame()
    return orders_df, cube_df, quarantined_df


def test_sales_anomaly_all_6_events(sales_data):
    orders_df, cube_df, quarantined_df = sales_data
    detector = SalesAnomalyDetector(orders_df, cube_df, quarantined_df)
    anomalies = detector.detect_all_anomalies()


    assert "sudden_sales_spikes" in anomalies
    assert "sudden_sales_drops" in anomalies
    assert "abnormally_high_order_values" in anomalies
    assert "unusual_discounts" in anomalies
    assert "unexpected_demand" in anomalies
    assert "duplicate_transactions" in anomalies


    assert len(anomalies["sudden_sales_spikes"]) > 0
    assert len(anomalies["sudden_sales_drops"]) > 0
    assert len(anomalies["abnormally_high_order_values"]) > 0
    assert len(anomalies["unusual_discounts"]) > 0
    assert len(anomalies["unexpected_demand"]) > 0
    assert len(anomalies["duplicate_transactions"]) >= 200


def test_whale_order_iqr_threshold(sales_data):
    orders_df, cube_df, quarantined_df = sales_data
    detector = SalesAnomalyDetector(orders_df, cube_df, quarantined_df)
    whales = detector.detect_abnormally_high_order_values(iqr_multiplier=3.0)

    q1 = orders_df["total_amount"].quantile(0.25)
    q3 = orders_df["total_amount"].quantile(0.75)
    cutoff = q3 + 3.0 * (q3 - q1)

    assert (whales["total_amount"] > cutoff).all()


def test_sales_anomaly_pipeline_execution():
    counts, anomalies = run_sales_anomaly_pipeline()
    assert counts["total_sales_anomalies_flagged"] > 500

    out_dir = os.path.join(PROJECT_ROOT, "processed_data", "anomalies")
    rep_dir = os.path.join(PROJECT_ROOT, "reports", "anomalies")

    assert os.path.exists(os.path.join(out_dir, "sales_anomalies.parquet"))
    assert os.path.exists(os.path.join(out_dir, "sales_anomaly_summary.parquet"))
    assert os.path.exists(os.path.join(rep_dir, "sales_anomaly_report.md"))
    assert os.path.exists(os.path.join(rep_dir, "sales_anomaly_report.json"))
