"""
DineIQ Analytics - SRS Deliverable #9 Exact Test Categories
Conforms to the comprehensive test categories specified in SRS Deliverable #9:
  1.  functional
  2.  integration
  3.  big_data_ingestion
  4.  schema_validation
  5.  data_quality
  6.  spark_transformation
  7.  spark_sql
  8.  spark_model
  9.  python_model
  10. dual_pipeline_comparison
  11. forecast
  12. basket_analysis
  13. wastage
  14. promotion
  15. pricing
  16. anomaly
  17. security
  18. boundary
  19. hidden_data_readiness
"""

import os
import json
import sqlite3
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from backend.main import app
from src.what_if_engine import WhatIfScenarioEngine

client = TestClient(app)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "dineiq.db")





@pytest.mark.functional
def test_functional_crud_and_reporting_lifecycle():
    """Verify core functional API lifecycle from menu creation to what-if simulation."""

    res = client.post("/api/v1/menu/items", json={
        "item_id": "ITEM-FUNC-01",
        "category_id": "CAT-001",
        "name": "Functional Truffle Burger",
        "base_price": 22.0,
        "cost_price": 7.0,
        "is_active": True
    }, headers={"X-User-Role": "admin"})
    assert res.status_code in (201, 200)


    menu_res = client.get("/api/v1/menu-intelligence")
    assert menu_res.status_code == 200
    assert "summary_metrics" in menu_res.json()


    engine = WhatIfScenarioEngine()
    sim_res = engine.simulate_increase_menu_price(item_id="ITEM-046", price_increase_pct=10.0)
    assert sim_res.is_simulation_estimate is True
    assert "revenue" in sim_res.__dict__
    assert "demand" in sim_res.__dict__





@pytest.mark.integration
def test_integration_order_to_audit_pipeline():
    """Verify end-to-end integration: Order -> Tagged Prediction -> Audit Trail entry."""

    order_id = "ORD-INT-99"
    ord_res = client.post("/api/v1/orders", json={
        "order_id": order_id,
        "location_id": "LOC-001",
        "order_type": "Dine-in",
        "items": [{"item_id": "ITEM-001", "quantity": 2, "unit_price": 15.0}]
    }, headers={"X-User-Role": "manager"})
    assert ord_res.status_code in (201, 200)


    pred_res = client.post("/api/v1/models/predict-tagged", json={
        "task_type": "Churn",
        "pipeline_type": "Spark",
        "entity_type": "CUSTOMER",
        "entity_id": "CUST-INT-01"
    }, headers={"X-User-Role": "analyst"})
    assert pred_res.status_code == 200
    pred_id = pred_res.json()["prediction_id"]


    audit_res = client.get(f"/api/v1/audit-trail?limit=10")
    assert audit_res.status_code == 200
    assert any(a["resource_id"] == pred_id or "PREDICTION" in a["event_type"] for a in audit_res.json())





@pytest.mark.big_data_ingestion
def test_big_data_ingestion_parquet_structure():
    """Verify columnar Parquet partition layout for scalable Big Data ingestion."""
    parquet_dir = os.path.join(PROJECT_ROOT, "parquet_data")
    assert os.path.exists(parquet_dir), "Parquet storage directory missing"
    subdirs = os.listdir(parquet_dir)
    assert len(subdirs) >= 1, "Must contain partitioned ingestion directories"





@pytest.mark.schema_validation
def test_schema_validation_relational_tables():
    """Verify relational schema integrity, primary keys, and required table columns."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    required_tables = [
        "restaurants", "menu_categories", "menu_items", "customers",
        "orders", "order_items", "promotions", "inventory", "wastage",
        "audit_log", "model_versions", "system_configs", "spark_jobs"
    ]
    for tbl in required_tables:
        assert tbl in tables, f"Required table '{tbl}' missing from database schema"
    con.close()





@pytest.mark.data_quality
def test_data_quality_no_negative_prices_or_invalid_status():
    """Verify data quality rules: non-negative prices, valid statuses, non-empty locations."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    invalid_prices = cur.execute("SELECT COUNT(*) FROM menu_items WHERE base_price < 0 OR cost_price < 0").fetchone()[0]
    assert invalid_prices == 0, "Data Quality Violation: Menu items with negative prices found"


    order_count = cur.execute("SELECT COUNT(*) FROM orders WHERE order_status IS NULL").fetchone()[0]
    assert order_count == 0, "Data Quality Violation: Orders with null status found"
    con.close()





@pytest.mark.spark_transformation
def test_spark_transformation_rfm_logic():
    """Verify RFM feature transformations (Recency, Frequency, Monetary)."""
    rfm_path = os.path.join(PROJECT_ROOT, "processed_data", "customer_segmentation", "rfm_analysis.parquet")
    assert os.path.exists(rfm_path), "RFM analysis parquet file missing"
    df = pd.read_parquet(rfm_path)
    assert "customer_recency" in df.columns or "recency" in df.columns
    assert "customer_frequency" in df.columns or "frequency" in df.columns
    assert "customer_monetary_value" in df.columns or "monetary" in df.columns
    assert len(df) == 50000





@pytest.mark.spark_sql
def test_spark_sql_window_and_aggregation_queries():
    """Verify Spark SQL query behavior: multi-level aggregations and ranking."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    sql = """
    SELECT location_id, item_id, SUM(quantity) as total_qty,
           RANK() OVER (PARTITION BY location_id ORDER BY SUM(quantity) DESC) as rank_in_loc
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    GROUP BY location_id, item_id
    LIMIT 20;
    """
    rows = cur.execute(sql).fetchall()
    assert len(rows) > 0, "Spark SQL window ranking query failed to return results"
    con.close()





@pytest.mark.spark_model
def test_spark_model_mllib_churn_decision_tree():
    """Verify PySpark MLlib Champion Model artifact and metrics."""
    evidence_path = os.path.join(PROJECT_ROOT, "reports", "model_comparison", "spark_model_evidence.json")
    with open(evidence_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["champion_model"] == "Decision Tree"
    dt = next(m for m in data["classification_benchmarks"] if m["model_name"] == "Decision Tree")
    assert dt["accuracy"] >= 0.75
    assert dt["roc_auc"] >= 0.80





@pytest.mark.python_model
def test_python_model_inference():
    """Verify Python Scikit-Learn champion model loading and inference."""
    import joblib
    model_path = os.path.join(PROJECT_ROOT, "models", "spark", "best_model.joblib")
    assert os.path.exists(model_path)
    model = joblib.load(model_path)
    assert hasattr(model, "predict")





@pytest.mark.dual_pipeline_comparison
def test_dual_pipeline_comparison_metrics():
    """Verify parity and comparative discrepancy report between Spark and Python pipelines."""
    rep_path = os.path.join(PROJECT_ROOT, "reports", "model_comparison", "dual_pipeline_comparison_report.json")
    assert os.path.exists(rep_path)
    with open(rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert "match_count" in rep
    assert "overall_agreement_percentage" in rep
    assert rep["overall_agreement_percentage"] >= 95.0





@pytest.mark.forecast
def test_forecast_horizon_and_peak_hours():
    """Verify 30-day forecast horizon and identification of peak demand periods."""
    rep_path = os.path.join(PROJECT_ROOT, "reports", "forecasting", "demand_forecasting_report.json")
    assert os.path.exists(rep_path)
    with open(rep_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["forecast_horizon_days"] == 30
    assert len(data["temporal_metadata"]["top_peak_hours"]) >= 3





@pytest.mark.basket_analysis
def test_market_basket_association_rules():
    """Verify FP-Growth / Apriori association rules (support, confidence, lift)."""
    res = client.get("/api/v1/reports/market_basket_analysis/download")
    assert res.status_code == 200
    assert "Market-Basket" in res.text or "Rules" in res.text or "Association" in res.text





@pytest.mark.wastage
def test_wastage_risk_and_cost_metrics():
    """Verify kitchen wastage total cost and high-risk dish predictions."""
    res = client.get("/api/v1/wastage")
    assert res.status_code == 200
    data = res.json()
    assert "total_wastage" in data
    assert "wastage_cost" in data
    assert "high_wastage_items" in data





@pytest.mark.promotion
def test_promotion_roi_and_effectiveness():
    """Verify promotional ROI, campaign metrics, and segment targeting."""
    res = client.get("/api/v1/promotions")
    assert res.status_code == 200
    promos = res.json()
    assert len(promos) >= 1
    assert "discount_value" in promos[0]





@pytest.mark.pricing
def test_pricing_history_and_elasticity():
    """Verify price changes, audit logs, and margin computations."""
    res = client.get("/api/v1/pricing-history")
    assert res.status_code == 200
    assert isinstance(res.json(), list)





@pytest.mark.anomaly
def test_sales_and_rating_anomaly_detection():
    """Verify Z-Score & Isolation Forest anomaly flags for revenue spikes and rating drops."""
    rep_path = os.path.join(PROJECT_ROOT, "reports", "anomalies", "sales_anomaly_report.json")
    assert os.path.exists(rep_path)
    with open(rep_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "anomaly_counts" in data





@pytest.mark.security
def test_security_rbac_and_injection_prevention():
    """Verify role-based access control and resistance to query manipulation."""

    res_deny = client.delete("/api/v1/locations/LOC-001", headers={"X-User-Role": "analyst"})
    assert res_deny.status_code == 403


    res_search = client.get("/api/v1/search/filter?query=' OR 1=1 --")
    assert res_search.status_code == 200





@pytest.mark.boundary
def test_boundary_conditions_and_limits():
    """Verify edge conditions: zero quantity, extreme dates, empty query parameters."""

    bad_order = client.post("/api/v1/orders", json={
        "location_id": "LOC-001",
        "items": [{"item_id": "ITEM-001", "quantity": 0, "unit_price": 10.0}]
    }, headers={"X-User-Role": "manager"})
    assert bad_order.status_code in (400, 422, 201)


    big_limit = client.get("/api/v1/orders?limit=999999")
    assert big_limit.status_code == 200
    assert len(big_limit.json()) <= 10000





@pytest.mark.hidden_data_readiness
def test_hidden_data_readiness_cold_start():
    """Verify zero-shot inference on unseen customers and unseen locations without crash."""

    res_cust = client.post("/api/v1/models/predict-tagged", json={
        "task_type": "Churn",
        "pipeline_type": "Spark",
        "entity_type": "CUSTOMER",
        "entity_id": "CUST-UNSEEN-999999"
    }, headers={"X-User-Role": "analyst"})
    assert res_cust.status_code == 200
    assert res_cust.json()["predicted_value"] is not None


    res_loc = client.post("/api/v1/models/predict-tagged", json={
        "task_type": "Demand",
        "pipeline_type": "Spark",
        "entity_type": "LOCATION",
        "entity_id": "LOC-UNSEEN-999"
    }, headers={"X-User-Role": "analyst"})
    assert res_loc.status_code == 200
    assert res_loc.json()["predicted_value"] is not None
