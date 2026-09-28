"""
Tests for SRS Functional Requirements (lxi) through (lxv):
  (lxi)  Database Storage (Config, metadata, users, recommendations, results)
  (lxii) Model Version Tracking (Every prediction tagged with model version)
  (lxiii) Audit Trail (Log all data-processing jobs, predictions, exports, admin actions)
  (lxiv) Error Handling (Understandable errors for processing, model, Spark, database failures)
  (lxv)  Spark Job Monitoring (Display job status, stages, duration, metrics)
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

ADMIN_HEADERS = {"X-User-Role": "admin", "X-Username": "admin_tester"}
ANALYST_HEADERS = {"X-User-Role": "analyst", "X-Username": "analyst_tester"}






def test_database_storage_and_config():
    """Verify system config storage, admin updates, and database storage metrics."""

    cfg_res = client.get("/api/v1/system/config")
    assert cfg_res.status_code == 200
    configs = cfg_res.json()
    assert len(configs) >= 1
    assert any(c["config_key"] == "pipeline.default_engine" for c in configs)


    up_res = client.put(
        "/api/v1/system/config/pipeline.default_engine",
        json={"config_value": "PySpark-Enterprise", "description": "Updated cluster engine"},
        headers=ADMIN_HEADERS
    )
    assert up_res.status_code == 200


    metrics_res = client.get("/api/v1/system/storage-metrics")
    assert metrics_res.status_code == 200
    metrics = metrics_res.json()
    assert metrics["storage_status"] == "ONLINE_SECURE"
    assert "entity_counts" in metrics
    assert metrics["entity_counts"]["users"] >= 1
    assert metrics["entity_counts"]["audit_logs"] >= 1






def test_model_version_tracking_and_tagged_predictions():
    """Verify model version registration and that every prediction is tagged with model version."""

    ver_res = client.get("/api/v1/models/versions")
    assert ver_res.status_code == 200
    versions = ver_res.json()
    assert len(versions) >= 2
    assert any(v["task_type"] == "Churn" for v in versions)


    new_v = {
        "model_name": "Location Demand Gradient Boosting",
        "version_tag": "v3.0.0-mllib",
        "framework": "PySpark MLlib",
        "pipeline_type": "Spark",
        "task_type": "Demand",
        "metrics": '{"rmse": 10.4, "r2": 0.91}',
        "artifact_uri": "models/spark/demand_gbt_v3.model"
    }
    reg_res = client.post("/api/v1/models/versions", json=new_v, headers=ANALYST_HEADERS)
    assert reg_res.status_code == 201
    assert "version_id" in reg_res.json()


    pred_res = client.post("/api/v1/models/predict-tagged", json={
        "task_type": "Churn",
        "pipeline_type": "Spark",
        "entity_type": "CUSTOMER",
        "entity_id": "CUST-TEST-99",
        "features": {"recency_days": 45, "frequency_orders": 2}
    }, headers=ANALYST_HEADERS)
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert "prediction_id" in pred_data
    assert "model_version_id" in pred_data
    assert "version_tag" in pred_data
    assert pred_data["version_tag"] == "v2.1.0-mllib"
    assert pred_data["predicted_value"] == "AT_RISK"


    list_preds = client.get("/api/v1/models/predictions?entity_id=CUST-TEST-99")
    assert list_preds.status_code == 200
    assert len(list_preds.json()) >= 1
    assert list_preds.json()[0]["model_version_id"] is not None






def test_audit_trail_logging_and_queries():
    """Verify audit logs record data jobs, predictions, exports, and admin operations."""
    trail_res = client.get("/api/v1/audit-trail")
    assert trail_res.status_code == 200
    logs = trail_res.json()
    assert len(logs) >= 4

    event_types = {l["event_type"] for l in logs}

    assert "DATA_PROCESSING_JOB" in event_types or "ADMIN_ACTION" in event_types or "PREDICTION" in event_types


    new_entry = client.post("/api/v1/audit-trail", json={
        "event_type": "DATA_EXPORT",
        "action": "EXPORT_LOCATION_INTELLIGENCE",
        "resource_id": "LOC-001",
        "status": "SUCCESS",
        "details": "User exported location intelligence report"
    }, headers=ADMIN_HEADERS)
    assert new_entry.status_code == 201






def test_understandable_error_handling_processing():
    """Verify processing failure returns understandable error response."""
    res = client.get("/api/v1/system/test-error/processing")
    assert res.status_code == 422
    data = res.json()
    assert data["status"] == "error"
    assert data["category"] == "PROCESSING"
    assert "suggested_action" in data
    assert "orders_batch_04.csv" in data["message"]


def test_understandable_error_handling_model():
    """Verify model failure returns understandable error response."""
    res = client.get("/api/v1/system/test-error/model")
    assert res.status_code == 500
    data = res.json()
    assert data["status"] == "error"
    assert data["category"] == "MODEL"
    assert "suggested_action" in data
    assert "feature dimension mismatch" in data["message"]


def test_understandable_error_handling_spark():
    """Verify Spark failure returns understandable error response."""
    res = client.get("/api/v1/system/test-error/spark")
    assert res.status_code == 502
    data = res.json()
    assert data["status"] == "error"
    assert data["category"] == "SPARK"
    assert "suggested_action" in data
    assert "out-of-memory" in data["message"]


def test_understandable_error_handling_database():
    """Verify database failure returns understandable error response."""
    res = client.get("/api/v1/system/test-error/database")
    assert res.status_code == 503
    data = res.json()
    assert data["status"] == "error"
    assert data["category"] == "DATABASE"
    assert "suggested_action" in data
    assert "Foreign key constraint" in data["message"]






def test_spark_job_monitoring():
    """Verify Spark job monitoring displays status, stages, duration, and telemetry."""

    jobs_res = client.get("/api/v1/spark/jobs")
    assert jobs_res.status_code == 200
    data = jobs_res.json()
    assert "summary" in data
    assert data["summary"]["total_jobs"] >= 3
    assert len(data["jobs"]) >= 3


    job = data["jobs"][0]
    assert "job_id" in job
    assert "status" in job
    assert "stages_completed" in job
    assert "total_stages" in job
    assert "duration_seconds" in job


    single_res = client.get(f"/api/v1/spark/jobs/{job['job_id']}")
    assert single_res.status_code == 200
    assert single_res.json()["job_id"] == job["job_id"]


    trig_res = client.post("/api/v1/spark/jobs/trigger", json={
        "job_name": "PySpark MLlib Feature Transformation",
        "pipeline_type": "PySpark",
        "total_stages": 6,
        "records_processed": 50000
    }, headers=ADMIN_HEADERS)
    assert trig_res.status_code == 201
    assert trig_res.json()["status"] == "RUNNING"
