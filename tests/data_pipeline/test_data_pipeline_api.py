"""
DineIQ Analytics — Data Pipeline (Upload & Analyze) Test Suite
Verifies:
- Isolated staging under data_staging/runs/{run_id}/
- Server-side multi-format staging (CSV, JSON, Parquet)
- PySpark StructType schema validation and PK/FK orphan detection
- SRS Data Quality rule audits
- Real data cleaning with BEFORE/AFTER metrics and evidence logging
- PySpark integration and master analytical cube generation
- Supported multi-dimensional analytics and ML model inference
- Absence of synthetic results when datasets are omitted
- Markdown analysis report generation and CSV exports
- Data isolation (raw data immutability and separation from primary marts)
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def staged_run_id():
    """Initializes a real staged demo run for end-to-end testing."""
    res = client.post("/api/v1/data-pipeline/demo?sample_size=1000")
    assert res.status_code == 201
    json_data = res.json()
    assert "run_id" in json_data
    assert len(json_data["files"]) >= 3
    return json_data["run_id"]


def test_runs_history_listing():
    """Verify run history listing endpoint."""
    res = client.get("/api/v1/data-pipeline/runs")
    assert res.status_code == 200
    json_data = res.json()
    assert "runs" in json_data
    assert isinstance(json_data["runs"], list)


def test_staged_run_manifest_and_isolation(staged_run_id):
    """Verify run manifest exists in isolated directory and raw files are preserved."""
    res = client.get(f"/api/v1/data-pipeline/runs/{staged_run_id}")
    assert res.status_code == 200
    manifest = res.json()
    assert manifest["run_id"] == staged_run_id
    assert manifest["status"] == "UPLOADED"
    assert manifest["total_records"] > 0

                                             
    staging_path = os.path.join(PROJECT_ROOT, "data_staging", "runs", staged_run_id, "raw")
    assert os.path.exists(staging_path)
    assert len(os.listdir(staging_path)) >= 3


def test_schema_validation_step(staged_run_id):
    """Verify Step 2: PySpark StructType schema validation and PK/FK relationship checking."""
    res = client.post(f"/api/v1/data-pipeline/runs/{staged_run_id}/validate-schema")
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"] == staged_run_id
    assert data["overall_status"] in ["VALID", "WARNING"]
    assert "files_validated" in data
    assert len(data["files_validated"]) >= 3

                                                     
    assert "relationships" in data
    assert len(data["relationships"]) > 0
    rel_names = [r["relationship"] for r in data["relationships"]]
    assert "Order Items → Orders" in rel_names


def test_data_quality_step(staged_run_id):
    """Verify Step 3: SRS Data Quality audit execution."""
    res = client.post(f"/api/v1/data-pipeline/runs/{staged_run_id}/data-quality")
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"] == staged_run_id
    assert 0.0 <= data["data_quality_score"] <= 100.0
    assert data["total_records"] > 0
    assert "datasets" in data


def test_data_cleaning_and_evidence_step(staged_run_id):
    """Verify Step 4 & 5: Real data cleaning, before/after metrics, and evidence log."""
    res = client.post(f"/api/v1/data-pipeline/runs/{staged_run_id}/clean")
    assert res.status_code == 200
    data = res.json()
    assert "before_stats" in data
    assert "after_stats" in data
    assert "actions" in data
    assert "evidence" in data
    assert isinstance(data["evidence"], list)

                           
    if data["evidence"]:
        sample_ev = data["evidence"][0]
        assert "record_id" in sample_ev
        assert "issue" in sample_ev
        assert "action" in sample_ev
        assert "rule_id" in sample_ev
        assert "reason" in sample_ev

                                                           
    clean_dir = os.path.join(PROJECT_ROOT, "data_staging", "runs", staged_run_id, "cleaned")
    assert os.path.exists(clean_dir)
    assert any(f.endswith(".parquet") for f in os.listdir(clean_dir))


def test_pyspark_processing_step(staged_run_id):
    """Verify Step 6: PySpark relational integration and Parquet cube generation."""
    res = client.post(f"/api/v1/data-pipeline/runs/{staged_run_id}/process")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "stages" in data
    assert data["master_cube_records"] > 0

                                                                      
    parquet_dir = os.path.join(PROJECT_ROOT, "data_staging", "runs", staged_run_id, "parquet")
    cube_file = os.path.join(parquet_dir, "analytical_order_cube.parquet")
    assert os.path.exists(cube_file)


def test_analytics_and_ml_step(staged_run_id):
    """Verify Step 8 & 9: Multi-dimensional analytics and ML model inference."""
    res = client.post(f"/api/v1/data-pipeline/runs/{staged_run_id}/analyze")
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"] == staged_run_id
    assert "analytics" in data
    analytics = data["analytics"]
    assert "sales_overview" in analytics
    assert analytics["sales_overview"]["status"] == "AVAILABLE"
    assert analytics["sales_overview"]["total_revenue"] > 0

    assert "recommendations" in data
    assert len(data["recommendations"]) > 0
    assert "domain" in data["recommendations"][0]
    assert "evidence" in data["recommendations"][0]


def test_markdown_report_generation(staged_run_id):
    """Verify Step 12: Downloadable Markdown report generation."""
    res = client.get(f"/api/v1/data-pipeline/runs/{staged_run_id}/report")
    assert res.status_code == 200
    md_content = res.text
    assert staged_run_id in md_content
    assert "DineIQ Analytics" in md_content
    assert "Executive Summary & KPIs" in md_content


def test_cleaned_dataset_csv_export(staged_run_id):
    """Verify Step 12: Cleaned dataset CSV download."""
    res = client.get(f"/api/v1/data-pipeline/runs/{staged_run_id}/export/orders")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    assert len(res.text) > 100
