"""
Unit & Integration Tests for SRS Steps 48, 49, 50:
- Step 48: Search & Multi-Dimensional Filtering (11 dimensions)
- Step 49: Downloadable Reports (12 required SRS topics)
- Step 50: Role-Based Data Export (CSV & Excel with RBAC permission check)
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from src.search_filter_engine import SearchFilterEngine
from src.report_generator import ReportGenerator
from src.data_export_engine import DataExportEngine, PERMITTED_ROLES

client = TestClient(app)





def test_search_options_all_11_dimensions():
    """Verify that options endpoint returns metadata covering all 11 SRS dimensions."""
    resp = client.get("/api/v1/search/options")
    assert resp.status_code == 200
    data = resp.json()
    assert "dimensions" in data
    dims = data["dimensions"]


    assert "date_range" in dims
    assert "min_date" in dims["date_range"]
    assert "max_date" in dims["date_range"]


    assert "locations" in dims and len(dims["locations"]) > 0


    assert "menu_items" in dims and len(dims["menu_items"]) > 0


    assert "categories" in dims and len(dims["categories"]) > 0


    assert "customer_segments" in dims and len(dims["customer_segments"]) > 0


    assert "ordering_channels" in dims and len(dims["ordering_channels"]) > 0


    assert "promotions" in dims and len(dims["promotions"]) > 0


    assert "performance_classes" in dims and len(dims["performance_classes"]) > 0


    assert "price_range" in dims
    assert dims["price_range"]["min"] <= dims["price_range"]["max"]


    assert "rating_range" in dims
    assert dims["rating_range"]["min"] >= 1.0


    assert "wastage_range" in dims
    assert dims["wastage_range"]["min"] >= 0.0


def test_filter_by_location():
    """Test filtering by specific location."""
    resp = client.get("/api/v1/search/filter?location=LOC-001")
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "success"
    assert res["total_matches"] > 0
    for record in res["records"]:
        assert record["location_id"] == "LOC-001"


def test_filter_by_performance_class():
    """Test filtering by menu performance class (e.g. Profit Driver)."""
    resp = client.get("/api/v1/search/filter?performance_class=Profit Driver")
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "success"
    assert res["total_matches"] > 0
    for record in res["records"]:
        assert record["performance_class"] == "Profit Driver"


def test_filter_by_price_and_rating_ranges():
    """Test numeric range filters on price, rating, and wastage."""
    resp = client.get("/api/v1/search/filter?min_price=10.0&max_price=30.0&min_rating=3.5")
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "success"
    for record in res["records"]:
        assert 10.0 <= record["base_price"] <= 30.0
        if record.get("avg_rating"):
            assert record["avg_rating"] >= 3.5


def test_filter_combined_multi_dimension():
    """Test combining multiple SRS dimensions simultaneously."""
    resp = client.get(
        "/api/v1/search/filter?location=LOC-001&menu_category=Steaks&min_price=5.0"
    )
    assert resp.status_code == 200
    res = resp.json()
    assert "summary_kpis" in res
    assert "total_revenue" in res["summary_kpis"]
    assert "total_volume" in res["summary_kpis"]
    assert "avg_margin_pct" in res["summary_kpis"]






EXACT_12_SRS_REPORTS = [
    "menu_performance",
    "profitability",
    "customer_segmentation",
    "market_basket_analysis",
    "demand_forecast",
    "wastage",
    "promotions",
    "pricing",
    "location_performance",
    "anomalies",
    "recommendations",
    "spark_vs_python_comparison"
]

def test_reports_manifest_exact_12_topics():
    """Ensure catalog lists exactly the 12 SRS reports."""
    resp = client.get("/api/v1/reports")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_reports"] == 12
    returned_keys = [r["key"] for r in data["reports"]]
    for key in EXACT_12_SRS_REPORTS:
        assert key in returned_keys, f"Missing required SRS report key: {key}"


@pytest.mark.parametrize("report_key", EXACT_12_SRS_REPORTS)
def test_download_each_srs_report(report_key):
    """Test downloading each of the 12 SRS reports."""
    resp = client.get(f"/api/v1/reports/{report_key}/download")
    assert resp.status_code == 200
    assert "text/markdown" in resp.headers.get("content-type", "")
    assert len(resp.content) > 100
    assert "Content-Disposition" in resp.headers


def test_download_invalid_report_key():
    """Test downloading nonexistent report key returns 404."""
    resp = client.get("/api/v1/reports/nonexistent_report_xyz/download")
    assert resp.status_code == 404






def test_export_datasets_list():
    """Ensure export dataset catalog is accessible."""
    resp = client.get("/api/v1/export/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert "datasets" in data
    assert len(data["datasets"]) > 0


def test_export_denied_for_viewer_role():
    """Verify viewers receive 403 Forbidden on data export."""
    resp = client.get(
        "/api/v1/export/menu_classification?format=csv",
        headers={"X-User-Role": "viewer"}
    )
    assert resp.status_code == 403
    assert "permission" in resp.json()["detail"].lower()


def test_export_denied_without_role_header():
    """Verify unauthorized request without role header receives 403."""
    resp = client.get("/api/v1/export/menu_classification?format=csv")
    assert resp.status_code == 403


@pytest.mark.parametrize("role", ["admin", "analyst", "executive"])
def test_export_allowed_for_permitted_roles_csv(role):
    """Verify permitted roles can export CSV."""
    resp = client.get(
        "/api/v1/export/menu_classification?format=csv",
        headers={"X-User-Role": role}
    )
    assert resp.status_code == 200
    assert "text/csv" in resp.headers.get("content-type", "")
    assert "Content-Disposition" in resp.headers
    text = resp.content.decode("utf-8-sig")
    assert "item_id" in text or "item_name" in text


def test_export_allowed_for_permitted_roles_excel():
    """Verify permitted roles can export Excel (.xlsx)."""
    resp = client.get(
        "/api/v1/export/menu_classification?format=excel",
        headers={"X-User-Role": "analyst"}
    )
    assert resp.status_code == 200
    assert "spreadsheetml" in resp.headers.get("content-type", "")
    assert len(resp.content) > 1000
