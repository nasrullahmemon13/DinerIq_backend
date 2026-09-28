"""
Unit and Integration Tests for Food Wastage Dashboard (SRS Step 45)
Verifies:
- total wastage
- wastage cost
- high-wastage items
- high-wastage locations
- wastage trends
- wastage-risk predictions
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_wastage_dashboard_all_srs_fields():
    """Verify that /api/v1/wastage returns all exact Step 45 fields."""
    resp = client.get("/api/v1/wastage")
    assert resp.status_code == 200
    data = resp.json()


    assert data["srs_step"] == 45
    assert "summary_metrics" in data


    assert "total_wastage" in data
    assert data["total_wastage"] > 0
    assert "wastage_cost" in data
    assert data["wastage_cost"] > 0


    assert "high_wastage_items" in data
    assert len(data["high_wastage_items"]) == 150
    sample_item = data["high_wastage_items"][0]
    for k in ["item_id", "item_name", "category_name", "wasted_quantity", "total_loss_amount", "incident_count"]:
        assert k in sample_item, f"Missing key in high-wastage item: {k}"


    assert "high_wastage_locations" in data
    assert len(data["high_wastage_locations"]) == 20
    sample_loc = data["high_wastage_locations"][0]
    for k in ["location_id", "restaurant_name", "wasted_quantity", "total_loss_amount", "loss_share_pct"]:
        assert k in sample_loc, f"Missing key in high-wastage location: {k}"


    assert "wastage_trends" in data
    trends = data["wastage_trends"]
    assert "day_of_week_trends" in trends
    assert len(trends["day_of_week_trends"]) == 7
    assert "shift_period_trends" in trends
    assert len(trends["shift_period_trends"]) == 3
    assert "monthly_trends" in trends
    assert len(trends["monthly_trends"]) == 12


    assert "wastage_risk_predictions" in data
    assert len(data["wastage_risk_predictions"]) > 0
    sample_pred = data["wastage_risk_predictions"][0]
    for k in ["location_id", "item_id", "name", "preparation_quantity", "quantity_wasted", "wastage_risk_tier", "predicted_risk_probability", "actionable_mitigation_strategy"]:
        assert k in sample_pred, f"Missing key in wastage risk prediction: {k}"


def test_wastage_items_endpoint():
    """Verify /api/v1/wastage/items endpoint."""
    resp = client.get("/api/v1/wastage/items")
    assert resp.status_code == 200
    data = resp.json()
    assert "high_wastage_items" in data
    assert len(data["high_wastage_items"]) == 150


def test_wastage_locations_endpoint():
    """Verify /api/v1/wastage/locations endpoint."""
    resp = client.get("/api/v1/wastage/locations")
    assert resp.status_code == 200
    data = resp.json()
    assert "high_wastage_locations" in data
    assert len(data["high_wastage_locations"]) == 20


def test_wastage_trends_endpoint():
    """Verify /api/v1/wastage/trends endpoint."""
    resp = client.get("/api/v1/wastage/trends")
    assert resp.status_code == 200
    data = resp.json()
    assert "wastage_trends" in data
    assert "day_of_week_trends" in data["wastage_trends"]


def test_wastage_predictions_endpoint():
    """Verify /api/v1/wastage/predictions endpoint."""
    resp = client.get("/api/v1/wastage/predictions")
    assert resp.status_code == 200
    data = resp.json()
    assert "wastage_risk_predictions" in data
    assert len(data["wastage_risk_predictions"]) > 0
