"""
Unit and Integration Tests for Menu Intelligence Dashboard (SRS Step 43)
Verifies:
- menu-item performance
- Profit Drivers
- Volume Drivers
- Hidden Opportunities
- Low Performers
- slow-moving items
- ratings
- margins
- wastage
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_menu_intelligence_dashboard_all_srs_fields():
    """Verify that /api/v1/menu-intelligence returns all exact Step 43 fields."""
    resp = client.get("/api/v1/menu-intelligence")
    assert resp.status_code == 200
    data = resp.json()

                   
    assert data["srs_step"] == 43
    assert "summary_metrics" in data

                              
    assert "menu_item_performance" in data
    assert len(data["menu_item_performance"]) == 150
    sample_item = data["menu_item_performance"][0]
    required_item_keys = [
        "item_id", "item_name", "category_name", "base_price", "cost_price",
        "quantity_sold", "revenue", "contribution_margin", "margin_pct",
        "customer_rating", "wastage_percentage", "total_wastage_cost", "classification"
    ]
    for key in required_item_keys:
        assert key in sample_item, f"Missing key in menu item: {key}"

                       
    assert "profit_drivers" in data
    assert len(data["profit_drivers"]) == 31
    for item in data["profit_drivers"]:
        assert item["classification"] == "Profit Driver"

                       
    assert "volume_drivers" in data
    assert len(data["volume_drivers"]) == 39
    for item in data["volume_drivers"]:
        assert item["classification"] == "Volume Driver"

                             
    assert "hidden_opportunities" in data
    assert len(data["hidden_opportunities"]) == 9
    for item in data["hidden_opportunities"]:
        assert item["classification"] == "Hidden Opportunity"

                       
    assert "low_performers" in data
    assert len(data["low_performers"]) == 71
    for item in data["low_performers"]:
        assert item["classification"] == "Low Performer"

                             
    quadrant_sum = (
        len(data["profit_drivers"]) +
        len(data["volume_drivers"]) +
        len(data["hidden_opportunities"]) +
        len(data["low_performers"])
    )
    assert quadrant_sum == 150

                                    
    assert "slow_moving_items" in data
    assert len(data["slow_moving_items"]) == 73
    sample_slow = data["slow_moving_items"][0]
    assert "movement_class" in sample_slow
    assert "recommended_action" in sample_slow
    assert "srs_dimensions_triggered_count" in sample_slow

                
    assert "ratings" in data
    assert "overall_average_rating" in data["ratings"]
    assert 1.0 <= data["ratings"]["overall_average_rating"] <= 5.0
    assert "rating_distribution" in data["ratings"]
    assert "top_rated_items" in data["ratings"]
    assert "lowest_rated_items" in data["ratings"]

                
    assert "margins" in data
    assert "overall_average_margin_pct" in data["margins"]
    assert data["margins"]["overall_average_margin_pct"] > 0
    assert "margin_distribution" in data["margins"]
    assert "highest_margin_items" in data["margins"]
    assert "lowest_margin_items" in data["margins"]

                
    assert "wastage" in data
    assert "total_wastage_cost" in data["wastage"]
    assert data["wastage"]["total_wastage_cost"] > 0
    assert "total_wastage_units" in data["wastage"]
    assert "average_wastage_pct" in data["wastage"]
    assert "highest_wastage_items" in data["wastage"]
    assert "category_wastage_breakdown" in data["wastage"]


@pytest.mark.parametrize("quadrant,expected_count", [
    ("profit_drivers", 31),
    ("volume_drivers", 39),
    ("hidden_opportunities", 9),
    ("low_performers", 71),
])
def test_menu_quadrant_filtering(quadrant, expected_count):
    """Test filtering by specific quadrant endpoint."""
    resp = client.get(f"/api/v1/menu-intelligence/quadrant/{quadrant}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == expected_count
    assert len(data["items"]) == expected_count


def test_menu_quadrant_invalid_name():
    """Test error handling on invalid quadrant."""
    resp = client.get("/api/v1/menu-intelligence/quadrant/super_unicorns")
    assert resp.status_code == 400
    assert "Invalid quadrant" in resp.json()["detail"]


def test_slow_moving_dishes_endpoint():
    """Test slow moving dishes endpoint returns 73 flagged dishes."""
    resp = client.get("/api/v1/menu-intelligence/slow-moving")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_slow_moving"] == 73
    assert len(data["slow_moving_items"]) == 73
