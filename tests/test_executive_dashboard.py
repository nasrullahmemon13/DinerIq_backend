"""
Unit and Integration Tests for SRS Step 42: Executive Dashboard
Verifies that the Executive Dashboard serves all 10 SRS-mandated fields:
1. Total revenue
2. Total profit
3. Total orders
4. Average order value
5. Active customers
6. Repeat customers
7. Wastage
8. Forecast demand
9. Critical recommendations
10. Anomalies
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app
from backend.services.dashboard_service import DashboardService


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def service():
    return DashboardService()


def test_executive_service_data(service):
    """Assert DashboardService returns valid payload with all 10 SRS Step 42 fields."""
    data = service.get_executive_dashboard_data()

                            
    assert "total_revenue" in data
    assert data["total_revenue"] > 1000000.0

                           
    assert "total_profit" in data
    assert "net_profitability" in data
    assert data["total_profit"] > 0.0

                           
    assert "total_orders" in data
    assert data["total_orders"] > 50000

                                  
    assert "average_order_value" in data
    assert data["average_order_value"] > 50.0

                               
    assert "active_customers" in data
    assert data["active_customers"] > 10000

                               
    assert "repeat_customers" in data
    assert data["repeat_customers"] > 5000

                      
    assert "wastage" in data
    assert "total_wastage_cost" in data["wastage"]
    assert "total_wastage_units" in data["wastage"]
    assert data["wastage"]["total_wastage_cost"] > 0.0

                              
    assert "forecast_demand" in data
    assert "projected_demand_units" in data["forecast_demand"]
    assert data["forecast_demand"]["projected_demand_units"] > 1000000

                                       
    assert "critical_recommendations" in data
    assert isinstance(data["critical_recommendations"], list)
    assert len(data["critical_recommendations"]) > 0

                                                                   
    for rec in data["critical_recommendations"]:
        assert rec["priority"] == "Critical"
        assert len(rec["recommended_action"]) > 5
        assert len(rec["reason_bullets"]) >= 3
        assert rec["potential_business_impact"] > 0

                         
    assert "anomalies" in data
    assert isinstance(data["anomalies"], list)
    assert len(data["anomalies"]) > 0


def test_executive_api_endpoint(client):
    """Assert GET /api/v1/dashboard/executive returns 200 and matches SRS contract."""
    response = client.get("/api/v1/dashboard/executive")
    assert response.status_code == 200

    payload = response.json()
    assert payload["srs_step"] == 42

    required_fields = [
        "total_revenue",
        "total_profit",
        "total_orders",
        "average_order_value",
        "active_customers",
        "repeat_customers",
        "wastage",
        "forecast_demand",
        "critical_recommendations",
        "anomalies"
    ]
    for f in required_fields:
        assert f in payload, f"Missing required SRS Step 42 field: {f}"


def test_frontend_root_serving(client):
    """Assert root route serves React distribution HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<!doctype html>" in response.text.lower()
