"""
DineIQ Analytics - Food Wastage Dashboard Router (SRS Step 45)
Serves data for:
- total wastage
- wastage cost
- high-wastage items
- high-wastage locations
- wastage trends
- wastage-risk predictions
"""

from fastapi import APIRouter, HTTPException
from backend.services.dashboard_service import DashboardService

router = APIRouter(prefix="/api/v1/wastage", tags=["Wastage Intelligence (Step 45)"])
service = DashboardService()


@router.get("")
def get_wastage_dashboard():
    """
    Returns full Wastage Dashboard payload conforming to SRS Step 45:
    total wastage, wastage cost, high-wastage items, high-wastage locations,
    wastage trends, wastage-risk predictions.
    """
    data = service.get_wastage_dashboard_data()
    if "error" in data:
        raise HTTPException(status_code=500, detail=data["error"])
    return data


@router.get("/items")
def get_high_wastage_items():
    """
    Returns items ranked by total wastage cost and units lost.
    """
    data = service.get_wastage_dashboard_data()
    return {
        "srs_step": 45,
        "total_items": len(data["high_wastage_items"]),
        "high_wastage_items": data["high_wastage_items"]
    }


@router.get("/locations")
def get_high_wastage_locations():
    """
    Returns locations ranked by total food wastage loss.
    """
    data = service.get_wastage_dashboard_data()
    return {
        "srs_step": 45,
        "total_locations": len(data["high_wastage_locations"]),
        "high_wastage_locations": data["high_wastage_locations"]
    }


@router.get("/trends")
def get_wastage_trends():
    """
    Returns day-of-week, shift period root cause, and monthly wastage trends.
    """
    data = service.get_wastage_dashboard_data()
    return {
        "srs_step": 45,
        "wastage_trends": data["wastage_trends"]
    }


@router.get("/predictions")
def get_wastage_risk_predictions():
    """
    Returns high-risk predictive batches and mitigation recommendations.
    """
    data = service.get_wastage_dashboard_data()
    return {
        "srs_step": 45,
        "predictions_count": len(data["wastage_risk_predictions"]),
        "wastage_risk_predictions": data["wastage_risk_predictions"]
    }
