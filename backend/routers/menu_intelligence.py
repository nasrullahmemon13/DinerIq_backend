"""
DineIQ Analytics - Menu Intelligence Dashboard Router (SRS Step 43)
Serves data for:
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

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from backend.services.dashboard_service import DashboardService

router = APIRouter(prefix="/api/v1/menu-intelligence", tags=["Menu Intelligence (Step 43)"])
service = DashboardService()


@router.get("")
def get_menu_intelligence_dashboard():
    """
    Returns full Menu Intelligence Dashboard payload conforming to SRS Step 43:
    menu-item performance, Profit Drivers, Volume Drivers, Hidden Opportunities,
    Low Performers, slow-moving items, ratings, margins, wastage.
    """
    data = service.get_menu_intelligence_dashboard_data()
    if "error" in data:
        raise HTTPException(status_code=500, detail=data["error"])
    return data


@router.get("/quadrant/{quadrant_name}")
def get_menu_quadrant(quadrant_name: str):
    """
    Returns menu items belonging to a specific 4-quadrant classification:
    profit_drivers, volume_drivers, hidden_opportunities, low_performers.
    """
    data = service.get_menu_intelligence_dashboard_data()
    q_map = {
        "profit_drivers": "profit_drivers",
        "volume_drivers": "volume_drivers",
        "hidden_opportunities": "hidden_opportunities",
        "low_performers": "low_performers"
    }
    normalized = quadrant_name.lower().replace("-", "_").replace(" ", "_")
    if normalized not in q_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid quadrant '{quadrant_name}'. Must be one of: {list(q_map.keys())}"
        )

    field = q_map[normalized]
    return {
        "srs_step": 43,
        "quadrant": field,
        "count": len(data[field]),
        "items": data[field]
    }


@router.get("/slow-moving")
def get_slow_moving_dishes():
    """
    Returns slow-moving dishes identified by the 7 SRS Step 32 factors.
    """
    data = service.get_menu_intelligence_dashboard_data()
    return {
        "srs_step": 43,
        "total_slow_moving": len(data["slow_moving_items"]),
        "slow_moving_items": data["slow_moving_items"]
    }
