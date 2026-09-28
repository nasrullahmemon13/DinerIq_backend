"""
DineIQ Analytics - Executive Dashboard Router (SRS Step 42)
Provides API endpoints for:
- total revenue
- total profit
- total orders
- average order value
- active customers
- repeat customers
- wastage
- forecast demand
- critical recommendations
- anomalies
"""

from typing import Optional
from fastapi import APIRouter, Query
from backend.services.dashboard_service import DashboardService

router = APIRouter(prefix="/api/v1/dashboard/executive", tags=["Executive Dashboard"])


@router.get("")
def get_executive_dashboard(
    location_id: Optional[str] = Query(None, description="Optional location filter ID (e.g. LOC-001 or ALL)"),
    date_range: Optional[str] = Query(None, description="Optional date range filter (e.g. Q1, Q2, ALL, 30_DAYS)")
):
    """
    Returns complete Executive Dashboard payload implementing SRS Step 42 dynamically.
    """
    service = DashboardService()
    return service.get_executive_dashboard_data(location_id=location_id, date_range=date_range)
