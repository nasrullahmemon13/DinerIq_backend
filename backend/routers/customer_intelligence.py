"""
DineIQ Analytics - Customer Intelligence Dashboard Router (SRS Step 44)
Serves data for:
- customer segments
- RFM distribution
- high-value customers
- at-risk customers
- promotion-sensitive customers
- customer trends
"""

from fastapi import APIRouter, HTTPException
from backend.services.dashboard_service import DashboardService

router = APIRouter(prefix="/api/v1/customer-intelligence", tags=["Customer Intelligence (Step 44)"])
service = DashboardService()


@router.get("")
def get_customer_intelligence_dashboard():
    """
    Returns full Customer Intelligence Dashboard payload conforming to SRS Step 44:
    customer segments, RFM distribution, high-value customers, at-risk customers,
    promotion-sensitive customers, customer trends.
    """
    data = service.get_customer_intelligence_dashboard_data()
    if "error" in data:
        raise HTTPException(status_code=500, detail=data["error"])
    return data


@router.get("/segments")
def get_customer_segments_summary():
    """
    Returns breakdown of the 6 customer segments.
    """
    data = service.get_customer_intelligence_dashboard_data()
    return {
        "srs_step": 44,
        "segments": data["customer_segments"]
    }


@router.get("/high-value")
def get_high_value_customers():
    """
    Returns high-value customer cohort directory.
    """
    data = service.get_customer_intelligence_dashboard_data()
    return {
        "srs_step": 44,
        "high_value_customers": data["high_value_customers"]
    }


@router.get("/at-risk")
def get_at_risk_customers():
    """
    Returns at-risk customer cohort with Step 36 churn indicators.
    """
    data = service.get_customer_intelligence_dashboard_data()
    return {
        "srs_step": 44,
        "at_risk_customers": data["at_risk_customers"]
    }


@router.get("/promotion-sensitive")
def get_promotion_sensitive_customers():
    """
    Returns promotion-sensitive customer cohort.
    """
    data = service.get_customer_intelligence_dashboard_data()
    return {
        "srs_step": 44,
        "promotion_sensitive_customers": data["promotion_sensitive_customers"]
    }
