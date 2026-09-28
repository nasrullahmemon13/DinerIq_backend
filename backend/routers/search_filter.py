"""
DineIQ Analytics - Search & Filtering Router (SRS Step 48)
"""

from typing import Optional
from fastapi import APIRouter, Query
from src.search_filter_engine import SearchFilterEngine, FilterCriteria

router = APIRouter(prefix="/api/v1/search", tags=["Search and Filtering"])
engine = SearchFilterEngine()


@router.get("/options")
def get_filter_options():
    """Returns metadata and selectable options for all 11 SRS Step 48 dimensions."""
    return engine.get_filter_options()


@router.get("/filter")
def execute_filter(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    location: Optional[str] = Query(None, description="Location ID or restaurant name"),
    menu_item: Optional[str] = Query(None, description="Menu item ID or name"),
    menu_category: Optional[str] = Query(None, description="Menu category name"),
    customer_segment: Optional[str] = Query(None, description="Customer segment (e.g. HIGH_VALUE)"),
    ordering_channel: Optional[str] = Query(None, description="Ordering channel"),
    promotion: Optional[str] = Query(None, description="Promotion ID or name"),
    performance_class: Optional[str] = Query(None, description="Performance class (Profit Driver, etc.)"),
    min_price: Optional[float] = Query(None, description="Minimum price ($)"),
    max_price: Optional[float] = Query(None, description="Maximum price ($)"),
    min_rating: Optional[float] = Query(None, description="Minimum rating (1.0-5.0)"),
    max_rating: Optional[float] = Query(None, description="Maximum rating (1.0-5.0)"),
    min_wastage_rate: Optional[float] = Query(None, description="Minimum wastage rate (%)"),
    max_wastage_rate: Optional[float] = Query(None, description="Maximum wastage rate (%)"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return")
):
    """
    Executes search and filter query across the 11 SRS Step 48 dimensions.
    """
    criteria = FilterCriteria(
        start_date=start_date,
        end_date=end_date,
        location=location,
        menu_item=menu_item,
        menu_category=menu_category,
        customer_segment=customer_segment,
        ordering_channel=ordering_channel,
        promotion=promotion,
        performance_class=performance_class,
        min_price=min_price,
        max_price=max_price,
        min_rating=min_rating,
        max_rating=max_rating,
        min_wastage_rate=min_wastage_rate,
        max_wastage_rate=max_wastage_rate
    )
    return engine.execute_filter(criteria, limit=limit)
