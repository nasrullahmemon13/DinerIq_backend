"""
DineIQ Analytics - Ratings Management Router
Implements SRS Step 1 & Step 29 / Functional Requirement (ix) Rating Management:
- Customer rating and dining review feed linked to menu items and restaurant locations
- Summary KPIs (Average Rating, Total Rating Count, 5-Star Share %, CSAT Benchmark)
- Filter by star ratings, location, search review text
"""

import math
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from database.connection import get_db
from database.models import Rating, Restaurant, MenuItem

router = APIRouter(prefix="/api/v1/ratings", tags=["Rating Management (FR-ix)"])


@router.get("/feed")
def list_ratings_feed(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=5, le=100),
    location_id: Optional[str] = Query(None, description="Filter by location ID"),
    star_rating: Optional[int] = Query(None, ge=1, le=5, description="Filter by exact star rating"),
    min_rating: Optional[int] = Query(None, ge=1, le=5, description="Filter by minimum star rating"),
    search: Optional[str] = Query(None, description="Search review text, menu item, or customer reference"),
    db: Session = Depends(get_db)
):
    """
    Retrieve customer ratings and dining reviews with overall rating summary metrics.
    """
    query = db.query(Rating)

    if location_id and location_id.upper() not in ("ALL", ""):
        query = query.filter(Rating.location_id == location_id.strip())

    if star_rating:
        query = query.filter(Rating.overall_rating == star_rating)
    elif min_rating:
        query = query.filter(Rating.overall_rating >= min_rating)

    if search and search.strip():
        s = f"%{search.strip().lower()}%"
        query = query.filter(
            (func.lower(Rating.review_text).like(s)) |
            (func.lower(Rating.item_id).like(s)) |
            (func.lower(Rating.customer_id).like(s))
        )

    total_count = query.count()
    avg_rating_val = query.with_entities(func.avg(Rating.overall_rating)).scalar()
    avg_food_val = query.with_entities(func.avg(Rating.food_rating)).scalar()
    avg_service_val = query.with_entities(func.avg(Rating.service_rating)).scalar()
    five_star_count = query.filter(Rating.overall_rating == 5).count()
    five_star_pct = round((five_star_count / total_count) * 100, 1) if total_count > 0 else 0.0

    summary = {
        "average_rating": round(float(avg_rating_val), 2) if avg_rating_val is not None else None,
        "rating_count": total_count,
        "star_distribution": [{"stars":stars,"count":query.filter(Rating.overall_rating == stars).count()} for stars in range(1,6)],
        "five_star_pct": five_star_pct,
        "avg_food_score": round(float(avg_food_val), 2) if avg_food_val is not None else None,
        "avg_service_score": round(float(avg_service_val), 2) if avg_service_val is not None else None,
        "csat_pct": round(query.filter(Rating.overall_rating >= 4).count() / total_count * 100, 1) if total_count else None
    }

    query = query.order_by(desc(Rating.review_date), desc(Rating.rating_id))
    offset = (page - 1) * page_size
    records = query.offset(offset).limit(page_size).all()

    loc_ids = {r.location_id for r in records if r.location_id}
    locs_map = {}
    if loc_ids:
        locs = db.query(Restaurant).filter(Restaurant.location_id.in_(loc_ids)).all()
        for l in locs:
            locs_map[l.location_id] = l.name

    item_ids = {r.item_id for r in records if r.item_id}
    items_map = {}
    if item_ids:
        mis = db.query(MenuItem).filter(MenuItem.item_id.in_(item_ids)).all()
        for mi in mis:
            items_map[mi.item_id] = mi.name

    ratings_list = []
    for r in records:
        ratings_list.append({
            "rating_id": r.rating_id,
            "order_id": r.order_id or "N/A",
            "customer_id": r.customer_id or "Anonymous Guest",
            "item_id": r.item_id or "General",
            "item_name": items_map.get(r.item_id, r.item_id or "Overall Experience"),
            "location_id": r.location_id,
            "location_name": locs_map.get(r.location_id, r.location_id),
            "overall_rating": r.overall_rating,
            "food_rating": r.food_rating,
            "service_rating": r.service_rating,
            "ambiance_rating": r.ambiance_rating,
            "review_text": r.review_text or "No comment provided.",
            "review_date": str(r.review_date) if r.review_date else "Recent"
        })

    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1

    return {
        "summary": summary,
        "ratings": ratings_list,
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "total_pages": total_pages
    }
