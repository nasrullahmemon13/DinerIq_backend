"""
DineIQ Analytics - Restaurant Location Management Router
Implements SRS Step 2 / Functional Requirement (iii) Restaurant Location Management:
- Complete dynamic restaurant listing with operational performance metrics (Orders, Revenue, AOV, Rating)
- Summary KPIs (Total Locations, Active Locations, Total Orders, Total Revenue)
- Location Details view with Overview, Performance KPIs, Menu Breakdown, and Channel Distribution
- Admin & Regional Manager CRUD endpoints (Create, Update, Status Toggle, Delete) with DB persistence
"""

import uuid
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import Restaurant
from backend.services.dashboard_service import DashboardService
from src.routes import get_current_user

router = APIRouter(prefix="/api/v1/restaurants", tags=["Restaurant Location Management"])


class RestaurantCreateSchema(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    city: str = Field(..., min_length=2, max_length=50)
    state: str = Field(..., min_length=2, max_length=20)
    country: Optional[str] = "USA"
    location_id: Optional[str] = None
    restaurant_id: Optional[str] = None
    operating_status: Optional[str] = "ACTIVE"
    seating_capacity: Optional[int] = Field(120, ge=1, le=2000)
    manager_name: Optional[str] = None
    phone_number: Optional[str] = None
    cost_index: Optional[float] = 1.0
    location_tier: Optional[str] = "Standard"
    has_drive_thru: Optional[bool] = False
    has_outdoor_seating: Optional[bool] = False


class RestaurantUpdateSchema(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    operating_status: Optional[str] = None
    seating_capacity: Optional[int] = None
    manager_name: Optional[str] = None
    phone_number: Optional[str] = None
    cost_index: Optional[float] = None
    location_tier: Optional[str] = None
    has_drive_thru: Optional[bool] = None
    has_outdoor_seating: Optional[bool] = None


class StatusToggleSchema(BaseModel):
    operating_status: str = Field(..., description="ACTIVE or INACTIVE")


def require_restaurant_management_role(current_user: Dict[str, Any] = Depends(get_current_user)):
    role = current_user.get("role_id")
    if role not in ["admin", "regional_manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access forbidden: role '{role}' is not authorized to manage restaurant locations. Required: admin or regional_manager."
        )
    return current_user


@router.get("")
def list_restaurants(
    search: Optional[str] = Query(None, description="Search query across name, city, location ID, or manager"),
    city: Optional[str] = Query(None, description="Filter by city name"),
    status_filter: Optional[str] = Query(None, description="Filter by operating status (e.g. ACTIVE, INACTIVE, ALL)"),
    sort_by: Optional[str] = Query("revenue_desc", description="Sort option: revenue_desc, revenue_asc, orders_desc, name_asc, rating_desc"),
    db: Session = Depends(get_db)
):
    """
    List all restaurant locations with operational performance metrics merged from analytics engine.
    Also provides dynamic summary KPIs (Total Locations, Active Locations, Total Orders, Total Revenue).
    """
    service = DashboardService()
    loc_matrix = service.loc_matrix

    db_query = db.query(Restaurant)
    all_restaurants = db_query.all()

    matrix_by_loc = {}
    if not loc_matrix.empty and "location_id" in loc_matrix.columns:
        for _, row in loc_matrix.iterrows():
            loc_id = str(row["location_id"]).strip()
            matrix_by_loc[loc_id] = row.to_dict()

    items = []
    total_revenue_sum = 0.0
    total_orders_sum = 0
    active_count = 0
    distinct_cities = set()

    for r in all_restaurants:
        loc_id = r.location_id or r.restaurant_id
        op_status = (r.operating_status or "ACTIVE").upper()
        if op_status == "ACTIVE":
            active_count += 1

        if r.city:
            distinct_cities.add(r.city)

        m = matrix_by_loc.get(loc_id, {})
        rev = round(float(m.get("total_revenue", 0.0)), 2)
        orders = int(m.get("total_orders", 0))
        aov = round(float(m.get("average_order_value", 0.0)), 2)
        rating = round(float(m.get("avg_overall_rating", 4.2)), 2)
        gross_profit = round(float(m.get("gross_profit", 0.0)), 2)
        wastage_loss = round(float(m.get("total_wastage_loss_amount", 0.0)), 2)
        unique_customers = int(m.get("unique_customer_count", 0))
        csat_pct = round(float(m.get("csat_pct", 0.0)), 1)
        margin_pct = round(float(m.get("contribution_margin_pct", 55.0)), 1)
        tier = str(m.get("location_tier", r.location_tier or "Standard"))

        total_revenue_sum += rev
        total_orders_sum += orders

        item = {
            "restaurant_id": r.restaurant_id,
            "location_id": loc_id,
            "name": r.name,
            "city": r.city,
            "state": r.state,
            "country": r.country or "USA",
            "seating_capacity": r.seating_capacity,
            "cost_index": r.cost_index,
            "operating_status": op_status,
            "manager_name": r.manager_name or "Unassigned",
            "phone_number": r.phone_number or "N/A",
            "location_tier": tier,
            "has_drive_thru": bool(r.has_drive_thru),
            "has_outdoor_seating": bool(r.has_outdoor_seating),
            "total_revenue": rev,
            "total_orders": orders,
            "average_order_value": aov,
            "avg_overall_rating": rating,
            "gross_profit": gross_profit,
            "total_wastage_loss_amount": wastage_loss,
            "unique_customer_count": unique_customers,
            "csat_pct": csat_pct,
            "margin_pct": margin_pct,
        }
        items.append(item)

    filtered = items

    if status_filter and status_filter.upper() not in ["ALL", ""]:
        filtered = [x for x in filtered if x["operating_status"] == status_filter.upper()]

    if city and city.upper() not in ["ALL", ""]:
        filtered = [x for x in filtered if x["city"].lower() == city.lower()]

    if search:
        s = search.strip().lower()
        filtered = [
            x for x in filtered
            if s in x["name"].lower()
            or s in x["city"].lower()
            or s in x["location_id"].lower()
            or s in (x["manager_name"] or "").lower()
        ]

    if sort_by == "revenue_desc":
        filtered.sort(key=lambda x: x["total_revenue"], reverse=True)
    elif sort_by == "revenue_asc":
        filtered.sort(key=lambda x: x["total_revenue"])
    elif sort_by == "orders_desc":
        filtered.sort(key=lambda x: x["total_orders"], reverse=True)
    elif sort_by == "name_asc":
        filtered.sort(key=lambda x: x["name"].lower())
    elif sort_by == "rating_desc":
        filtered.sort(key=lambda x: x["avg_overall_rating"], reverse=True)

    summary = {
        "total_locations": len(all_restaurants),
        "active_locations": active_count,
        "total_orders": total_orders_sum,
        "total_revenue": round(total_revenue_sum, 2)
    }

    return {
        "summary": summary,
        "restaurants": filtered,
        "cities": sorted(list(distinct_cities))
    }


@router.get("/{location_id}")
def get_restaurant_detail(
    location_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieve comprehensive details, operational KPIs, top menu items, and channel breakdown
    for a specific restaurant location.
    """
    loc_clean = location_id.strip()

    rest = db.query(Restaurant).filter(
        (Restaurant.location_id == loc_clean) | (Restaurant.restaurant_id == loc_clean)
    ).first()

    if not rest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Restaurant location '{loc_clean}' not found"
        )

    canonical_loc_id = rest.location_id

    service = DashboardService()
    loc_matrix = service.loc_matrix
    lmp = service.lmp
    orders_df = service.orders

    perf = {}
    if not loc_matrix.empty and "location_id" in loc_matrix.columns:
        match = loc_matrix[loc_matrix["location_id"] == canonical_loc_id]
        if not match.empty:
            perf = match.iloc[0].to_dict()

    menu_items = []
    if not lmp.empty and "location_id" in lmp.columns:
        sub_lmp = lmp[lmp["location_id"] == canonical_loc_id]
        if not sub_lmp.empty:
            sorted_items = sub_lmp.sort_values("revenue", ascending=False).head(10)
            for _, r in sorted_items.iterrows():
                menu_items.append({
                    "item_id": str(r.get("item_id", "")),
                    "dish_name": str(r.get("item_name", "")),
                    "category_name": str(r.get("category_name", "General")),
                    "quantity_sold": int(r.get("quantity_sold", 0)),
                    "revenue": round(float(r.get("revenue", 0.0)), 2),
                    "cost": round(float(r.get("cost", 0.0)), 2),
                    "gross_profit": round(float(r.get("gross_profit", 0.0)), 2),
                    "margin_pct": round(float(r.get("contribution_margin_pct", 50.0)), 1),
                    "avg_rating": round(float(r["avg_rating"]), 1) if pd.notna(r.get("avg_rating")) else None,
                    "wastage_cost": round(float(r.get("wastage_cost", 0.0)), 2),
                    "classification": str(r.get("location_menu_classification", "Performer"))
                })

    channels = []
    if not orders_df.empty and "location_id" in orders_df.columns:
        sub_orders = orders_df[orders_df["location_id"] == canonical_loc_id]
        if not sub_orders.empty and "order_type" in sub_orders.columns:
            tot_sub_rev = float(sub_orders["total_amount"].sum()) or 1.0
            tot_sub_orders = len(sub_orders) or 1
            grouped = sub_orders.groupby("order_type").agg(
                orders=("order_id", "count"),
                revenue=("total_amount", "sum")
            ).reset_index()

            for _, row in grouped.iterrows():
                ch_name = str(row["order_type"]).replace("_", " ").title()
                rev = round(float(row["revenue"]), 2)
                cnt = int(row["orders"])
                channels.append({
                    "channel": ch_name,
                    "orders": cnt,
                    "revenue": rev,
                    "share_pct": round((rev / tot_sub_rev) * 100, 1),
                    "order_pct": round((cnt / tot_sub_orders) * 100, 1)
                })
            channels.sort(key=lambda x: x["revenue"], reverse=True)

    return {
        "restaurant": {
            "restaurant_id": rest.restaurant_id,
            "location_id": rest.location_id,
            "name": rest.name,
            "city": rest.city,
            "state": rest.state,
            "country": rest.country or "USA",
            "postal_code": rest.postal_code or "N/A",
            "operating_status": (rest.operating_status or "ACTIVE").upper(),
            "manager_name": rest.manager_name or "Unassigned",
            "phone_number": rest.phone_number or "N/A",
            "seating_capacity": rest.seating_capacity,
            "cost_index": rest.cost_index,
            "location_tier": str(perf.get("location_tier", rest.location_tier or "Standard")),
            "has_drive_thru": bool(rest.has_drive_thru),
            "has_outdoor_seating": bool(rest.has_outdoor_seating),
            "opened_date": str(rest.opened_date) if rest.opened_date else None,
        },
        "performance": {
            "total_revenue": round(float(perf.get("total_revenue", 0.0)), 2),
            "gross_profit": round(float(perf.get("gross_profit", 0.0)), 2),
            "total_orders": int(perf.get("total_orders", 0)),
            "average_order_value": round(float(perf.get("average_order_value", 0.0)), 2),
            "unique_customer_count": int(perf.get("unique_customer_count", 0)),
            "total_wastage_loss_amount": round(float(perf.get("total_wastage_loss_amount", 0.0)), 2),
            "quantity_wasted": int(perf.get("quantity_wasted", 0)),
            "avg_overall_rating": round(float(perf.get("avg_overall_rating", 4.2)), 2),
            "avg_food_rating": round(float(perf.get("avg_food_rating", 4.1)), 2),
            "avg_service_rating": round(float(perf.get("avg_service_rating", 4.1)), 2),
            "avg_ambiance_rating": round(float(perf.get("avg_ambiance_rating", 4.2)), 2),
            "csat_pct": round(float(perf.get("csat_pct", 75.0)), 1),
            "repeat_purchase_rate": round(float(perf.get("repeat_purchase_rate", 5.5)), 1),
            "contribution_margin_pct": round(float(perf.get("contribution_margin_pct", 55.0)), 1),
            "revenue_rank": int(perf.get("revenue_rank", 0)),
            "profit_rank": int(perf.get("profit_rank", 0))
        },
        "menu_breakdown": menu_items,
        "channel_distribution": channels
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def create_restaurant(
    payload: RestaurantCreateSchema,
    current_user: Dict[str, Any] = Depends(require_restaurant_management_role),
    db: Session = Depends(get_db)
):
    """
    Create a new restaurant location (Admin / Regional Manager).
    Persists directly to database table `restaurants`.
    """
    if not payload.location_id:
        existing_count = db.query(Restaurant).count()
        new_loc_id = f"LOC-{str(existing_count + 1).zfill(3)}"
    else:
        new_loc_id = payload.location_id.strip()

    existing = db.query(Restaurant).filter(
        (Restaurant.location_id == new_loc_id) | (Restaurant.name == payload.name.strip())
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Restaurant with location ID '{new_loc_id}' or name '{payload.name}' already exists."
        )

    rest_id = payload.restaurant_id or f"REST-{uuid.uuid4().hex[:6].upper()}"

    new_restaurant = Restaurant(
        restaurant_id=rest_id,
        location_id=new_loc_id,
        name=payload.name.strip(),
        city=payload.city.strip(),
        state=payload.state.strip(),
        country=payload.country or "USA",
        operating_status=(payload.operating_status or "ACTIVE").upper(),
        seating_capacity=payload.seating_capacity or 120,
        manager_name=payload.manager_name.strip() if payload.manager_name else None,
        phone_number=payload.phone_number.strip() if payload.phone_number else None,
        cost_index=payload.cost_index or 1.0,
        location_tier=payload.location_tier or "Standard",
        has_drive_thru=bool(payload.has_drive_thru),
        has_outdoor_seating=bool(payload.has_outdoor_seating)
    )

    db.add(new_restaurant)
    db.commit()
    db.refresh(new_restaurant)

    return {
        "message": "Restaurant location created successfully",
        "restaurant": {
            "restaurant_id": new_restaurant.restaurant_id,
            "location_id": new_restaurant.location_id,
            "name": new_restaurant.name,
            "city": new_restaurant.city,
            "state": new_restaurant.state,
            "operating_status": new_restaurant.operating_status,
            "seating_capacity": new_restaurant.seating_capacity,
            "manager_name": new_restaurant.manager_name,
            "phone_number": new_restaurant.phone_number
        }
    }


@router.put("/{location_id}")
def update_restaurant(
    location_id: str,
    payload: RestaurantUpdateSchema,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update restaurant details (Admin, Regional Manager, or Store Manager for assigned store).
    """
    role = current_user.get("role_id")
    assigned_loc = current_user.get("assigned_location_id")

    if role == "manager":
        if not assigned_loc or assigned_loc.strip() != location_id.strip():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Store Managers can only edit their assigned store ({assigned_loc})."
            )
    elif role not in ["admin", "regional_manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{role}' is not authorized to update restaurant locations."
        )

    loc_clean = location_id.strip()
    rest = db.query(Restaurant).filter(
        (Restaurant.location_id == loc_clean) | (Restaurant.restaurant_id == loc_clean)
    ).first()

    if not rest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Restaurant '{loc_clean}' not found."
        )

    data = payload.model_dump(exclude_unset=True) if hasattr(payload, "model_dump") else payload.dict(exclude_unset=True)
    for field, val in data.items():
        if val is not None:
            if field == "operating_status":
                setattr(rest, field, str(val).upper())
            else:
                setattr(rest, field, val)

    db.commit()
    db.refresh(rest)

    return {
        "message": "Restaurant location updated successfully",
        "location_id": rest.location_id,
        "name": rest.name,
        "operating_status": rest.operating_status
    }


@router.patch("/{location_id}/status")
def toggle_restaurant_status(
    location_id: str,
    payload: StatusToggleSchema,
    current_user: Dict[str, Any] = Depends(require_restaurant_management_role),
    db: Session = Depends(get_db)
):
    """
    Toggle operating status (ACTIVE <-> INACTIVE) (Admin / Regional Manager).
    """
    loc_clean = location_id.strip()
    rest = db.query(Restaurant).filter(
        (Restaurant.location_id == loc_clean) | (Restaurant.restaurant_id == loc_clean)
    ).first()

    if not rest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Restaurant '{loc_clean}' not found."
        )

    new_status = payload.operating_status.strip().upper()
    if new_status not in ["ACTIVE", "INACTIVE", "UNDER_RENOVATION"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Status must be ACTIVE, INACTIVE, or UNDER_RENOVATION"
        )

    rest.operating_status = new_status
    db.commit()
    db.refresh(rest)

    return {
        "message": f"Restaurant status updated to {new_status}",
        "location_id": rest.location_id,
        "operating_status": rest.operating_status
    }


@router.delete("/{location_id}")
def delete_restaurant(
    location_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Deactivate or remove restaurant location (Admin only).
    """
    if current_user.get("role_id") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can delete or de-register restaurant locations."
        )

    loc_clean = location_id.strip()
    rest = db.query(Restaurant).filter(
        (Restaurant.location_id == loc_clean) | (Restaurant.restaurant_id == loc_clean)
    ).first()

    if not rest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Restaurant '{loc_clean}' not found."
        )

    rest.operating_status = "INACTIVE"
    db.commit()

    return {
        "message": f"Restaurant '{rest.name}' ({rest.location_id}) deactivated successfully.",
        "location_id": rest.location_id,
        "operating_status": rest.operating_status
    }
