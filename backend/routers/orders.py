"""
DineIQ Analytics - Orders Management Router
Implements SRS Step 1 & Step 6 / Functional Requirement (vii) Order Management:
- Server-side paginated orders explorer with search, date range, location, channel, and status filters
- Dynamic business summary cards (Total Orders, Completed Orders, Cancelled Orders, Total Revenue, Average Order Value)
- Detailed order header and itemized order lines drill-down
- Order status update (e.g. COMPLETED, CANCELLED, PENDING, PREPARING)
"""

import math
from typing import Optional, List, Dict, Any
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, asc

from database.connection import get_db
from database.models import Order, OrderItem, Restaurant, MenuItem, Customer, Promotion
from src.routes import get_current_user, require_roles

router = APIRouter(prefix="/api/v1/orders", tags=["Order Management (FR-vii)"])


class OrderStatusUpdate(BaseModel):
    order_status: str = Field(..., description="COMPLETED, CANCELLED, PENDING, PREPARING")


@router.get("/paginated")
def list_orders_paginated(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=5, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search Order ID, Customer ID, or Location Name"),
    location_id: Optional[str] = Query(None, description="Filter by location ID"),
    channel: Optional[str] = Query(None, description="Filter by ordering channel (e.g. DINE_IN, TAKEOUT, DELIVERY, DRIVE_THRU)"),
    status_filter: Optional[str] = Query(None, description="Filter by status (COMPLETED, CANCELLED, PENDING)"),
    date_from: Optional[str] = Query(None, description="Filter orders from date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter orders to date (YYYY-MM-DD)"),
    sort_by: Optional[str] = Query("date_desc", description="Sorting: date_desc, date_asc, amount_desc, amount_asc"),
    db: Session = Depends(get_db)
):
    """
    Retrieve server-side paginated orders with dynamic summary counters,
    restaurant location names, and order-line counts.
    """
    query = db.query(Order)

    if location_id and location_id.upper() not in ("ALL", ""):
        query = query.filter(Order.location_id == location_id.strip())

    if channel and channel.upper() not in ("ALL", ""):
        query = query.filter(func.upper(Order.order_type) == channel.strip().upper())

    if status_filter and status_filter.upper() not in ("ALL", ""):
        query = query.filter(func.upper(Order.order_status) == status_filter.strip().upper())

    if date_from:
        try:
            d_from = date.fromisoformat(date_from.strip())
            query = query.filter(Order.order_date >= d_from)
        except ValueError:
            pass

    if date_to:
        try:
            d_to = date.fromisoformat(date_to.strip())
            query = query.filter(Order.order_date <= d_to)
        except ValueError:
            pass

    if search and search.strip():
        s = f"%{search.strip().lower()}%"
        query = query.filter(
            (func.lower(Order.order_id).like(s)) |
            (func.lower(Order.customer_id).like(s)) |
            (func.lower(Order.location_id).like(s))
        )

    total_count = query.count()

    completed_count = query.filter(func.upper(Order.order_status) == "COMPLETED").count()
    cancelled_count = query.filter(func.upper(Order.order_status) == "CANCELLED").count()

    rev_result = query.with_entities(func.sum(Order.total_amount)).scalar() or 0.0
    total_revenue = round(float(rev_result), 2)
    avg_order_val = round(total_revenue / total_count, 2) if total_count > 0 else 0.0

    summary = {
        "total_orders": total_count,
        "channel_distribution": [{"channel":name or "Unknown","count":int(count)} for name,count in query.with_entities(Order.order_type,func.count(Order.order_id)).group_by(Order.order_type).all()],
        "completed_orders": completed_count,
        "cancelled_orders": cancelled_count,
        "total_revenue": total_revenue,
        "average_order_value": avg_order_val
    }

    if sort_by == "date_asc":
        query = query.order_by(asc(Order.order_date), asc(Order.order_time))
    elif sort_by == "amount_desc":
        query = query.order_by(desc(Order.total_amount))
    elif sort_by == "amount_asc":
        query = query.order_by(asc(Order.total_amount))
    else:             
        query = query.order_by(desc(Order.order_date), desc(Order.order_time))

    offset = (page - 1) * page_size
    records = query.offset(offset).limit(page_size).all()

    loc_ids = {r.location_id for r in records if r.location_id}
    locations_map = {}
    if loc_ids:
        locs = db.query(Restaurant).filter(Restaurant.location_id.in_(loc_ids)).all()
        for loc in locs:
            locations_map[loc.location_id] = loc.name

    order_ids = [r.order_id for r in records]
    item_counts_map = {}
    if order_ids:
        item_counts = db.query(
            OrderItem.order_id, func.count(OrderItem.order_item_id)
        ).filter(OrderItem.order_id.in_(order_ids)).group_by(OrderItem.order_id).all()
        for oid, count in item_counts:
            item_counts_map[oid] = count

    orders_list = []
    for o in records:
        orders_list.append({
            "order_id": o.order_id,
            "order_date": str(o.order_date),
            "order_time": o.order_time or "12:00:00",
            "customer_id": o.customer_id,
            "location_id": o.location_id,
            "location_name": locations_map.get(o.location_id, o.location_id),
            "order_type": o.order_type or "DINE_IN",
            "order_status": (o.order_status or "COMPLETED").upper(),
            "payment_method": o.payment_method or "Credit Card",
            "item_count": item_counts_map.get(o.order_id, 1),
            "subtotal_amount": round(float(o.subtotal_amount or 0.0), 2),
            "discount_amount": round(float(o.discount_amount or 0.0), 2),
            "total_amount": round(float(o.total_amount or 0.0), 2),
            "promotion_id": o.promotion_id or None,
            "table_number": o.table_number
        })

    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1

    return {
        "summary": summary,
        "orders": orders_list,
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "total_pages": total_pages
    }


@router.get("/{order_id}/details")
def get_order_details(order_id: str, db: Session = Depends(get_db)):
    """
    Retrieve comprehensive order header and itemized order lines.
    """
    clean_id = order_id.strip()
    order = db.query(Order).filter(Order.order_id == clean_id).first()
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order '{clean_id}' not found."
        )

    location = db.query(Restaurant).filter(Restaurant.location_id == order.location_id).first()
    loc_name = location.name if location else order.location_id
    loc_city = location.city if location else ""

    promo_name = None
    if order.promotion_id:
        p = db.query(Promotion).filter(Promotion.promotion_id == order.promotion_id).first()
        if p:
            promo_name = p.promotion_name

    order_items = db.query(OrderItem).filter(OrderItem.order_id == clean_id).all()
    item_ids = [itm.item_id for itm in order_items]
    menu_items_map = {}
    if item_ids:
        mis = db.query(MenuItem).filter(MenuItem.item_id.in_(item_ids)).all()
        for mi in mis:
            menu_items_map[mi.item_id] = {
                "name": mi.name,
                "category_id": mi.category_id,
                "base_price": mi.base_price
            }

    lines = []
    for itm in order_items:
        m_info = menu_items_map.get(itm.item_id, {})
        lines.append({
            "order_item_id": itm.order_item_id,
            "item_id": itm.item_id,
            "item_name": m_info.get("name", itm.item_id),
            "quantity": itm.quantity,
            "unit_price": round(float(itm.unit_price or 0.0), 2),
            "discount": round(float(itm.item_discount or 0.0), 2),
            "line_total": round(float(itm.item_total or 0.0), 2)
        })

    return {
        "header": {
            "order_id": order.order_id,
            "order_date": str(order.order_date),
            "order_time": order.order_time or "12:00:00",
            "customer_id": order.customer_id,
            "location_id": order.location_id,
            "location_name": loc_name,
            "location_city": loc_city,
            "order_type": order.order_type or "DINE_IN",
            "order_status": (order.order_status or "COMPLETED").upper(),
            "payment_method": order.payment_method or "Credit Card",
            "promotion_id": order.promotion_id,
            "promotion_name": promo_name,
            "table_number": order.table_number,
            "subtotal_amount": round(float(order.subtotal_amount or 0.0), 2),
            "discount_amount": round(float(order.discount_amount or 0.0), 2),
            "tax_amount": round(float(order.tax_amount or 0.0), 2),
            "tip_amount": round(float(order.tip_amount or 0.0), 2),
            "delivery_fee": round(float(order.delivery_fee or 0.0), 2),
            "total_amount": round(float(order.total_amount or 0.0), 2),
        },
        "order_lines": lines
    }


@router.patch("/{order_id}/status")
def update_order_status(
    order_id: str,
    payload: OrderStatusUpdate,
    current_user: Dict[str, Any] = Depends(require_roles(["admin", "manager", "regional_manager"])),
    db: Session = Depends(get_db)
):
    """Update order lifecycle status (Admin, Manager, Regional Manager)."""
    clean_id = order_id.strip()
    order = db.query(Order).filter(Order.order_id == clean_id).first()
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order '{clean_id}' not found."
        )

    new_st = payload.order_status.strip().upper()
    valid_statuses = ["COMPLETED", "CANCELLED", "PENDING", "PREPARING"]
    if new_st not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Status must be one of: {valid_statuses}"
        )

    order.order_status = new_st
    db.commit()
    db.refresh(order)

    return {
        "message": f"Order '{clean_id}' status updated to {new_st}",
        "order_id": clean_id,
        "order_status": order.order_status
    }
