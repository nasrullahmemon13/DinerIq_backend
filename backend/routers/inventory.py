"""
DineIQ Analytics - Inventory Management Router
Implements SRS Step 1 / Functional Requirement (x) Inventory Management:
- Inventory usage, stock levels, replenishment status, and item consumption
- Real-time stock status calculations (Optimal, Low Stock Warning, Critical Stockout)
- Snapshot recording and stock replenishment adjustments
"""

from datetime import date
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from database.connection import get_db
from database.models import Inventory, MenuItem, Restaurant
from src.routes import get_current_user, require_roles

router = APIRouter(prefix="/api/v1/inventory", tags=["Inventory Management (FR-x)"])


class RestockPayload(BaseModel):
    inventory_id: str
    quantity_added: int = Field(..., ge=1, description="Quantity to add to inventory")


@router.get("/stock")
def list_inventory_stock(
    location_id: Optional[str] = Query(None, description="Filter by location ID"),
    stock_status: Optional[str] = Query(None, description="ALL, LOW, CRITICAL, ADEQUATE"),
    search: Optional[str] = Query(None, description="Search item name or ID"),
    db: Session = Depends(get_db)
):
    """
    Retrieve inventory stock levels with dynamic status indicators
    and replenishment thresholds.
    """
    query = db.query(Inventory)

    if location_id and location_id.upper() not in ("ALL", ""):
        query = query.filter(Inventory.location_id == location_id.strip())

    items = query.all()

    item_ids = {i.item_id for i in items if i.item_id}
    mis_map = {}
    if item_ids:
        mis = db.query(MenuItem).filter(MenuItem.item_id.in_(item_ids)).all()
        for m in mis:
            mis_map[m.item_id] = {
                "name": m.name,
                "category_id": m.category_id,
                "shelf_life_days": m.shelf_life_days or 3
            }

    loc_ids = {i.location_id for i in items if i.location_id}
    locs_map = {}
    if loc_ids:
        locs = db.query(Restaurant).filter(Restaurant.location_id.in_(loc_ids)).all()
        for l in locs:
            locs_map[l.location_id] = l.name

    adequate_count = 0
    low_stock_count = 0
    critical_count = 0

    enriched = []
    for inv in items:
        end_stock = inv.ending_stock
        reorder_pt = inv.reorder_point or 15

        if end_stock <= 0:
            status_calc = "OUT_OF_STOCK"
            critical_count += 1
        elif end_stock <= reorder_pt:
            status_calc = "LOW_STOCK"
            low_stock_count += 1
        else:
            status_calc = "ADEQUATE"
            adequate_count += 1

        m_info = mis_map.get(inv.item_id, {})
        item_name = m_info.get("name", inv.item_id)
        loc_name = locs_map.get(inv.location_id, inv.location_id)

        enriched.append({
            "inventory_id": inv.inventory_id,
            "item_id": inv.item_id,
            "item_name": item_name,
            "category_id": m_info.get("category_id", "General"),
            "shelf_life_days": m_info.get("shelf_life_days", 3),
            "location_id": inv.location_id,
            "location_name": loc_name,
            "snapshot_date": str(inv.snapshot_date) if inv.snapshot_date else str(date.today()),
            "starting_stock": inv.starting_stock,
            "quantity_received": inv.quantity_received or 0,
            "quantity_sold": inv.quantity_sold or 0,
            "quantity_wasted": inv.quantity_wasted or 0,
            "ending_stock": end_stock,
            "reorder_point": reorder_pt,
            "stock_status": status_calc
        })

    filtered = enriched
    if stock_status and stock_status.upper() not in ("ALL", ""):
        st_upper = stock_status.upper()
        if st_upper in ("LOW", "LOW_STOCK"):
            filtered = [x for x in filtered if x["stock_status"] in ("LOW_STOCK", "OUT_OF_STOCK")]
        elif st_upper in ("CRITICAL", "OUT_OF_STOCK"):
            filtered = [x for x in filtered if x["stock_status"] == "OUT_OF_STOCK"]
        elif st_upper in ("ADEQUATE", "OPTIMAL"):
            filtered = [x for x in filtered if x["stock_status"] == "ADEQUATE"]

    if search and search.strip():
        s = search.strip().lower()
        filtered = [x for x in filtered if s in x["item_name"].lower() or s in x["item_id"].lower() or s in x["location_name"].lower()]

    summary = {
        "total_monitored_items": len(enriched),
        "adequate_stock_count": adequate_count,
        "low_stock_warnings": low_stock_count,
        "critical_stockouts": critical_count
    }

    return {
        "summary": summary,
        "inventory": filtered
    }


@router.post("/restock")
def restock_inventory_item(
    payload: RestockPayload,
    current_user: Dict[str, Any] = Depends(require_roles(["admin", "manager", "regional_manager"])),
    db: Session = Depends(get_db)
):
    """
    Replenish stock for an inventory item (Admin, Store Manager, Regional Manager).
    """
    clean_id = payload.inventory_id.strip()
    inv = db.query(Inventory).filter(Inventory.inventory_id == clean_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail=f"Inventory record '{clean_id}' not found.")

    qty = payload.quantity_added
    inv.quantity_received = (inv.quantity_received or 0) + qty
    inv.ending_stock = (inv.ending_stock or 0) + qty

    reorder_pt = inv.reorder_point or 15
    if inv.ending_stock <= 0:
        inv.stock_status = "Out of Stock"
    elif inv.ending_stock <= reorder_pt:
        inv.stock_status = "Low Stock"
    else:
        inv.stock_status = "Adequate"

    db.commit()
    db.refresh(inv)

    return {
        "message": f"Successfully replenished {qty} units for inventory item {clean_id}.",
        "inventory_id": inv.inventory_id,
        "new_ending_stock": inv.ending_stock,
        "stock_status": inv.stock_status
    }
