"""
DineIQ Analytics - Promotions Management Router
Implements SRS Step 1 & Step 27-28 / Functional Requirement (viii) Promotion Management:
- Promotional campaign list with analytics performance metrics merged from Spark evaluations
- Dynamic summary cards (Total Promotions, Active Promotions, Promoted Orders, Promoted Revenue)
- Promotion creation, update, and deactivation with DB persistence
- Campaign detail drawer data including definition, applicable items, usage, revenue, profit, customer behavior, and wastage impact
"""

import os
from datetime import date
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import pandas as pd
from backend.services.data_store import read_frame, data_exists

from database.connection import get_db
from database.models import Promotion, MenuItem
from src.routes import get_current_user, require_roles

router = APIRouter(prefix="/api/v1/promotions", tags=["Promotion Management (FR-viii)"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_PATH = os.path.join(PROJECT_ROOT, "processed_data", "promotion", "promotion_evaluations.parquet")
TRAP_PATH = os.path.join(PROJECT_ROOT, "processed_data", "promotion", "promotion_trap_detection.parquet")


class PromotionCreateSchema(BaseModel):
    promotion_name: str = Field(..., min_length=2, max_length=100)
    discount_type: str = Field(..., description="PERCENTAGE, FIXED_AMOUNT, BOGO")
    discount_value: float = Field(..., ge=0.0)
    min_order_amount: Optional[float] = 0.0
    max_discount_amount: Optional[float] = 50.0
    applicable_category: Optional[str] = "ALL"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    target_segment: Optional[str] = "ALL"
    description: Optional[str] = None
    is_misleading: Optional[bool] = False


class PromotionUpdateSchema(BaseModel):
    promotion_name: Optional[str] = None
    discount_type: Optional[str] = None
    discount_value: Optional[float] = None
    min_order_amount: Optional[float] = None
    max_discount_amount: Optional[float] = None
    applicable_category: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    target_segment: Optional[str] = None
    description: Optional[str] = None
    is_misleading: Optional[bool] = None


@router.get("/analytics")
def list_promotions_with_analytics(
    status_filter: Optional[str] = Query(None, description="ALL, ACTIVE, EXPIRED"),
    search: Optional[str] = Query(None, description="Search campaign name or ID"),
    db: Session = Depends(get_db)
):
    """
    Retrieve all promotional campaigns merged with actual Spark analytical performance
    metrics (Orders, Revenue, Contribution Margin, Wastage impact, and Traps).
    """
    db_promotions = db.query(Promotion).all()

    eval_dict = {}
    if data_exists(EVAL_PATH):
        try:
            df = read_frame(EVAL_PATH)
            for _, r in df.iterrows():
                pid = str(r["promotion_id"]).strip()
                eval_dict[pid] = r.to_dict()
        except Exception as e:
            print(f"[Error] Failed to read promotion evaluations: {e}")

    today = date.today()
    total_promo_count = len(db_promotions)
    active_count = 0
    total_orders_sum = 0
    total_rev_sum = 0.0

    items = []
    for p in db_promotions:
        is_active = True
        if p.end_date and p.end_date < today:
            is_active = False
        if p.start_date and p.start_date > today:
            is_active = False

        if is_active:
            active_count += 1

        ev = eval_dict.get(p.promotion_id, {})
        vol = int(ev.get("order_volume", 0))
        rev = round(float(ev.get("promotional_revenue", 0.0)), 2)
        cm = round(float(ev.get("contribution_margin_dollars", 0.0)), 2)
        margin_pct = round(float(ev.get("profit_margin_pct", 52.0)), 1)
        aov = round(float(ev.get("average_order_value", 0.0)), 2)
        wasted_loss = round(float(ev.get("wasted_loss_amount", 0.0)), 2)
        repeat_rate = round(float(ev.get("repeat_purchase_rate_pct", 0.0)), 1)
        eval_status = str(ev.get("evaluation_status", "ACTIVE_HEALTHY"))
        rationale = str(ev.get("status_rationale", "Normal campaign execution."))

        total_orders_sum += vol
        total_rev_sum += rev

        items.append({
            "promotion_id": p.promotion_id,
            "promotion_name": p.promotion_name,
            "discount_type": p.discount_type,
            "discount_value": p.discount_value,
            "min_order_amount": p.min_order_amount,
            "max_discount_amount": p.max_discount_amount,
            "applicable_category": p.applicable_category or "ALL",
            "start_date": str(p.start_date) if p.start_date else None,
            "end_date": str(p.end_date) if p.end_date else None,
            "target_segment": p.target_segment or "ALL",
            "description": p.description or "",
            "is_misleading": bool(p.is_misleading),
            "is_active": is_active,
            "order_volume": vol,
            "promotional_revenue": rev,
            "contribution_margin_dollars": cm,
            "profit_margin_pct": margin_pct,
            "average_order_value": aov,
            "wasted_loss_amount": wasted_loss,
            "repeat_purchase_rate_pct": repeat_rate,
            "evaluation_status": eval_status,
            "status_rationale": rationale
        })

    filtered = items
    if status_filter and status_filter.upper() == "ACTIVE":
        filtered = [x for x in filtered if x["is_active"]]
    elif status_filter and status_filter.upper() == "EXPIRED":
        filtered = [x for x in filtered if not x["is_active"]]

    if search and search.strip():
        s = search.strip().lower()
        filtered = [x for x in filtered if s in x["promotion_name"].lower() or s in x["promotion_id"].lower()]

    summary = {
        "total_promotions": total_promo_count,
        "active_promotions": active_count,
        "promotion_orders": total_orders_sum,
        "promotion_revenue": round(total_rev_sum, 2)
    }

    return {
        "summary": summary,
        "promotions": filtered
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def create_promotion(
    payload: PromotionCreateSchema,
    current_user: Dict[str, Any] = Depends(require_roles(["admin", "regional_manager"])),
    db: Session = Depends(get_db)
):
    """Create a new promotional campaign (Admin / Regional Manager)."""
    import uuid
    new_id = f"PROMO-{str(uuid.uuid4().hex[:6]).upper()}"

    s_date = date.fromisoformat(payload.start_date) if payload.start_date else date.today()
    e_date = date.fromisoformat(payload.end_date) if payload.end_date else None

    promo = Promotion(
        promotion_id=new_id,
        promotion_name=payload.promotion_name.strip(),
        discount_type=payload.discount_type.strip().upper(),
        discount_value=payload.discount_value,
        min_order_amount=payload.min_order_amount or 0.0,
        max_discount_amount=payload.max_discount_amount or 50.0,
        applicable_category=payload.applicable_category or "ALL",
        start_date=s_date,
        end_date=e_date,
        target_segment=payload.target_segment or "ALL",
        description=payload.description or "",
        is_misleading=bool(payload.is_misleading)
    )

    db.add(promo)
    db.commit()
    db.refresh(promo)

    return {
        "message": "Promotion campaign created successfully",
        "promotion_id": promo.promotion_id
    }


@router.put("/{promotion_id}")
def update_promotion(
    promotion_id: str,
    payload: PromotionUpdateSchema,
    current_user: Dict[str, Any] = Depends(require_roles(["admin", "regional_manager"])),
    db: Session = Depends(get_db)
):
    """Update promotional campaign parameters (Admin / Regional Manager)."""
    clean_id = promotion_id.strip()
    promo = db.query(Promotion).filter(Promotion.promotion_id == clean_id).first()
    if not promo:
        raise HTTPException(status_code=404, detail=f"Promotion '{clean_id}' not found.")

    data = payload.model_dump(exclude_unset=True) if hasattr(payload, "model_dump") else payload.dict(exclude_unset=True)
    for field, val in data.items():
        if val is not None:
            if field in ("start_date", "end_date") and isinstance(val, str):
                setattr(promo, field, date.fromisoformat(val))
            else:
                setattr(promo, field, val)

    db.commit()
    db.refresh(promo)

    return {
        "message": "Promotion updated successfully",
        "promotion_id": promo.promotion_id
    }


@router.delete("/{promotion_id}")
def delete_promotion(
    promotion_id: str,
    current_user: Dict[str, Any] = Depends(require_roles(["admin"])),
    db: Session = Depends(get_db)
):
    """Delete a promotional campaign (Admin only)."""
    clean_id = promotion_id.strip()
    promo = db.query(Promotion).filter(Promotion.promotion_id == clean_id).first()
    if not promo:
        raise HTTPException(status_code=404, detail=f"Promotion '{clean_id}' not found.")

    db.delete(promo)
    db.commit()

    return {"message": f"Promotion '{clean_id}' deleted successfully."}
