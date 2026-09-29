"""
DineIQ Analytics - What-If Scenario Analysis Router
Implements SRS Steps 40 & 41 / Functional Requirement (li):
- Interactive scenario simulation across prices, promotions, preparation, demand, and wastage
- Returns baseline vs simulated indicators: Revenue, Margin, Demand, Wastage, Profitability
- MANDATORY DISCLAIMER: "SIMULATION / ESTIMATE ONLY"
"""

import os
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from backend.services.data_store import read_frame, data_exists

from src.what_if_engine import WhatIfScenarioEngine, SIMULATION_DISCLAIMER

router = APIRouter(prefix="/api/v1/analytics/what-if", tags=["What-If Simulator"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_engine = None

def get_engine() -> WhatIfScenarioEngine:
    global _engine
    if _engine is None:
        _engine = WhatIfScenarioEngine()
    return _engine

class SimulationRequest(BaseModel):
    scenario_type: str = Field(..., description="Scenario: price_increase, price_decrease, discount, prep_reduction, demand_increase, wastage_change, remove_item")
    item_id: Optional[str] = "ITEM-001"
    percentage_change: float = Field(10.0, description="Percentage adjustment (+/- 50%)")
    target_category: Optional[str] = None

@router.get("/items")
def get_simulatable_items() -> List[Dict[str, Any]]:
    engine = get_engine()
    df = engine.baseline_df
    if df is not None and not df.empty:
        records = []
        for _, row in df.iterrows():
            price_val = row.get("base_price")
            if pd.isna(price_val) or price_val is None:
                price_val = row.get("price", 0.0)

            cost_val = row.get("cost_price", 0.0)
            if pd.isna(cost_val) or cost_val is None:
                cost_val = 0.0

            margin_val = row.get("profit_percentage", 50.0)
            if pd.isna(margin_val) or margin_val is None:
                margin_val = 50.0

            elasticity_val = row.get("avg_empirical_elasticity", 1.2)
            if pd.isna(elasticity_val) or elasticity_val is None:
                elasticity_val = 1.2

            records.append({
                "item_id": str(row.get("item_id", "")),
                "item_name": str(row.get("item_name", "")),
                "category_name": str(row.get("category_name", "General")),
                "price": round(float(price_val), 2),
                "cost_price": round(float(cost_val), 2),
                "profit_margin_pct": round(float(margin_val), 1),
                "price_sensitivity_tier": str(row.get("price_sensitivity_tier", "Moderately Price Sensitive")),
                "avg_elasticity": round(float(elasticity_val), 2),
                "current_quantity_sold": int(row.get("quantity_sold", 0)) if not pd.isna(row.get("quantity_sold")) else 0,
                "current_revenue": round(float(row.get("revenue", 0.0)), 2) if not pd.isna(row.get("revenue")) else 0.0
            })
        return sorted(records, key=lambda x: x["item_name"])
    return []

@router.get("/benchmarks")
def get_benchmarks() -> List[Dict[str, Any]]:
    p = os.path.join(PROJECT_ROOT, "processed_data", "what_if", "what_if_scenario_benchmark.parquet")
    if data_exists(p):
        df = read_frame(p)
        records = []
        for _, row in df.iterrows():
            records.append({
                "scenario_name": str(row.get("scenario_name", "")),
                "scenario_variant": str(row.get("scenario_variant", "")),
                "revenue_simulated": round(float(row.get("estimated_revenue", 0.0)), 2),
                "revenue_baseline": round(float(row.get("baseline_revenue", 0.0)), 2),
                "revenue_delta": round(float(row.get("estimated_revenue_delta", 0.0)), 2),
                "profit_simulated": round(float(row.get("estimated_net_profitability", 0.0)), 2),
                "profit_baseline": round(float(row.get("baseline_net_profitability", 0.0)), 2),
                "profit_delta": round(float(row.get("estimated_profitability_delta", 0.0)), 2),
                "wastage_simulated": round(float(row.get("estimated_wastage_cost", 0.0)), 2),
                "wastage_baseline": round(float(row.get("baseline_wastage_cost", 0.0)), 2),
                "wastage_delta": round(float(row.get("estimated_wastage_cost_delta", 0.0)), 2),
                "business_rationale": str(row.get("scenario_variant", "System-level benchmark optimization lever."))
            })
        return records
    return []

@router.post("/simulate")
def run_simulation(req: SimulationRequest) -> Dict[str, Any]:
    engine = get_engine()
    res = None

    try:
        if req.scenario_type == "price_increase":
            res = engine.simulate_increase_menu_price(item_id=req.item_id, price_increase_pct=abs(req.percentage_change))
        elif req.scenario_type == "price_decrease":
            res = engine.simulate_reduce_item_price(item_id=req.item_id, price_reduction_pct=abs(req.percentage_change))
        elif req.scenario_type == "discount":
            res = engine.simulate_change_discount_percentage(item_id=req.item_id, discount_pct=abs(req.percentage_change))
        elif req.scenario_type == "prep_reduction":
            res = engine.simulate_reduce_preparation_quantity(item_id=req.item_id, prep_reduction_pct=abs(req.percentage_change))
        elif req.scenario_type == "demand_increase":
            res = engine.simulate_increase_predicted_demand(item_id=req.item_id, demand_increase_pct=abs(req.percentage_change))
        elif req.scenario_type == "wastage_change":
            res = engine.simulate_change_wastage_assumptions(item_id=req.item_id, wastage_reduction_pct=abs(req.percentage_change))
        elif req.scenario_type == "remove_item":
            res = engine.simulate_remove_menu_item(item_id=req.item_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported scenario_type: {req.scenario_type}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

    if not res:
        raise HTTPException(status_code=404, detail="Item or scenario not found")

    rev = res.revenue or {}
    prof = res.profitability or {}
    dem = res.demand or {}
    cm = res.contribution_margin or {}
    wst = res.wastage or {}

    findings_text = ""
    if isinstance(res.key_findings, list) and len(res.key_findings) > 0:
        findings_text = " • ".join(res.key_findings)
    elif res.key_findings:
        findings_text = str(res.key_findings)
    else:
        findings_text = "Simulation evaluated econometric elasticity and multi-dimensional operational impact across financial indicators."

    return {
        "scenario_name": res.scenario_name,
        "is_simulation_estimate": True,
        "disclaimer": SIMULATION_DISCLAIMER,
        "indicators": {
            "revenue": {
                "baseline": round(float(rev.get("baseline", 0.0)), 2),
                "simulated": round(float(rev.get("estimated", 0.0)), 2),
                "delta": round(float(rev.get("delta", 0.0)), 2),
                "pct_change": round(float(rev.get("pct_change", 0.0)), 2)
            },
            "profit": {
                "baseline": round(float(prof.get("baseline_net_profit", 0.0)), 2),
                "simulated": round(float(prof.get("estimated_net_profit", 0.0)), 2),
                "delta": round(float(prof.get("delta_net_profit", 0.0)), 2),
                "pct_change": round(float(prof.get("pct_change", 0.0)), 2)
            },
            "demand": {
                "baseline": round(float(dem.get("baseline_units", 0.0)), 0),
                "simulated": round(float(dem.get("estimated_units", 0.0)), 0),
                "delta": round(float(dem.get("delta_units", 0.0)), 0),
                "pct_change": round(float(dem.get("pct_change", 0.0)), 2)
            },
            "margin_pct": {
                "baseline": round(float(cm.get("baseline_pct", 0.0)), 2),
                "simulated": round(float(cm.get("estimated_pct", 0.0)), 2),
                "delta": round(float(cm.get("delta_pct", 0.0)), 2)
            },
            "wastage": {
                "baseline": round(float(wst.get("baseline_cost", 0.0)), 2),
                "simulated": round(float(wst.get("estimated_cost", 0.0)), 2),
                "delta": round(float(wst.get("delta_cost", 0.0)), 2),
                "pct_change": round(float(wst.get("pct_change", 0.0)), 2)
            }
        },
        "target_entity": res.scenario_parameters.get("item_id", req.item_id) if hasattr(res, "scenario_parameters") else req.item_id,
        "parameters": res.scenario_parameters if hasattr(res, "scenario_parameters") else {},
        "business_rationale": findings_text
    }
