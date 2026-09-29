"""
DineIQ Analytics - Dynamic Pricing & Promotion Optimization Router
Implements SRS Step 32 / Functional Requirement (xxxii):
- Price sensitivity tiers (High, Balanced, Low)
- Price elasticity of demand across 150 menu items
- Category price sensitivity summary
- Promotion trap detection (5 trap types)
"""

import os
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Query
import pandas as pd
from backend.services.data_store import read_frame, data_exists

router = APIRouter(prefix="/api/v1/analytics/pricing", tags=["Pricing Intelligence"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def load_parquet(folder: str, filename: str) -> pd.DataFrame:
    p = os.path.join(PROJECT_ROOT, "processed_data", folder, filename)
    if data_exists(p):
        try:
            return read_frame(p)
        except Exception as e:
            print(f"Error loading {p}: {e}")
    return pd.DataFrame()

@router.get("")
def get_pricing_intelligence(
    tier: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None)
) -> Dict[str, Any]:
    psa_df = load_parquet("pricing", "price_sensitivity_analysis.parquet")
    cps_df = load_parquet("pricing", "category_price_sensitivity.parquet")
    trap_df = load_parquet("promotion", "promotion_trap_detection.parquet")

    items = []
    filtered = psa_df.copy()
    if not psa_df.empty:
        filtered = psa_df.copy()
        if tier and tier != "All":
            filtered = filtered[filtered["price_sensitivity_tier"] == tier]
        if category and category != "All":
            filtered = filtered[filtered["category_name"] == category]
        if search:
            filtered = filtered[filtered["item_name"].str.contains(search, case=False, na=False, regex=False)]
        items = filtered.to_dict(orient="records")

    categories = cps_df.to_dict(orient="records") if not cps_df.empty else []
    traps = trap_df.to_dict(orient="records") if not trap_df.empty else []

    summary = {
        "total_analyzed_dishes": len(filtered),
        "high_elasticity_count": int((filtered["price_sensitivity_tier"] == "High Sensitivity").sum()) if not filtered.empty else 0,
        "balanced_count": int((filtered["price_sensitivity_tier"] == "Balanced").sum()) if not filtered.empty else 0,
        "low_elasticity_count": int((filtered["price_sensitivity_tier"] == "Low Sensitivity").sum()) if not filtered.empty else 0,
        "active_promotion_traps": len(trap_df),
        "avg_system_margin": round(float(filtered["profit_margin_pct"].mean()), 1) if not filtered.empty else None
    }

    return {
        "status": "success",
        "summary": summary,
        "items": items,
        "categories": categories,
        "traps": traps
    }
