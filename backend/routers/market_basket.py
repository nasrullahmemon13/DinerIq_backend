"""
DineIQ Analytics - Market Basket Analysis Router
Implements SRS Step 17-18 / Functional Requirements (xxv), (xxvi), (xxvii):
- Market-Basket Analysis using distributed FP-Growth association rules
- Support, Confidence, and Lift calculations
- Frequent menu combinations and co-occurrence patterns
- Cross-sell, Upsell, and Bundle Recommendations with dynamic business interpretation
"""

import os
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException, status
import pandas as pd

router = APIRouter(prefix="/api/v1/analytics/basket", tags=["Market Basket Analysis (FR-xxv-xxvii)"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RULES_PATH = os.path.join(PROJECT_ROOT, "processed_data", "basket_analysis", "association_rules.parquet")
RECS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "basket_analysis", "menu_recommendations.parquet")


@router.get("")
def get_market_basket_analysis(
    min_lift: Optional[float] = Query(None, description="Minimum lift threshold"),
    min_confidence: Optional[float] = Query(None, description="Minimum confidence threshold"),
    category: Optional[str] = Query(None, description="Filter by menu category"),
    search: Optional[str] = Query(None, description="Search item name")
):
    """
    Returns complete Market Basket Analysis payload with Support, Confidence, Lift,
    and bundle recommendations.
    """
    if not os.path.exists(RULES_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market basket association rules dataset not found. Ensure pipeline has executed."
        )

    rules_df = pd.read_parquet(RULES_PATH)
    recs_df = pd.read_parquet(RECS_PATH) if os.path.exists(RECS_PATH) else pd.DataFrame()

    total_rules_count = len(rules_df)
    avg_lift_val = round(float(rules_df["lift"].mean()), 2) if not rules_df.empty else 1.0
    max_lift_val = round(float(rules_df["lift"].max()), 2) if not rules_df.empty else 1.0

    filtered_rules = rules_df.copy()
    if min_lift is not None:
        filtered_rules = filtered_rules[filtered_rules["lift"] >= min_lift]
    if min_confidence is not None:
        filtered_rules = filtered_rules[filtered_rules["confidence"] >= min_confidence]
    if category and category.upper() not in ("ALL", ""):
        c_lower = category.lower()
        filtered_rules = filtered_rules[
            filtered_rules["antecedent_category"].str.lower().str.contains(c_lower, regex=False, na=False) |
            filtered_rules["consequent_category"].str.lower().str.contains(c_lower, regex=False, na=False)
        ]
    if search and search.strip():
        s = search.strip().lower()
        filtered_rules = filtered_rules[
            filtered_rules["antecedent_name"].str.lower().str.contains(s, regex=False, na=False) |
            filtered_rules["consequent_name"].str.lower().str.contains(s, regex=False, na=False)
        ]

    filtered_rules = filtered_rules.sort_values("lift", ascending=False)

    rules_list = []
    for _, r in filtered_rules.head(100).iterrows():
        a_name = str(r["antecedent_name"])
        c_name = str(r["consequent_name"])
        lift_v = round(float(r["lift"]), 2)
        conf_pct = round(float(r["confidence"]) * 100, 1)
        supp_pct = round(float(r["support"]) * 100, 2)
        orders_together = int(r.get("joint_orders_count", 0))

        interpretation = (
            f"Guests ordering '{a_name}' have a {lift_v}x higher likelihood of pairing with "
            f"'{c_name}' ({conf_pct}% confidence across {orders_together} orders)."
        )

        rules_list.append({
            "antecedent_id": str(r["antecedent_id"]),
            "antecedent_name": a_name,
            "antecedent_category": str(r.get("antecedent_category", "")),
            "consequent_id": str(r["consequent_id"]),
            "consequent_name": c_name,
            "consequent_category": str(r.get("consequent_category", "")),
            "support": supp_pct,
            "confidence": conf_pct,
            "lift": lift_v,
            "joint_orders_count": orders_together,
            "business_interpretation": interpretation
        })

    bundles_list = []
    if not recs_df.empty:
        for _, b in recs_df.head(25).iterrows():
            bundles_list.append({
                "recommendation_type": str(b.get("recommendation_type", "Frequently Paired")),
                "primary_item": str(b.get("primary_item", "")),
                "recommended_item": str(b.get("recommended_item", "")),
                "category_pair": str(b.get("category_pair", "")),
                "support": round(float(b.get("support", 0.0)) * 100, 2),
                "confidence": round(float(b.get("confidence", 0.0)) * 100, 1),
                "lift": round(float(b.get("lift", 1.0)), 2),
                "commercial_rationale": str(b.get("commercial_rationale", ""))
            })

    summary = {
        "total_rules": len(filtered_rules),
        "population_rules": total_rules_count,
        "returned_rules": len(rules_list),
        "average_lift": round(float(filtered_rules.lift.mean()),2) if len(filtered_rules) else None,
        "max_lift": round(float(filtered_rules.lift.max()),2) if len(filtered_rules) else None,
        "strong_associations_count": len(filtered_rules[filtered_rules["lift"] > 1.1])
    }

    return {
        "summary": summary,
        "association_rules": rules_list,
        "bundle_recommendations": bundles_list,
        "available_categories": sorted(set(rules_df.antecedent_category.dropna()) | set(rules_df.consequent_category.dropna()))
    }
