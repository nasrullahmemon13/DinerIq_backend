"""
DineIQ Analytics - Locations & Channels Intelligence Router
Implements SRS Step 33-35, 38-40 / Functional Requirements (xxxviii), (xxxix), (xl):
- Comprehensive Location Performance Benchmarking across 20 stores
- Location Comparison across Revenue, Profit, Orders, AOV, Customers, Ratings, Wastage
- Location-Specific Menu Classification & Cross-Location Divergent Dishes
- Ordering Channel Intelligence (Dine-in, Takeaway, Third-party Delivery, Restaurant Website/App, Drive-thru)
- Channel Profitability, Basket Sizes, Discounts, Peak Hourly Patterns, Location x Channel Heatmap Matrix
- Evidence-based Dynamic Analytical Observations & Location Recommendations
"""

import os
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException, status
import pandas as pd
import numpy as np
from backend.services import location_scope

router = APIRouter(prefix="/api/v1/analytics/locations-channels", tags=["Locations & Channels Intelligence"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

LOC_MATRIX_PATH = os.path.join(PROJECT_ROOT, "processed_data", "locations", "location_comparison_matrix.parquet")
LOC_MENU_PATH = os.path.join(PROJECT_ROOT, "processed_data", "locations", "location_menu_performance.parquet")
DIVERGENT_PATH = os.path.join(PROJECT_ROOT, "processed_data", "locations", "cross_location_divergent_dishes.parquet")
CHANNEL_COMP_PATH = os.path.join(PROJECT_ROOT, "processed_data", "channels", "ordering_channel_comparison.parquet")
CHANNEL_HOURLY_PATH = os.path.join(PROJECT_ROOT, "processed_data", "channels", "channel_hourly_patterns.parquet")
CHANNEL_MENU_PATH = os.path.join(PROJECT_ROOT, "processed_data", "channels", "channel_menu_preferences.parquet")
ORDERS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "orders", "orders.parquet")
RECOMMENDATIONS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "recommendations", "recommendations.parquet")


class LocationsChannelsAnalyticsService:
    """In-memory singleton caching layer for locations and channels intelligence."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocationsChannelsAnalyticsService, cls).__new__(cls)
            cls._instance._init_datasets()
        return cls._instance

    def _init_datasets(self):
        self.loc_matrix = self._load_df(LOC_MATRIX_PATH)
        self.lmp = self._load_df(LOC_MENU_PATH)
        self.divergent = self._load_df(DIVERGENT_PATH)
        self.channel_comp = self._load_df(CHANNEL_COMP_PATH)
        self.channel_hourly = self._load_df(CHANNEL_HOURLY_PATH)
        self.channel_menu = self._load_df(CHANNEL_MENU_PATH)
        self.orders = self._load_df(ORDERS_PATH)
        self.recommendations = self._load_df(RECOMMENDATIONS_PATH)

        if not self.orders.empty and "order_date" in self.orders.columns:
            self.orders["order_date_dt"] = pd.to_datetime(self.orders["order_date"], errors="coerce")
            self.orders["order_month"] = self.orders["order_date_dt"].dt.month
            self.orders["day_name"] = self.orders["order_date_dt"].dt.day_name()

    @staticmethod
    def _load_df(path: str) -> pd.DataFrame:
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception as e:
                print(f"[Error] Failed to load {path}: {e}")
        csv_path = os.path.splitext(path)[0] + '.csv'
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return pd.DataFrame()


@router.get("/performance")
def get_locations_performance(
    date_range: Optional[str] = Query("ALL", description="Date filter (ALL, Q1, Q2, Q3, Q4, 30_DAYS, 90_DAYS)"),
    location_ids: Optional[str] = Query(None, description="Comma-separated location IDs e.g. LOC-001,LOC-004"),
    city: Optional[str] = Query(None, description="Optional city filter"),
    tier: Optional[str] = Query(None, description="Optional tier filter")
):
    """
    Returns complete location performance matrix, dynamic summary KPIs,
    grouped revenue & profit data, orders & AOV distributions, wastage, ratings,
    and side-by-side comparison payload.
    """
    return location_scope.performance(date_range, location_ids, city, tier)


@router.get("/menu-intelligence")
def get_location_menu_intelligence(
    location_id: Optional[str] = Query(None, description="Location ID filter (e.g. LOC-001 or ALL)"),
    category: Optional[str] = Query(None, description="Category filter"),
    performance_class: Optional[str] = Query(None, description="Class filter: Profit Driver, Volume Driver, Hidden Opportunity, Low Performer"),
    selected_item_id: Optional[str] = Query(None, description="Item ID to compare across all locations"),
    search: Optional[str] = Query(None, description="Dish name search"),
    limit: int = Query(150, ge=1, le=1000)
):
    """
    Returns location-specific menu classifications (3,000 pairs), 85 divergent dishes
    exposing cross-location performance variations, and single-dish cross-location comparisons.
    """
    service = LocationsChannelsAnalyticsService()
    lmp = service.lmp.copy()
    divergent_df = service.divergent.copy()

    if lmp.empty:
        raise HTTPException(status_code=404, detail="Location menu performance dataset unavailable.")

    if location_id and location_id.strip() and location_id.upper() != "ALL":
        lmp = lmp[lmp["location_id"].str.upper() == location_id.strip().upper()]

    if category and category.strip() and category.upper() != "ALL":
        lmp = lmp[lmp["category_name"].str.lower() == category.strip().lower()]

    if performance_class and performance_class.strip() and performance_class.upper() != "ALL":
        lmp = lmp[lmp["location_menu_classification"].str.lower() == performance_class.strip().lower()]

    if search and search.strip():
        s = search.strip().lower()
        lmp = lmp[lmp["item_name"].str.lower().str.contains(s, na=False) | lmp["item_id"].str.lower().str.contains(s, na=False)]

    menu_rows = []
    for _, r in lmp.head(limit).iterrows():
        menu_rows.append({
            "location_id": str(r["location_id"]),
            "restaurant_name": str(r.get("restaurant_name", "")),
            "item_id": str(r["item_id"]),
            "item_name": str(r["item_name"]),
            "category_name": str(r.get("category_name", "")),
            "quantity_sold": int(r.get("quantity_sold", 0)),
            "revenue": round(float(r.get("revenue", 0.0)), 2),
            "cost": round(float(r.get("cost", 0.0)), 2),
            "gross_profit": round(float(r.get("gross_profit", 0.0)), 2),
            "margin_pct": round(float(r.get("contribution_margin_pct", 50.0)), 1),
            "avg_rating": round(float(r.get("avg_rating", 4.2)), 1),
            "rating_count": int(r.get("rating_count", 0)),
            "wastage_cost": round(float(r.get("wastage_cost", 0.0)), 2),
            "wastage_percentage": round(float(r.get("wastage_percentage", 0.0)), 1),
            "classification": str(r.get("location_menu_classification", "Unclassified")),
            "volume_percentile": round(float(r.get("loc_volume_percentile", 0.5)), 2),
            "margin_percentile": round(float(r.get("loc_margin_percentile", 0.5)), 2)
        })

    divergent_list = []
    if not divergent_df.empty:
        for _, r in divergent_df.head(20).iterrows():
            divergent_list.append({
                "item_id": str(r["item_id"]),
                "item_name": str(r["item_name"]),
                "category_name": str(r.get("category_name", "")),
                "distinct_class_count": int(r.get("distinct_class_count", 0)),
                "classes_observed": str(r.get("classes_observed", "")),
                "profit_driver_locs": int(r.get("profit_driver_locs", 0)),
                "volume_driver_locs": int(r.get("volume_driver_locs", 0)),
                "hidden_opportunity_locs": int(r.get("hidden_opportunity_locs", 0)),
                "low_performer_locs": int(r.get("low_performer_locs", 0)),
                "quantity_spread": int(r.get("quantity_spread", 0)),
                "min_quantity": int(r.get("min_quantity", 0)),
                "max_quantity": int(r.get("max_quantity", 0))
            })

    item_comparison = []
    item_comparison_summary = {}
    target_item = selected_item_id or (divergent_list[0]["item_id"] if divergent_list else "ITEM-081")

    if not service.lmp.empty and target_item:
        sub_item = service.lmp[service.lmp["item_id"] == target_item]
        if not sub_item.empty:
            sorted_sub = sub_item.sort_values("revenue", ascending=False)
            for _, r in sorted_sub.iterrows():
                item_comparison.append({
                    "location_id": str(r["location_id"]),
                    "restaurant_name": str(r.get("restaurant_name", "")).replace("DineIQ ", ""),
                    "full_restaurant_name": str(r.get("restaurant_name", "")),
                    "quantity_sold": int(r.get("quantity_sold", 0)),
                    "revenue": round(float(r.get("revenue", 0.0)), 2),
                    "gross_profit": round(float(r.get("gross_profit", 0.0)), 2),
                    "margin_pct": round(float(r.get("contribution_margin_pct", 50.0)), 1),
                    "avg_rating": round(float(r.get("avg_rating", 4.2)), 1),
                    "wastage_cost": round(float(r.get("wastage_cost", 0.0)), 2),
                    "classification": str(r.get("location_menu_classification", "Unclassified"))
                })

            first_r = sub_item.iloc[0]
            item_comparison_summary = {
                "item_id": target_item,
                "item_name": str(first_r.get("item_name", "")),
                "category_name": str(first_r.get("category_name", "")),
                "total_network_revenue": round(float(sub_item["revenue"].sum()), 2),
                "total_network_units": int(sub_item["quantity_sold"].sum()),
                "classes_count": sub_item["location_menu_classification"].value_counts().to_dict()
            }

    categories = sorted(list(service.lmp["category_name"].dropna().unique())) if not service.lmp.empty else []
    classes = ["Profit Driver", "Volume Driver", "Hidden Opportunity", "Low Performer"]

    distinct_items = (
        service.lmp[["item_id", "item_name", "category_name"]]
        .drop_duplicates(subset=["item_id"])
        .sort_values("item_name")
        .to_dict(orient="records")
    ) if not service.lmp.empty else []

    for dish in divergent_list:
        matches = service.lmp[service.lmp.item_id.eq(dish['item_id'])]
        dish['spread_ratio'] = round(dish['max_quantity'] / dish['min_quantity'], 2) if dish['min_quantity'] else None
        dish['top_location_name'] = str(matches.loc[matches.quantity_sold.idxmax(), 'restaurant_name']) if not matches.empty else None
        dish['worst_location_name'] = str(matches.loc[matches.quantity_sold.idxmin(), 'restaurant_name']) if not matches.empty else None

    return {
        "menu_items": menu_rows,
        "divergent_dishes": divergent_list,
        "item_comparison": item_comparison,
        "item_comparison_summary": item_comparison_summary,
        "categories": categories,
        "classes": classes,
        "distinct_items": distinct_items
    }


@router.get("/channels")
def get_ordering_channels_analytics(
    location_id: Optional[str] = Query(None, description="Optional location filter"),
    date_range: Optional[str] = Query("ALL", description="Date filter")
):
    """
    Returns channel KPI cards, ordering channel mix, channel profitability,
    basket size comparison, menu preferences per channel, discount/promotions analysis,
    24-hour peak heatmap matrix, and location x channel cross-tabulation.
    """
    return location_scope.channels(date_range, location_id)


@router.get("/location/{location_id}/drilldown")
def get_location_drilldown(location_id: str):
    """
    Returns granular drilldown payload for a single restaurant location:
    performance summary, top & weak menu items, channel split, and active recommendations.
    """
    service = LocationsChannelsAnalyticsService()
    loc_clean = location_id.strip()

    loc_match = service.loc_matrix[service.loc_matrix["location_id"] == loc_clean]
    if loc_match.empty:
        raise HTTPException(status_code=404, detail=f"Location '{loc_clean}' not found.")

    r = loc_match.iloc[0]

    top_items = []
    weak_items = []
    if not service.lmp.empty:
        sub_lmp = service.lmp[service.lmp["location_id"] == loc_clean]
        if not sub_lmp.empty:
            top_df = sub_lmp.sort_values("revenue", ascending=False).head(5)
            for _, item in top_df.iterrows():
                top_items.append({
                    "item_name": str(item["item_name"]),
                    "category": str(item.get("category_name", "")),
                    "quantity": int(item.get("quantity_sold", 0)),
                    "revenue": round(float(item.get("revenue", 0.0)), 2),
                    "margin_pct": round(float(item.get("contribution_margin_pct", 50.0)), 1),
                    "rating": round(float(item.get("avg_rating", 4.2)), 1),
                    "classification": str(item.get("location_menu_classification", "Profit Driver"))
                })

            weak_df = sub_lmp[sub_lmp["location_menu_classification"] == "Low Performer"].sort_values("revenue").head(5)
            for _, item in weak_df.iterrows():
                weak_items.append({
                    "item_name": str(item["item_name"]),
                    "category": str(item.get("category_name", "")),
                    "quantity": int(item.get("quantity_sold", 0)),
                    "revenue": round(float(item.get("revenue", 0.0)), 2),
                    "margin_pct": round(float(item.get("contribution_margin_pct", 50.0)), 1),
                    "classification": "Low Performer"
                })

    loc_recs = []
    if not service.recommendations.empty:
        matched_recs = service.recommendations[
            service.recommendations["target_entity_id"].str.contains(loc_clean, na=False) |
            service.recommendations["target_entity_name"].str.contains(str(r.get("restaurant_name", "")), na=False)
        ]
        for _, rec in matched_recs.head(5).iterrows():
            loc_recs.append({
                "priority": str(rec.get("priority", "High")),
                "category": str(rec.get("category", "")),
                "action": str(rec.get("recommended_action", "")),
                "evidence": str(rec.get("formatted_evidence", "")),
                "potential_impact": round(float(rec.get("potential_business_impact", 0.0)), 2),
                "recommendation_id": str(rec.get("recommendation_id", ""))
            })

    return {
        "location_id": loc_clean,
        "restaurant_name": str(r["restaurant_name"]),
        "city": str(r.get("restaurant_city", "")),
        "state": str(r.get("restaurant_state", "")),
        "tier": str(r.get("location_tier", "Standard")),
        "revenue": round(float(r["total_revenue"]), 2),
        "gross_profit": round(float(r["gross_profit"]), 2),
        "margin_pct": round(float(r.get("contribution_margin_pct", 56.0)), 1),
        "orders": int(r["total_orders"]),
        "aov": round(float(r.get("average_order_value", 0.0)), 2),
        "customers": int(r.get("unique_customer_count", 0)),
        "repeat_rate": round(float(r.get("repeat_purchase_rate", 5.5)), 1),
        "overall_rating": round(float(r.get("avg_overall_rating", 4.2)), 2),
        "csat_pct": round(float(r.get("csat_pct", 65.0)), 1),
        "wastage_loss": round(float(r.get("total_wastage_loss_amount", 0.0)), 2),
        "wastage_rate": round(float(r.get("wastage_rate_pct", 14.5)), 1),
        "top_menu_items": top_items,
        "weak_menu_items": weak_items,
        "recommendations": loc_recs
    }


@router.get("/recommendations")
def get_location_channel_recommendations(
    category: Optional[str] = Query(None, description="Category filter"),
    priority: Optional[str] = Query(None, description="Priority filter: Critical, High, Medium")
):
    """
    Returns prescriptive AI recommendations tailored to location optimization,
    cross-location menu rebalancing, and channel margin preservation.
    """
    service = LocationsChannelsAnalyticsService()
    recs_df = service.recommendations.copy()

    if recs_df.empty:
        return {"recommendations": []}

    relevant_cats = [
        "Investigate anomalous locations",
        "Promote high-margin Hidden Opportunities",
        "Remove or redesign persistent Low Performers",
        "Reduce preparation quantity of high-wastage dishes"
    ]
    recs_df = recs_df[recs_df["category"].isin(relevant_cats)]

    if priority and priority.strip() and priority.upper() != "ALL":
        recs_df = recs_df[recs_df["priority"].str.lower() == priority.strip().lower()]

    results = []
    for _, r in recs_df.iterrows():
        bullets = r.get("reason_bullets")
        bullets_list = list(bullets) if isinstance(bullets, (list, np.ndarray)) else []
        results.append({
            "recommendation_id": str(r.get("recommendation_id", "")),
            "priority": str(r.get("priority", "High")),
            "category": str(r.get("category", "")),
            "target_entity": str(r.get("target_entity_name", "")),
            "target_type": str(r.get("target_entity_type", "")),
            "action": str(r.get("recommended_action", "")),
            "evidence_bullets": bullets_list,
            "formatted_evidence": str(r.get("formatted_evidence", "")),
            "potential_impact": round(float(r.get("potential_business_impact", 0.0)), 2),
            "impact_rationale": str(r.get("business_impact_rationale", "")),
            "implementation_effort": str(r.get("implementation_effort", "Medium"))
        })

    return {"recommendations": results}
