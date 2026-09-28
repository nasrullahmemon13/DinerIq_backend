"""
DineIQ Analytics - Peak Periods & Temporal Demand Router
Implements SRS Step 19 / Functional Requirements (xxviii), (xxix), (xxx):
- Peak dining hours identification (Lunch vs Dinner rush)
- Day-of-week patterns and weekend vs. weekday traffic distribution
- 24h x 7d order intensity heatmap
- Channel-specific temporal volume distribution
- Seasonal demand indices (Spring, Summer, Fall, Winter)
"""

import os
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, status
import pandas as pd
import numpy as np

router = APIRouter(prefix="/api/v1/analytics/peak-periods", tags=["Peak Periods Analytics (FR-xxviii-xxx)"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HOURLY_PATH = os.path.join(PROJECT_ROOT, "processed_data", "forecasting", "temporal_patterns_hourly.parquet")
DAILY_PATH = os.path.join(PROJECT_ROOT, "processed_data", "forecasting", "temporal_patterns_daily.parquet")
SEASONAL_PATH = os.path.join(PROJECT_ROOT, "processed_data", "forecasting", "temporal_patterns_seasonal.parquet")
CHANNELS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "channels", "channel_hourly_patterns.parquet")
ORDERS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "orders", "orders.parquet")

DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@router.get("")
def get_peak_periods_analytics() -> Dict[str, Any]:
    """
    Returns complete temporal intelligence payload:
    - Summary metrics: peak hour, peak day, weekend share %, seasonal peak
    - Hourly traffic breakdown (0-23 hours)
    - Day-of-week traffic breakdown (Monday-Sunday)
    - 7x24 heatmap matrix with intensity scale
    - Channel hourly breakdown
    - Seasonal demand indices
    """
    if not os.path.exists(HOURLY_PATH) or not os.path.exists(DAILY_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Temporal pattern datasets not found. Ensure pipeline has executed."
        )

    hourly_df = pd.read_parquet(HOURLY_PATH)
    daily_df = pd.read_parquet(DAILY_PATH)
    seasonal_df = pd.read_parquet(SEASONAL_PATH) if os.path.exists(SEASONAL_PATH) else pd.DataFrame()
    channels_df = pd.read_parquet(CHANNELS_PATH) if os.path.exists(CHANNELS_PATH) else pd.DataFrame()

    hourly_df_sorted = hourly_df.sort_values("hour")
    hourly_data: List[Dict[str, Any]] = []
    for _, r in hourly_df_sorted.iterrows():
        h = int(r["hour"])
        ampm = "AM" if h < 12 else "PM"
        display_h = 12 if (h == 0 or h == 12) else h % 12
        time_label = f"{display_h} {ampm}"
        hourly_data.append({
            "hour": h,
            "label": time_label,
            "total_revenue": round(float(r["total_revenue"]), 2),
            "order_count": int(r["order_count"]),
            "revenue_share_pct": round(float(r["revenue_share_pct"]), 2),
            "avg_order_value": round(float(r["avg_order_value"]), 2)
        })

    peak_hour_row = hourly_df.loc[hourly_df["order_count"].idxmax()]
    peak_hour_int = int(peak_hour_row["hour"])
    peak_hour_ampm = "AM" if peak_hour_int < 12 else "PM"
    peak_hour_label = f"{12 if peak_hour_int in (0, 12) else peak_hour_int % 12} {peak_hour_ampm}"

    daily_df["sort_key"] = daily_df["day_name"].apply(lambda d: DAYS_ORDER.index(d) if d in DAYS_ORDER else 99)
    daily_sorted = daily_df.sort_values("sort_key")
    daily_data: List[Dict[str, Any]] = []
    weekend_orders = 0
    weekday_orders = 0

    for _, r in daily_sorted.iterrows():
        d_name = str(r["day_name"])
        orders = int(r.get("total_orders", r.get("order_count", 0)))
        rev = round(float(r.get("total_revenue", 0.0)), 2)
        share = round(float(r.get("revenue_share_pct", 0.0)), 2)
        avg_daily = round(float(r.get("avg_daily_orders", 0.0)), 1)

        is_wknd = d_name in ("Saturday", "Sunday")
        if is_wknd:
            weekend_orders += orders
        else:
            weekday_orders += orders

        daily_data.append({
            "day_name": d_name,
            "is_weekend": is_wknd,
            "total_orders": orders,
            "total_revenue": rev,
            "revenue_share_pct": share,
            "avg_daily_orders": avg_daily
        })

    peak_day_row = daily_df.loc[daily_df["total_orders"].idxmax()] if "total_orders" in daily_df else daily_df.iloc[0]
    peak_day_name = str(peak_day_row["day_name"])

    total_all_orders = weekend_orders + weekday_orders
    weekend_share_pct = round((weekend_orders / total_all_orders * 100), 1) if total_all_orders > 0 else 0.0

    heatmap_matrix = []
    max_cell_orders = 1
    heatmap_error = None
    excluded_timestamp_rows = 0
    if os.path.exists(ORDERS_PATH):
        try:
            orders_df = pd.read_parquet(ORDERS_PATH, columns=["order_id", "order_timestamp"])
            orders_df["dt"] = pd.to_datetime(orders_df["order_timestamp"], errors="coerce")
            excluded_timestamp_rows = int(orders_df["dt"].isna().sum())
            orders_df = orders_df.dropna(subset=["dt"])
            orders_df["hour"] = orders_df["dt"].dt.hour
            orders_df["day_name"] = orders_df["dt"].dt.day_name()
            pivot = orders_df.pivot_table(index="day_name", columns="hour", values="order_id", aggfunc="count", fill_value=0)

            for h in range(24):
                if h not in pivot.columns:
                    pivot[h] = 0
            pivot = pivot.reindex(columns=range(24), fill_value=0)

            max_cell_orders = max(int(pivot.values.max()), 1)

            for d in DAYS_ORDER:
                if d in pivot.index:
                    row_counts = [int(pivot.loc[d, h]) for h in range(24)]
                else:
                    row_counts = [0] * 24

                hours_list = [
                    {
                        "hour": h,
                        "hour_label": f"{12 if h in (0, 12) else h % 12} {'AM' if h < 12 else 'PM'}",
                        "count": row_counts[h],
                        "intensity_pct": round((row_counts[h] / max_cell_orders * 100), 1) if max_cell_orders > 0 else 0
                    }
                    for h in range(24)
                ]

                heatmap_matrix.append({
                    "day": d,
                    "day_name": d,
                    "hourly_counts": row_counts,
                    "hours": hours_list
                })
        except Exception as error:
            heatmap_error = str(error)
            heatmap_matrix = []

    seasonal_data: List[Dict[str, Any]] = []
    if not seasonal_df.empty:
        for _, r in seasonal_df.iterrows():
            seasonal_data.append({
                "season": str(r.get("season", "")),
                "total_revenue": round(float(r.get("total_revenue", 0.0)), 2),
                "seasonal_demand_index": round(float(r.get("seasonal_demand_index", 1.0)), 3),
                "revenue_share_pct": round(float(r.get("revenue_share_pct", 0.0)), 2)
            })

    channel_hourly_data = []
    if not channels_df.empty:
        for _, r in channels_df.iterrows():
            ch_name = str(r["ordering_channel"])
            counts = {}
            for col in channels_df.columns:
                if col != "ordering_channel":
                    try:
                        counts[int(col)] = int(r[col])
                    except (ValueError, TypeError):
                        pass
            channel_hourly_data.append({
                "channel": ch_name,
                "hourly_counts": counts
            })

    summary = {
        "peak_hour": peak_hour_label,
        "peak_hour_military": peak_hour_int,
        "peak_hour_orders": int(peak_hour_row["order_count"]),
        "peak_day": peak_day_name,
        "peak_day_share_pct": round(float(peak_day_row.get("total_orders", 0)) / total_all_orders * 100, 1) if total_all_orders else 0,
        "weekend_order_share_pct": weekend_share_pct,
        "weekday_order_share_pct": round(100.0 - weekend_share_pct, 1),
        "total_analyzed_orders": total_all_orders,
        "max_heatmap_density": max_cell_orders
    }

    return {
        "summary": summary,
        "hourly_distribution": hourly_data,
        "daily_distribution": daily_data,
        "heatmap": heatmap_matrix,
        "heatmap_error": heatmap_error,
        "excluded_timestamp_rows": excluded_timestamp_rows,
        "seasonal_patterns": seasonal_data,
        "channel_hourly_distribution": channel_hourly_data
    }
