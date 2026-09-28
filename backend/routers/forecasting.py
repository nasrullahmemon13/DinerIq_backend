"""
DineIQ Analytics - Demand Forecasting & Daypart Analytics Router
Implements SRS Step 46 / Functional Requirement (xxviii):
- Time-series demand forecasting at item, category, and location granularities
- Actual evaluation metrics: MAE, RMSE, MAPE, R2 score
- Daypart and hourly demand curves
"""

import os
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Query, HTTPException, Depends
import pandas as pd
import numpy as np
from database.connection import get_db
from database.models import SystemConfig
import json

router = APIRouter(prefix="/api/v1/analytics/forecasting", tags=["Demand Forecasting"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FORECAST_DIR = os.path.join(PROJECT_ROOT, "processed_data", "forecasting")

@router.get('/live')
def live_forecasting(horizon: int = Query(14, ge=1, le=90), location: Optional[str] = None, db=Depends(get_db)):
    from backend.services.live_forecast import forecast
    try:
        active = db.query(SystemConfig).filter(SystemConfig.config_key == 'forecast.active_model').first()
        return forecast(horizon, location, json.loads(active.config_value) if active else None)
    except ValueError as error:
        raise HTTPException(422, str(error))

def load_parquet_safe(filename: str) -> pd.DataFrame:
    p = os.path.join(FORECAST_DIR, filename)
    if os.path.exists(p):
        try:
            return pd.read_parquet(p)
        except Exception as e:
            print(f"Error loading {p}: {e}")
    return pd.DataFrame()

@router.get("")
def get_forecasting_overview(
    granularity: str = Query("category", enum=["category", "item", "location"]),
    item_search: Optional[str] = None
) -> Dict[str, Any]:
    eval_df = load_parquet_safe("forecast_evaluations.parquet")
    evaluations = eval_df.to_dict(orient="records") if not eval_df.empty else []

    cat_df = load_parquet_safe("category_demand_forecast.parquet")
    category_forecast = []
    if not cat_df.empty:
        cat_df["order_date_str"] = cat_df["order_date"].astype(str)
        category_forecast = cat_df.head(150).to_dict(orient="records")

    item_df = load_parquet_safe("item_demand_forecast.parquet")
    item_forecast = []
    if not item_df.empty:
        item_df["order_date_str"] = item_df["order_date"].astype(str)
        if item_search:
            filtered = item_df[item_df["item_name"].str.contains(item_search, case=False, na=False)]
            item_forecast = filtered.head(100).to_dict(orient="records")
        else:
            item_forecast = item_df.head(100).to_dict(orient="records")

    hourly_df = load_parquet_safe("temporal_patterns_hourly.parquet")
    hourly_patterns = hourly_df.to_dict(orient="records") if not hourly_df.empty else []

    r2 = pd.to_numeric(eval_df.get('r2_score', pd.Series(dtype=float)), errors='coerce').dropna()
    mape = pd.to_numeric(eval_df.get('mape_pct', pd.Series(dtype=float)), errors='coerce').dropna()
    summary = {
        "model_architecture": "Persisted evaluation report; see versioned live forecasts for current model",
        "best_r2_score": float(r2.max()) if len(r2) else None,
        "category_mape_pct": float(mape.mean()) if len(mape) else None,
        "items_evaluated": int(item_df['item_id'].nunique()) if 'item_id' in item_df else None,
        "forecast_horizon_days": None,
        "training_orders_volume": None
    }

    return {
        "status": "success",
        "summary": summary,
        "evaluations": evaluations,
        "category_forecast": category_forecast,
        "item_forecast": item_forecast,
        "hourly_patterns": hourly_patterns
    }
