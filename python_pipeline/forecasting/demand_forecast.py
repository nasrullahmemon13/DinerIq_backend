"""
DineIQ Analytics - Temporal Pattern Analysis & Time-Aware Demand Forecasting
Implements SRS Steps 19, 20, 21, and 22:

Step 19 (Temporal & Demand Pattern Identification):
 - Peak hours (hourly volumes, peak windows: lunch & dinner vs off-peak)
 - Peak days (day-of-week ranking, weekly traffic distribution)
 - Weekend patterns (weekday vs weekend volumes, spending, and ticket size)
 - Monthly trends (Jan-Dec 2025 revenue trajectory & MoM growth)
 - Seasonal trends (Winter, Spring, Summer, Fall demand index)
 - Location-specific peaks (peak hours and peak days per restaurant location)
 - Dine-in vs delivery peaks (hourly channel competition and delivery share)

Step 20 (Multi-Granularity Demand Forecasting):
 - Menu items daily demand forecasting
 - Menu categories daily demand forecasting
 - Restaurant locations daily demand forecasting
 - Configurable forecast periods (e.g., 7, 14, 30, 60 days)

Step 21 (Time-Aware Model Validation - Zero Data Leakage):
 - Strict temporal split: Training on earlier historical periods, Testing on strictly later unseen periods
 - Zero lookahead bias: Lags (t-1, t-2, t-7, t-14) and rolling stats computed strictly on past observations
 - Multi-step recursive forecasting without ground-truth future target leakage

Step 22 (Forecast Error Diagnostics & Evaluation):
 - Out-of-sample evaluation on unseen holdout test set using:
   * MAE (Mean Absolute Error)
   * RMSE (Root Mean Squared Error)
   * MAPE (Mean Absolute Percentage Error)
   * R² (Coefficient of Determination)
"""
import os
import sys
import time
import json
import argparse
from typing import Dict, Tuple, Any, List
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

CUBE_PATH = os.path.join(PROJECT_ROOT, "processed_data", "joined", "master_analytical_cube", "master_analytical_cube.parquet")
CLEANED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "forecasting")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "forecasting")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "forecasting")

DEFAULT_FORECAST_HORIZON_DAYS = 30


def load_and_prepare_data() -> pd.DataFrame:
    """Loads cleaned operational records and extracts standard temporal features."""
    t0 = time.time()
    if os.path.exists(CUBE_PATH):
        print(f"[Phase 1] Loading master analytical cube from: {CUBE_PATH}")
        cols = [
            "order_id", "order_date", "order_time", "order_type",
            "location_id", "restaurant_name", "restaurant_city",
            "category_id", "category_name", "item_id", "item_name",
            "quantity", "item_total"
        ]
        df = pd.read_parquet(CUBE_PATH, columns=cols)
    else:
        print("[Phase 1] Master cube not found; joining from processed_data/cleaned/...")
        orders = pd.read_parquet(os.path.join(CLEANED_DIR, "orders", "orders.parquet"))
        items = pd.read_parquet(os.path.join(CLEANED_DIR, "order_items", "order_items.parquet"))
        menu = pd.read_parquet(os.path.join(CLEANED_DIR, "menu_items", "menu_items.parquet"))
        cats = pd.read_parquet(os.path.join(CLEANED_DIR, "menu_categories", "menu_categories.parquet"))
        rests = pd.read_parquet(os.path.join(CLEANED_DIR, "restaurants", "restaurants.parquet"))

        merged = pd.merge(orders, items, on="order_id", how="inner")
        merged = pd.merge(merged, menu, on="item_id", how="inner")
        merged = pd.merge(merged, cats, on="category_id", how="left")
        df = pd.merge(merged, rests, on="location_id", how="left")

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["hour"] = df["order_time"].str.slice(0, 2).astype(int)
    df["day_name"] = df["order_date"].dt.day_name()
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["month"] = df["order_date"].dt.month
    df["month_name"] = df["order_date"].dt.month_name()
    df["quarter"] = df["order_date"].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([4, 5, 6]).astype(int)                

    def assign_season(month: int) -> str:
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    df["season"] = df["month"].apply(assign_season)

    def assign_daypart(hour: int) -> str:
        if 6 <= hour <= 10:
            return "Breakfast / Morning"
        elif 11 <= hour <= 14:
            return "Lunch Peak"
        elif 15 <= hour <= 16:
            return "Afternoon Slump"
        elif 17 <= hour <= 21:
            return "Dinner Peak"
        else:
            return "Late Night / Off-Peak"

    df["daypart"] = df["hour"].apply(assign_daypart)
    print(f"Loaded {len(df):,} line items across {df['order_id'].nunique():,} orders in {time.time() - t0:.2f}s")
    return df


def analyze_temporal_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifies all 7 SRS-mandated temporal and demand dimensions:
    1. Peak Hours
    2. Peak Days
    3. Weekend Patterns
    4. Monthly Trends
    5. Seasonal Trends
    6. Location-Specific Peaks
    7. Dine-In vs Delivery Peaks
    """
    print("\n" + "=" * 80)
    print("SRS Step 19: Comprehensive Temporal & Demand Pattern Identification")
    print("=" * 80)

    hourly = df.groupby("hour").agg(
        total_revenue=("item_total", "sum"),
        total_items=("quantity", "sum"),
        order_count=("order_id", "nunique")
    ).reset_index()
    hourly["revenue_share_pct"] = (hourly["total_revenue"] / hourly["total_revenue"].sum() * 100).round(2)
    hourly["avg_order_value"] = (hourly["total_revenue"] / hourly["order_count"]).round(2)
    hourly = hourly.sort_values(by="total_revenue", ascending=False).reset_index(drop=True)
    top_hours = hourly.head(3)["hour"].tolist()

    daypart_agg = df.groupby("daypart").agg(
        total_revenue=("item_total", "sum"),
        order_count=("order_id", "nunique")
    ).reset_index()
    daypart_agg["revenue_share_pct"] = (daypart_agg["total_revenue"] / daypart_agg["total_revenue"].sum() * 100).round(2)
    daypart_agg = daypart_agg.sort_values(by="total_revenue", ascending=False).reset_index(drop=True)

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_dow = df.groupby(["day_name", "day_of_week"]).agg(
        total_revenue=("item_total", "sum"),
        total_items=("quantity", "sum"),
        order_count=("order_id", "nunique"),
        distinct_dates=("order_date", "nunique")
    ).reset_index()
    daily_dow["avg_daily_revenue"] = (daily_dow["total_revenue"] / daily_dow["distinct_dates"]).round(2)
    daily_dow["avg_daily_orders"] = (daily_dow["order_count"] / daily_dow["distinct_dates"]).round(1)
    daily_dow["revenue_share_pct"] = (daily_dow["total_revenue"] / daily_dow["total_revenue"].sum() * 100).round(2)
    daily_dow = daily_dow.sort_values(by="avg_daily_revenue", ascending=False).reset_index(drop=True)
    top_days = daily_dow.head(3)["day_name"].tolist()

    weekend_df = df.copy()
    weekend_df["period_type"] = np.where(weekend_df["is_weekend"] == 1, "Weekend (Fri-Sun)", "Weekday (Mon-Thu)")
    weekend_summary = weekend_df.groupby("period_type").agg(
        total_revenue=("item_total", "sum"),
        total_items=("quantity", "sum"),
        order_count=("order_id", "nunique"),
        distinct_days=("order_date", "nunique")
    ).reset_index()
    weekend_summary["avg_daily_revenue"] = (weekend_summary["total_revenue"] / weekend_summary["distinct_days"]).round(2)
    weekend_summary["avg_daily_orders"] = (weekend_summary["order_count"] / weekend_summary["distinct_days"]).round(1)
    weekend_summary["avg_basket_spend"] = (weekend_summary["total_revenue"] / weekend_summary["order_count"]).round(2)
    weekend_summary["items_per_order"] = (weekend_summary["total_items"] / weekend_summary["order_count"]).round(2)

    monthly = df.groupby(["month", "month_name"]).agg(
        total_revenue=("item_total", "sum"),
        total_items=("quantity", "sum"),
        order_count=("order_id", "nunique"),
        days_in_month=("order_date", "nunique")
    ).reset_index().sort_values("month")
    monthly["avg_daily_revenue"] = (monthly["total_revenue"] / monthly["days_in_month"]).round(2)
    monthly["avg_order_value"] = (monthly["total_revenue"] / monthly["order_count"]).round(2)
    monthly["mom_growth_pct"] = (monthly["total_revenue"].pct_change() * 100).round(2).fillna(0.0)

    season_order = ["Winter", "Spring", "Summer", "Fall"]
    seasonal = df.groupby("season").agg(
        total_revenue=("item_total", "sum"),
        total_items=("quantity", "sum"),
        order_count=("order_id", "nunique"),
        distinct_days=("order_date", "nunique")
    ).reindex(season_order).reset_index()
    overall_daily_avg = df["item_total"].sum() / df["order_date"].nunique()
    seasonal["daily_avg_revenue"] = (seasonal["total_revenue"] / seasonal["distinct_days"]).round(2)
    seasonal["seasonal_demand_index"] = (seasonal["daily_avg_revenue"] / overall_daily_avg).round(4)
    seasonal["revenue_share_pct"] = (seasonal["total_revenue"] / seasonal["total_revenue"].sum() * 100).round(2)

    loc_hourly = df.groupby(["location_id", "restaurant_name", "restaurant_city", "hour"])["item_total"].sum().reset_index()
    idx_max_hour = loc_hourly.groupby("location_id")["item_total"].idxmax()
    loc_peak_hours = loc_hourly.loc[idx_max_hour][["location_id", "hour", "item_total"]].rename(
        columns={"hour": "peak_hour", "item_total": "peak_hour_revenue"}
    )

    loc_dow = df.groupby(["location_id", "day_name"])["item_total"].sum().reset_index()
    idx_max_dow = loc_dow.groupby("location_id")["item_total"].idxmax()
    loc_peak_dow = loc_dow.loc[idx_max_dow][["location_id", "day_name", "item_total"]].rename(
        columns={"day_name": "peak_day", "item_total": "peak_day_revenue"}
    )

    loc_base = df.groupby(["location_id", "restaurant_name", "restaurant_city"]).agg(
        total_revenue=("item_total", "sum"),
        total_orders=("order_id", "nunique"),
        weekend_revenue=("item_total", lambda s: s[df.loc[s.index, "is_weekend"] == 1].sum())
    ).reset_index()
    loc_summary = pd.merge(loc_base, loc_peak_hours, on="location_id")
    loc_summary = pd.merge(loc_summary, loc_peak_dow, on="location_id")
    loc_summary["weekend_share_pct"] = (loc_summary["weekend_revenue"] / loc_summary["total_revenue"] * 100).round(2)
    loc_summary = loc_summary.sort_values("total_revenue", ascending=False).reset_index(drop=True)

    channel_hourly = df.groupby(["hour", "order_type"]).agg(
        channel_revenue=("item_total", "sum"),
        channel_orders=("order_id", "nunique")
    ).reset_index()

    channel_pivot_rev = channel_hourly.pivot(index="hour", columns="order_type", values="channel_revenue").fillna(0)
    channel_pivot_ord = channel_hourly.pivot(index="hour", columns="order_type", values="channel_orders").fillna(0)

    channel_comparison = pd.DataFrame({
        "hour": channel_pivot_rev.index,
        "dine_in_revenue": channel_pivot_rev.get("DINE_IN", 0).round(2),
        "delivery_revenue": channel_pivot_rev.get("DELIVERY", 0).round(2),
        "takeout_revenue": channel_pivot_rev.get("TAKEOUT", 0).round(2),
        "drive_thru_revenue": channel_pivot_rev.get("DRIVE_THRU", 0).round(2),
        "dine_in_orders": channel_pivot_ord.get("DINE_IN", 0).astype(int),
        "delivery_orders": channel_pivot_ord.get("DELIVERY", 0).astype(int)
    })
    channel_comparison["total_hour_revenue"] = (
        channel_comparison["dine_in_revenue"] +
        channel_comparison["delivery_revenue"] +
        channel_comparison["takeout_revenue"] +
        channel_comparison["drive_thru_revenue"]
    )
    channel_comparison["delivery_revenue_share_pct"] = (
        channel_comparison["delivery_revenue"] / np.maximum(channel_comparison["total_hour_revenue"], 1.0) * 100
    ).round(2)

    dine_in_peak_hour = int(channel_comparison.loc[channel_comparison["dine_in_revenue"].idxmax(), "hour"])
    delivery_peak_hour = int(channel_comparison.loc[channel_comparison["delivery_revenue"].idxmax(), "hour"])

    print(f"Top Peak Hours: {top_hours} (Lunch: 11-13, Dinner: 18-20)")
    print(f"Top Peak Days:  {top_days}")
    print(f"Dine-In Peak:   {dine_in_peak_hour}:00 | Delivery Peak: {delivery_peak_hour}:00")

    return {
        "hourly_patterns": hourly,
        "daypart_summary": daypart_agg,
        "daily_patterns": daily_dow,
        "weekend_patterns": weekend_summary,
        "monthly_trends": monthly,
        "seasonal_trends": seasonal,
        "location_peaks": loc_summary,
        "channel_comparison": channel_comparison,
        "metadata": {
            "top_peak_hours": top_hours,
            "top_peak_days": top_days,
            "dine_in_peak_hour": dine_in_peak_hour,
            "delivery_peak_hour": delivery_peak_hour,
            "overall_annual_revenue": round(float(df["item_total"].sum()), 2),
            "overall_daily_avg_revenue": round(float(overall_daily_avg), 2)
        }
    }


def build_time_series_grid(
    df: pd.DataFrame,
    group_col: str,
    target_metric: str = "quantity",
    secondary_metric: str = "item_total"
) -> pd.DataFrame:
    """
    Constructs a complete contiguous Cartesian time grid (Date x Entity)
    guaranteeing zero date gaps and zero random sampling leakage.
    """
    daily = df.groupby(["order_date", group_col]).agg(
        target=(target_metric, "sum"),
        revenue=(secondary_metric, "sum")
    ).reset_index()

    dates = pd.date_range(daily["order_date"].min(), daily["order_date"].max(), freq="D")
    entities = daily[group_col].unique()

    grid = pd.MultiIndex.from_product([dates, entities], names=["order_date", group_col]).to_frame().reset_index(drop=True)
    merged = pd.merge(grid, daily, on=["order_date", group_col], how="left").fillna({
        "target": 0.0,
        "revenue": 0.0
    })
    return merged.sort_values([group_col, "order_date"]).reset_index(drop=True)


def engineer_strictly_historical_features(ts_df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """
    Generates time-series features strictly derived from past information.
    SRS Step 21 MANDATORY REQUIREMENT: No future information in training features!
     - Lags: t-1, t-2, t-7 (weekly), t-14 (bi-weekly)
     - Rolling windows: Computed strictly on lag_1 (excluding current day t)
     - Calendar attributes: Known deterministic time coordinates
    """
    df = ts_df.copy()

    df["lag_1"] = df.groupby(entity_col)["target"].shift(1)
    df["lag_2"] = df.groupby(entity_col)["target"].shift(2)
    df["lag_7"] = df.groupby(entity_col)["target"].shift(7)
    df["lag_14"] = df.groupby(entity_col)["target"].shift(14)

    grouped_lag1 = df.groupby(entity_col)["lag_1"]
    df["rolling_mean_7"] = grouped_lag1.transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["rolling_mean_14"] = grouped_lag1.transform(lambda s: s.rolling(14, min_periods=1).mean())
    df["rolling_std_7"] = grouped_lag1.transform(lambda s: s.rolling(7, min_periods=1).std()).fillna(0.0)

    df["dow"] = df["order_date"].dt.dayofweek
    df["day_of_month"] = df["order_date"].dt.day
    df["month"] = df["order_date"].dt.month
    df["quarter"] = df["order_date"].dt.quarter
    df["is_weekend"] = (df["dow"] >= 4).astype(int)                

    df["entity_code"] = df[entity_col].astype("category").cat.codes

    return df.dropna().reset_index(drop=True)


def temporal_train_test_split(
    features_df: pd.DataFrame,
    forecast_horizon_days: int = DEFAULT_FORECAST_HORIZON_DAYS
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Enforces SRS Step 21: Strict chronological split.
    Training = earlier historical periods
    Testing  = strictly later unseen periods
    MANDATORY ASSERTION: max(train_date) < min(test_date). Zero data leakage!
    """
    max_date = features_df["order_date"].max()
    split_date = max_date - pd.Timedelta(days=forecast_horizon_days)

    train_data = features_df[features_df["order_date"] <= split_date].copy()
    test_data = features_df[features_df["order_date"] > split_date].copy()

    train_max = train_data["order_date"].max()
    test_min = test_data["order_date"].min()
    assert train_max < test_min, (
        f"DATA LEAKAGE DETECTED! Training max date ({train_max}) "
        f"must strictly precede Test min date ({test_min})."
    )

    print(f"\n[Step 21 - Temporal Validation Split] Configured Horizon: {forecast_horizon_days} Days")
    print(f"  Training Set: {train_data['order_date'].min().date()} to {train_max.date()} ({len(train_data):,} records)")
    print(f"  Test Set:     {test_min.date()} to {test_data['order_date'].max().date()} ({len(test_data):,} records)")
    print(f"  Validation Integrity: Passed (0% overlap, 0% random leakage)")

    return train_data, test_data, split_date


def train_and_evaluate_forecaster(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    entity_col: str,
    granularity_name: str,
    feature_cols: List[str]
) -> Tuple[HistGradientBoostingRegressor, pd.DataFrame, Dict[str, float]]:
    """
    Trains a Gradient Boosted Regressor on earlier periods,
    evaluates multi-step predictions on the unseen test period (Step 22),
    and computes MAE, RMSE, MAPE, and R².
    """
    model = HistGradientBoostingRegressor(
        categorical_features=[feature_cols.index("entity_code")],
        loss="squared_error",
        learning_rate=0.08,
        max_iter=120,
        random_state=42
    )

    model.fit(train_df[feature_cols], train_df["target"])

    test_dates = sorted(test_df["order_date"].unique())
    entities = sorted(train_df[entity_col].unique())

    history_buffers = {
        ent: train_df[train_df[entity_col] == ent]["target"].tolist()
        for ent in entities
    }
    entity_code_map = dict(zip(train_df[entity_col], train_df["entity_code"]))

    predictions = []
    for cur_date in test_dates:
        step_actuals = test_df[test_df["order_date"] == cur_date]
        step_preds = {}

        for ent in entities:
            hist = history_buffers[ent]
            feat_row = {
                "entity_code": entity_code_map[ent],
                "lag_1": hist[-1],
                "lag_2": hist[-2],
                "lag_7": hist[-7] if len(hist) >= 7 else hist[-1],
                "lag_14": hist[-14] if len(hist) >= 14 else hist[-1],
                "rolling_mean_7": float(np.mean(hist[-7:])),
                "rolling_mean_14": float(np.mean(hist[-14:])),
                "rolling_std_7": float(np.std(hist[-7:])) if len(hist) >= 7 else 0.0,
                "dow": cur_date.dayofweek,
                "day_of_month": cur_date.day,
                "month": cur_date.month,
                "quarter": cur_date.quarter,
                "is_weekend": int(cur_date.dayofweek >= 4)
            }

            pred_df = pd.DataFrame([feat_row])
            y_pred = float(model.predict(pred_df[feature_cols])[0])
            y_pred = max(0.0, y_pred)                                              
            step_preds[ent] = y_pred

            actual_row = step_actuals[step_actuals[entity_col] == ent]
            y_true = float(actual_row["target"].iloc[0]) if len(actual_row) > 0 else 0.0

            predictions.append({
                "order_date": cur_date,
                entity_col: ent,
                "actual_demand": round(y_true, 2),
                "predicted_demand": round(y_pred, 2),
                "forecast_error": round(y_true - y_pred, 2),
                "granularity": granularity_name
            })

        for ent, pval in step_preds.items():
            history_buffers[ent].append(pval)

    pred_df = pd.DataFrame(predictions)

    y_true = pred_df["actual_demand"].values
    y_pred = pred_df["predicted_demand"].values

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100.0)
    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "granularity": granularity_name,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_pct": round(mape, 2),
        "r2_score": round(r2, 4),
        "eval_sample_count": len(pred_df)
    }

    print(f"  [Step 22 - {granularity_name}] MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}% | R²: {r2:.4f}")
    return model, pred_df, metrics


def run_demand_forecasting_pipeline(forecast_horizon_days: int = DEFAULT_FORECAST_HORIZON_DAYS) -> Dict[str, Any]:
    """
    Executes the entire end-to-end forecasting pipeline for Steps 19-22:
     - Configurable forecast horizon (SRS requirement)
     - Identification of all 7 temporal pattern dimensions
     - Time-aware validation and multi-granularity demand forecasting
     - Evaluation metrics and comprehensive reports
    """
    pipeline_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 80)
    print("DineIQ Analytics - SRS Steps 19-22: Demand Forecasting Pipeline")
    print(f"Configured Forecast Horizon: {forecast_horizon_days} Days")
    print("=" * 80)

    raw_df = load_and_prepare_data()

    patterns = analyze_temporal_patterns(raw_df)

    feature_cols = [
        "entity_code", "lag_1", "lag_2", "lag_7", "lag_14",
        "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
        "dow", "day_of_month", "month", "quarter", "is_weekend"
    ]

    all_evaluations = []

    print("\n[Step 20 & 21] Forecasting Category-Level Demand...")
    cat_grid = build_time_series_grid(raw_df, group_col="category_name", target_metric="quantity")
    cat_features = engineer_strictly_historical_features(cat_grid, entity_col="category_name")
    cat_train, cat_test, _ = temporal_train_test_split(cat_features, forecast_horizon_days)
    cat_model, cat_preds, cat_metrics = train_and_evaluate_forecaster(
        cat_train, cat_test, entity_col="category_name",
        granularity_name="Menu Categories", feature_cols=feature_cols
    )
    all_evaluations.append(cat_metrics)

    print("\n[Step 20 & 21] Forecasting Location-Level Demand...")
    loc_grid = build_time_series_grid(raw_df, group_col="restaurant_name", target_metric="quantity")
    loc_features = engineer_strictly_historical_features(loc_grid, entity_col="restaurant_name")
    loc_train, loc_test, _ = temporal_train_test_split(loc_features, forecast_horizon_days)
    loc_model, loc_preds, loc_metrics = train_and_evaluate_forecaster(
        loc_train, loc_test, entity_col="restaurant_name",
        granularity_name="Restaurant Locations", feature_cols=feature_cols
    )
    all_evaluations.append(loc_metrics)

    print("\n[Step 20 & 21] Forecasting Menu Item-Level Demand...")
    item_grid = build_time_series_grid(raw_df, group_col="item_name", target_metric="quantity")
    item_features = engineer_strictly_historical_features(item_grid, entity_col="item_name")
    item_train, item_test, _ = temporal_train_test_split(item_features, forecast_horizon_days)
    item_model, item_preds, item_metrics = train_and_evaluate_forecaster(
        item_train, item_test, entity_col="item_name",
        granularity_name="Menu Items", feature_cols=feature_cols
    )
    all_evaluations.append(item_metrics)

    agg_actuals = cat_preds.groupby("order_date")[["actual_demand", "predicted_demand"]].sum().reset_index()
    agg_mae = float(mean_absolute_error(agg_actuals["actual_demand"], agg_actuals["predicted_demand"]))
    agg_rmse = float(np.sqrt(mean_squared_error(agg_actuals["actual_demand"], agg_actuals["predicted_demand"])))
    agg_mape = float(np.mean(np.abs((agg_actuals["actual_demand"] - agg_actuals["predicted_demand"]) / agg_actuals["actual_demand"])) * 100.0)
    agg_r2 = float(r2_score(agg_actuals["actual_demand"], agg_actuals["predicted_demand"]))

    chain_metrics = {
        "granularity": "Total Chain Aggregate",
        "mae": round(agg_mae, 2),
        "rmse": round(agg_rmse, 2),
        "mape_pct": round(agg_mape, 2),
        "r2_score": round(agg_r2, 4),
        "eval_sample_count": len(agg_actuals)
    }
    all_evaluations.append(chain_metrics)
    print(f"  [Step 22 - Total Chain Aggregate] MAE: {agg_mae:.2f} | RMSE: {agg_rmse:.2f} | MAPE: {agg_mape:.2f}% | R²: {agg_r2:.4f}")

    eval_df = pd.DataFrame(all_evaluations)

    print("\n[Persistence] Saving serialized models to models/forecasting/...")
    cat_model_path = os.path.join(MODELS_DIR, "category_demand_forecaster.joblib")
    loc_model_path = os.path.join(MODELS_DIR, "location_demand_forecaster.joblib")
    item_model_path = os.path.join(MODELS_DIR, "item_demand_forecaster.joblib")

    joblib.dump(cat_model, cat_model_path)
    joblib.dump(loc_model, loc_model_path)
    joblib.dump(item_model, item_model_path)

    print("[Persistence] Saving forecast predictions and pattern datasets to processed_data/forecasting/...")
    patterns["hourly_patterns"].to_parquet(os.path.join(OUTPUT_DIR, "temporal_patterns_hourly.parquet"), compression="snappy")
    patterns["hourly_patterns"].to_csv(os.path.join(OUTPUT_DIR, "temporal_patterns_hourly.csv"), index=False)

    patterns["daily_patterns"].to_parquet(os.path.join(OUTPUT_DIR, "temporal_patterns_daily.parquet"), compression="snappy")
    patterns["daily_patterns"].to_csv(os.path.join(OUTPUT_DIR, "temporal_patterns_daily.csv"), index=False)

    patterns["monthly_trends"].to_parquet(os.path.join(OUTPUT_DIR, "temporal_patterns_monthly.parquet"), compression="snappy")
    patterns["monthly_trends"].to_csv(os.path.join(OUTPUT_DIR, "temporal_patterns_monthly.csv"), index=False)

    patterns["seasonal_trends"].to_parquet(os.path.join(OUTPUT_DIR, "temporal_patterns_seasonal.parquet"), compression="snappy")
    patterns["seasonal_trends"].to_csv(os.path.join(OUTPUT_DIR, "temporal_patterns_seasonal.csv"), index=False)

    patterns["location_peaks"].to_parquet(os.path.join(OUTPUT_DIR, "temporal_patterns_locations.parquet"), compression="snappy")
    patterns["location_peaks"].to_csv(os.path.join(OUTPUT_DIR, "temporal_patterns_locations.csv"), index=False)

    patterns["channel_comparison"].to_parquet(os.path.join(OUTPUT_DIR, "temporal_patterns_channels.parquet"), compression="snappy")
    patterns["channel_comparison"].to_csv(os.path.join(OUTPUT_DIR, "temporal_patterns_channels.csv"), index=False)

    cat_preds.to_parquet(os.path.join(OUTPUT_DIR, "category_demand_forecast.parquet"), compression="snappy")
    cat_preds.to_csv(os.path.join(OUTPUT_DIR, "category_demand_forecast.csv"), index=False)

    loc_preds.to_parquet(os.path.join(OUTPUT_DIR, "location_demand_forecast.parquet"), compression="snappy")
    loc_preds.to_csv(os.path.join(OUTPUT_DIR, "location_demand_forecast.csv"), index=False)

    item_preds.to_parquet(os.path.join(OUTPUT_DIR, "item_demand_forecast.parquet"), compression="snappy")
    item_preds.to_csv(os.path.join(OUTPUT_DIR, "item_demand_forecast.csv"), index=False)

    eval_df.to_parquet(os.path.join(OUTPUT_DIR, "forecast_evaluations.parquet"), compression="snappy")
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "forecast_evaluations.csv"), index=False)

    print("[Reports] Compiling demand forecasting reports in reports/forecasting/...")
    md_report_path = os.path.join(REPORTS_DIR, "demand_forecasting_report.md")
    json_report_path = os.path.join(REPORTS_DIR, "demand_forecasting_report.json")

    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "pipeline_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "forecast_horizon_days": forecast_horizon_days,
            "temporal_metadata": patterns["metadata"],
            "model_evaluations": all_evaluations,
            "hourly_summary": patterns["hourly_patterns"].head(10).to_dict(orient="records"),
            "daily_summary": patterns["daily_patterns"].to_dict(orient="records"),
            "seasonal_summary": patterns["seasonal_trends"].to_dict(orient="records"),
            "category_forecast_sample": cat_preds.head(15).to_dict(orient="records")
        }, f, indent=2, default=str)

    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# DineIQ Analytics: Temporal Pattern Analysis & Time-Aware Demand Forecasting\n\n")
        f.write(f"**Execution Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Configured Forecast Horizon:** {forecast_horizon_days} Days (Configurable per SRS Step 20)\n")
        f.write(f"**Validation Framework:** Strict Chronological Train/Test Split (SRS Step 21 Invariant: Zero Future Leakage)\n\n")

        f.write("## 1. Executive Summary & Evaluation Matrix (Step 22)\n\n")
        f.write("All models were trained strictly on earlier periods and evaluated on the out-of-sample holdout test window:\n\n")
        f.write("| Granularity Level | Target Dimension | MAE | RMSE | MAPE (%) | R² Score | Out-of-Sample Window |\n")
        f.write("|---|---|---:|---:|---:|---:|---|\n")
        for ev in all_evaluations:
            f.write(f"| **{ev['granularity']}** | Daily Quantity Sold | {ev['mae']:.2f} | {ev['rmse']:.2f} | **{ev['mape_pct']:.2f}%** | **{ev['r2_score']:.4f}** | Last {forecast_horizon_days} Days |\n")
        f.write("\n")

        f.write("## 2. Temporal Pattern Identification (Step 19)\n\n")

        f.write("### A. Peak Hours Analysis\n\n")
        f.write(f"- **Top Peak Hours:** {', '.join([f'{h}:00' for h in patterns['metadata']['top_peak_hours']])}\n")
        f.write("- **Lunch Peak Window:** 11:00 - 14:00 (driven by business lunch crowd)\n")
        f.write("- **Dinner Peak Window:** 17:00 - 21:00 (driven by dining groups and dinner entrees)\n")
        f.write("- **Off-Peak Window:** 00:00 - 06:00\n\n")
        f.write("| Hour | Order Count | Total Items | Total Revenue ($) | Share of Revenue (%) | Average Order Value ($) |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for _, r in patterns["hourly_patterns"].head(10).iterrows():
            f.write(f"| {int(r['hour'])}:00 | {int(r['order_count']):,} | {int(r['total_items']):,} | ${r['total_revenue']:,.2f} | {r['revenue_share_pct']:.2f}% | ${r['avg_order_value']:.2f} |\n")
        f.write("\n")

        f.write("### B. Peak Days Analysis\n\n")
        f.write(f"- **Top Peak Days:** {', '.join(patterns['metadata']['top_peak_days'])}\n\n")
        f.write("| Day of Week | Total Revenue ($) | Daily Avg Revenue ($) | Total Orders | Daily Avg Orders | Share of Week (%) |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for _, r in patterns["daily_patterns"].iterrows():
            f.write(f"| **{r['day_name']}** | ${r['total_revenue']:,.2f} | ${r['avg_daily_revenue']:,.2f} | {int(r['order_count']):,} | {r['avg_daily_orders']:,.1f} | {r['revenue_share_pct']:.2f}% |\n")
        f.write("\n")

        f.write("### C. Weekend vs Weekday Patterns\n\n")
        f.write("| Period Type | Total Revenue ($) | Daily Avg Revenue ($) | Total Orders | Daily Avg Orders | Avg Basket ($) | Items / Order |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for _, r in patterns["weekend_patterns"].iterrows():
            f.write(f"| **{r['period_type']}** | ${r['total_revenue']:,.2f} | ${r['avg_daily_revenue']:,.2f} | {int(r['order_count']):,} | {r['avg_daily_orders']:,.1f} | ${r['avg_basket_spend']:.2f} | {r['items_per_order']:.2f} |\n")
        f.write("\n")

        f.write("### D. Monthly Trends (Jan - Dec 2025)\n\n")
        f.write("| Month | Total Revenue ($) | Daily Avg Revenue ($) | Order Count | Avg Order Value ($) | MoM Growth (%) |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for _, r in patterns["monthly_trends"].iterrows():
            f.write(f"| **{r['month_name']}** | ${r['total_revenue']:,.2f} | ${r['avg_daily_revenue']:,.2f} | {int(r['order_count']):,} | ${r['avg_order_value']:.2f} | {r['mom_growth_pct']:+.2f}% |\n")
        f.write("\n")

        f.write("### E. Seasonal Trends\n\n")
        f.write("| Season | Months Included | Total Revenue ($) | Daily Avg Revenue ($) | Seasonal Demand Index | Share (%) |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for _, r in patterns["seasonal_trends"].iterrows():
            f.write(f"| **{r['season']}** | Dec-Feb / Mar-May / Jun-Aug / Sep-Nov | ${r['total_revenue']:,.2f} | ${r['daily_avg_revenue']:,.2f} | **{r['seasonal_demand_index']:.4f}** | {r['revenue_share_pct']:.2f}% |\n")
        f.write("\n")

        f.write("### F. Location-Specific Peaks (20 Locations)\n\n")
        f.write("| Location Name | City | Total Revenue ($) | Peak Hour | Peak Day | Weekend Share (%) |\n")
        f.write("|---|---|---:|---:|---|---:|\n")
        for _, r in patterns["location_peaks"].head(10).iterrows():
            f.write(f"| **{r['restaurant_name']}** | {r['restaurant_city']} | ${r['total_revenue']:,.2f} | {int(r['peak_hour'])}:00 | {r['peak_day']} | {r['weekend_share_pct']:.2f}% |\n")
        f.write("\n")

        f.write("### G. Dine-In vs Delivery Channel Peaks\n\n")
        f.write(f"- **Dine-In Peak Hour:** {patterns['metadata']['dine_in_peak_hour']}:00 (Dinner crowd table seating)\n")
        f.write(f"- **Delivery Peak Hour:** {patterns['metadata']['delivery_peak_hour']}:00 (At-home lunch & evening deliveries)\n\n")
        f.write("| Hour | Dine-In Revenue ($) | Delivery Revenue ($) | Takeout Revenue ($) | Delivery Share (%) |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for _, r in patterns["channel_comparison"].iloc[10:22].iterrows():
            f.write(f"| {int(r['hour'])}:00 | ${r['dine_in_revenue']:,.2f} | ${r['delivery_revenue']:,.2f} | ${r['takeout_revenue']:,.2f} | {r['delivery_revenue_share_pct']:.2f}% |\n")
        f.write("\n")

        f.write("## 3. Time-Aware Validation & Anti-Leakage Compliance (Step 21)\n\n")
        f.write("- **Zero Random Leakage:** Data is partitioned strictly on calendar dates (`train <= split_date < test`). Shuffling is disabled.\n")
        f.write("- **Zero Target Leakage in Features:** Lag features ($t-1, t-2, t-7, t-14$) and rolling statistics ($7\\text{d}, 14\\text{d}$) are calculated strictly on shifted prior observations.\n")
        f.write("- **Multi-Step Recursive Horizon:** During out-of-sample test evaluation, predicted values are fed recursively into lag buffers, ensuring true unseen forecast simulation.\n\n")

        f.write("## 4. Multi-Granularity Demand Forecasts (Step 20)\n\n")
        f.write("Sample category-level forecasts for the out-of-sample period:\n\n")
        f.write("| Order Date | Menu Category | Actual Demand | Predicted Demand | Error | Accuracy |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for _, r in cat_preds.head(10).iterrows():
            pct_err = abs(r['forecast_error']) / max(r['actual_demand'], 1.0) * 100
            f.write(f"| {str(r['order_date'])[:10]} | **{r['category_name']}** | {r['actual_demand']:.1f} | {r['predicted_demand']:.1f} | {r['forecast_error']:+.1f} | {100-pct_err:.1f}% |\n")
        f.write("\n")

    print(f"\n[OK] Reports saved to {md_report_path} and {json_report_path}")
    print(f"Pipeline executed in {time.time() - pipeline_start:.2f} seconds!")
    print("=" * 80)

    return {
        "patterns": patterns,
        "evaluations": eval_df,
        "category_forecast": cat_preds,
        "location_forecast": loc_preds,
        "item_forecast": item_preds
    }


def main():
    parser = argparse.ArgumentParser(description="DineIQ Demand Forecasting Pipeline (SRS Steps 19-22)")
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_FORECAST_HORIZON_DAYS,
        help="Configurable forecast horizon in days (default: 30)"
    )
    args = parser.parse_args()
    run_demand_forecasting_pipeline(forecast_horizon_days=args.horizon)


if __name__ == "__main__":
    main()
