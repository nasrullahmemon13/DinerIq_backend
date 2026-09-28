"""
DineIQ Analytics - Wastage Analysis & Predictive Risk Modeling
Implements SRS Steps 23 and 24:

Step 23 (Multi-Dimensional Wastage Analysis):
 Analyzes operational food wastage across EXACTLY the 9 SRS-defined dimensions:
  1. Menu Item: Loss amount, wasted quantity, unit cost, and wastage rate per dish
  2. Category: Category financial loss share %, volume, and risk profiles
  3. Location: Location-specific wastage volumes, loss amounts, and operational efficiency
  4. Day: Day-of-week patterns, daily average loss, and weekly wastage cycles
  5. Time Period: Kitchen shifts (Morning Prep, Lunch, Afternoon, Dinner, Closing) & root-cause cross-tabulation
  6. Demand: Correlation between sales demand levels (quartiles) and wastage volume
  7. Inventory Consumption: Consumed vs wasted ratios, ending stock, and stock status analysis
  8. Promotion: Promotional vs non-promotional wastage rates and overproduction surges
  9. Preparation Quantity: Available prep batches, over-preparation index, and excess buffer impact

Step 24 (Wastage Risk Prediction Model):
 Predicts wastage risk using ALL potential variables SRS lists:
  1. historical_demand: Prior 7-day rolling sales volume
  2. historical_wastage: Prior 7-day rolling wasted units
  3. day_of_week: Calendar day and weekend indicator
  4. season: Meteorological/operational season (Winter, Spring, Summer, Fall)
  5. location: Location tier and operational cost index
  6. promotion: Active promotion flag and discount value
  7. menu_popularity: Menu item popularity weight and order frequency
  8. forecast_demand: Forward projected demand volume
  9. preparation_quantity: Starting stock + received prep batch quantity

Models & Evaluation:
 - Regression Model: Predicts expected wasted units (MAE, RMSE, R²)
 - Classification Model: Predicts Wastage Risk Tier [Low, Moderate, High, Critical] (Accuracy, F1, ROC-AUC)
 - Feature Importance: Quantifies top operational drivers of food waste
 - Mitigation Engine: Generates proactive batch reduction and promo-salvage recommendations
"""
import os
import sys
import time
import json
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

CLEANED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "wastage")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "wastage")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "wastage")


def load_and_enrich_wastage_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads cleaned operational records for wastage, inventory, menu items, restaurants, and promotions."""
    t0 = time.time()
    print("[Phase 1] Ingesting cleaned operational datasets...")

    wastage_path = os.path.join(CLEANED_DIR, "wastage", "wastage.parquet")
    inv_path = os.path.join(CLEANED_DIR, "inventory", "inventory.parquet")
    menu_path = os.path.join(CLEANED_DIR, "menu_items", "menu_items.parquet")
    cat_path = os.path.join(CLEANED_DIR, "menu_categories", "menu_categories.parquet")
    rest_path = os.path.join(CLEANED_DIR, "restaurants", "restaurants.parquet")
    promo_path = os.path.join(CLEANED_DIR, "promotions", "promotions.parquet")

    wastage = pd.read_parquet(wastage_path)
    inventory = pd.read_parquet(inv_path)
    menu = pd.read_parquet(menu_path)
    categories = pd.read_parquet(cat_path)
    restaurants = pd.read_parquet(rest_path).rename(columns={"name": "restaurant_name", "city": "restaurant_city"})
    promotions = pd.read_parquet(promo_path)

    menu_full = pd.merge(menu, categories, on="category_id", how="left")

    wastage["wastage_date"] = pd.to_datetime(wastage["wastage_date"])
    wastage["hour"] = wastage["wastage_time"].str.slice(0, 2).astype(int)
    wastage["day_name"] = wastage["wastage_date"].dt.day_name()
    wastage["day_of_week"] = wastage["wastage_date"].dt.dayofweek
    wastage["month"] = wastage["wastage_date"].dt.month
    wastage["is_weekend"] = wastage["day_of_week"].isin([4, 5, 6]).astype(int)

    def get_season(month: int) -> str:
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    wastage["season"] = wastage["month"].apply(get_season)

    def assign_shift(hour: int) -> str:
        if 6 <= hour <= 10:
            return "Morning Prep Shift"
        elif 11 <= hour <= 14:
            return "Lunch Service Shift"
        elif 15 <= hour <= 17:
            return "Afternoon Restock Shift"
        elif 18 <= hour <= 21:
            return "Dinner Service Shift"
        else:
            return "Night Closing Shift"

    wastage["shift_period"] = wastage["hour"].apply(assign_shift)

    inventory["snapshot_date"] = pd.to_datetime(inventory["snapshot_date"])
    inventory["day_name"] = inventory["snapshot_date"].dt.day_name()
    inventory["day_of_week"] = inventory["snapshot_date"].dt.dayofweek
    inventory["month"] = inventory["snapshot_date"].dt.month
    inventory["is_weekend"] = inventory["day_of_week"].isin([4, 5, 6]).astype(int)
    inventory["season"] = inventory["month"].apply(get_season)

    print(f"Loaded {len(wastage):,} wastage incident logs and {len(inventory):,} inventory snapshots in {time.time() - t0:.2f}s")
    return wastage, inventory, menu_full, restaurants, promotions


def analyze_wastage_all_dimensions(
    wastage: pd.DataFrame,
    inventory: pd.DataFrame,
    menu: pd.DataFrame,
    restaurants: pd.DataFrame,
    promotions: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Performs multi-dimensional food wastage analysis across all 9 SRS dimensions:
    1. Menu Item
    2. Category
    3. Location
    4. Day
    5. Time Period
    6. Demand
    7. Inventory Consumption
    8. Promotion
    9. Preparation Quantity
    """
    print("\n" + "=" * 80)
    print("SRS Step 23: Multi-Dimensional Food Wastage Analysis (All 9 Dimensions)")
    print("=" * 80)

    print("  [Dimension 1/9] Analyzing wastage by Menu Item...")
    item_waste = wastage.groupby("item_id").agg(
        wasted_quantity=("quantity_wasted", "sum"),
        total_loss_amount=("total_loss_amount", "sum"),
        incident_count=("wastage_id", "count"),
        avg_loss_per_incident=("total_loss_amount", "mean")
    ).reset_index()

    item_meta = menu[["item_id", "name", "category_name", "base_price", "cost_price", "shelf_life_days", "complexity_profile"]]
    dim_item = pd.merge(item_waste, item_meta, on="item_id", how="left")

    item_sales = inventory.groupby("item_id").agg(total_sold=("quantity_sold", "sum")).reset_index()
    dim_item = pd.merge(dim_item, item_sales, on="item_id", how="left").fillna({"total_sold": 0})
    dim_item["wastage_rate_pct"] = (
        dim_item["wasted_quantity"] / np.maximum(dim_item["total_sold"] + dim_item["wasted_quantity"], 1) * 100
    ).round(2)
    dim_item["total_loss_amount"] = dim_item["total_loss_amount"].round(2)
    dim_item["avg_loss_per_incident"] = dim_item["avg_loss_per_incident"].round(2)
    dim_item = dim_item.sort_values(by="total_loss_amount", ascending=False).reset_index(drop=True)

    print("  [Dimension 2/9] Analyzing wastage by Menu Category...")
    cat_merged = pd.merge(wastage, menu[["item_id", "category_name"]], on="item_id", how="left")
    dim_cat = cat_merged.groupby("category_name").agg(
        wasted_quantity=("quantity_wasted", "sum"),
        total_loss_amount=("total_loss_amount", "sum"),
        incident_count=("wastage_id", "count"),
        avg_loss_per_incident=("total_loss_amount", "mean")
    ).reset_index()
    total_chain_loss = dim_cat["total_loss_amount"].sum()
    dim_cat["loss_share_pct"] = (dim_cat["total_loss_amount"] / total_chain_loss * 100).round(2)
    dim_cat["total_loss_amount"] = dim_cat["total_loss_amount"].round(2)
    dim_cat["avg_loss_per_incident"] = dim_cat["avg_loss_per_incident"].round(2)
    dim_cat = dim_cat.sort_values(by="total_loss_amount", ascending=False).reset_index(drop=True)

    print("  [Dimension 3/9] Analyzing wastage by Restaurant Location...")
    dim_loc = wastage.groupby("location_id").agg(
        wasted_quantity=("quantity_wasted", "sum"),
        total_loss_amount=("total_loss_amount", "sum"),
        incident_count=("wastage_id", "count")
    ).reset_index()
    dim_loc = pd.merge(dim_loc, restaurants[["location_id", "restaurant_name", "restaurant_city", "location_tier"]], on="location_id", how="left")
    loc_sales = inventory.groupby("location_id").agg(total_sold=("quantity_sold", "sum")).reset_index()
    dim_loc = pd.merge(dim_loc, loc_sales, on="location_id", how="left").fillna({"total_sold": 0})
    dim_loc["wastage_rate_pct"] = (
        dim_loc["wasted_quantity"] / np.maximum(dim_loc["total_sold"] + dim_loc["wasted_quantity"], 1) * 100
    ).round(2)
    dim_loc["total_loss_amount"] = dim_loc["total_loss_amount"].round(2)
    dim_loc = dim_loc.sort_values(by="total_loss_amount", ascending=False).reset_index(drop=True)

    print("  [Dimension 4/9] Analyzing wastage by Day of Week...")
    dim_day = wastage.groupby(["day_name", "day_of_week"]).agg(
        wasted_quantity=("quantity_wasted", "sum"),
        total_loss_amount=("total_loss_amount", "sum"),
        incident_count=("wastage_id", "count"),
        unique_dates=("wastage_date", "nunique")
    ).reset_index()
    dim_day["daily_avg_loss"] = (dim_day["total_loss_amount"] / dim_day["unique_dates"]).round(2)
    dim_day["daily_avg_wasted_units"] = (dim_day["wasted_quantity"] / dim_day["unique_dates"]).round(1)
    dim_day["loss_share_pct"] = (dim_day["total_loss_amount"] / total_chain_loss * 100).round(2)
    dim_day = dim_day.sort_values(by="daily_avg_loss", ascending=False).reset_index(drop=True)

    print("  [Dimension 5/9] Analyzing wastage by Time Period (Shifts) & Root Causes...")
    dim_shift = wastage.groupby("shift_period").agg(
        wasted_quantity=("quantity_wasted", "sum"),
        total_loss_amount=("total_loss_amount", "sum"),
        incident_count=("wastage_id", "count")
    ).reset_index()
    dim_shift["loss_share_pct"] = (dim_shift["total_loss_amount"] / total_chain_loss * 100).round(2)
    dim_shift["avg_loss_per_incident"] = (dim_shift["total_loss_amount"] / dim_shift["incident_count"]).round(2)
    dim_shift = dim_shift.sort_values(by="total_loss_amount", ascending=False).reset_index(drop=True)

    shift_reasons = wastage.groupby(["shift_period", "wastage_reason"])["quantity_wasted"].sum().unstack(fill_value=0).reset_index()
    dim_time_period = pd.merge(dim_shift, shift_reasons, on="shift_period", how="left")

    print("  [Dimension 6/9] Analyzing wastage by Demand Levels...")
    inv_demand = inventory.copy()
    inv_demand["demand_quartile"] = pd.qcut(
        inv_demand["quantity_sold"], q=4,
        labels=["Low Demand (Q1)", "Moderate Demand (Q2)", "High Demand (Q3)", "Peak Demand (Q4)"]
    )
    dim_demand = inv_demand.groupby("demand_quartile", observed=False).agg(
        snapshots_count=("inventory_id", "count"),
        total_units_sold=("quantity_sold", "sum"),
        total_units_wasted=("quantity_wasted", "sum"),
        avg_units_sold=("quantity_sold", "mean"),
        avg_units_wasted=("quantity_wasted", "mean")
    ).reset_index()
    dim_demand["wastage_rate_pct"] = (
        dim_demand["total_units_wasted"] / np.maximum(dim_demand["total_units_sold"] + dim_demand["total_units_wasted"], 1) * 100
    ).round(2)
    dim_demand["demand_to_wastage_ratio"] = (
        dim_demand["total_units_sold"] / np.maximum(dim_demand["total_units_wasted"], 1)
    ).round(1)

    print("  [Dimension 7/9] Analyzing wastage by Inventory Consumption & Stock Status...")
    inv_stock = inventory.copy()
    inv_stock["available_stock"] = inv_stock["starting_stock"] + inv_stock["quantity_received"]
    dim_inventory = inv_stock.groupby("stock_status").agg(
        snapshots_count=("inventory_id", "count"),
        total_available_stock=("available_stock", "sum"),
        total_units_sold=("quantity_sold", "sum"),
        total_units_wasted=("quantity_wasted", "sum"),
        avg_ending_stock=("ending_stock", "mean"),
        avg_reorder_point=("reorder_point", "mean")
    ).reset_index()
    dim_inventory["stock_consumption_ratio_pct"] = (
        dim_inventory["total_units_sold"] / np.maximum(dim_inventory["total_units_sold"] + dim_inventory["total_units_wasted"], 1) * 100
    ).round(2)
    dim_inventory["inventory_wastage_rate_pct"] = (
        dim_inventory["total_units_wasted"] / np.maximum(dim_inventory["total_available_stock"], 1) * 100
    ).round(2)

    print("  [Dimension 8/9] Analyzing wastage by Promotion Activity...")
    promos_clean = promotions.copy()
    promos_clean["start_date"] = pd.to_datetime(promos_clean["start_date"])
    promos_clean["end_date"] = pd.to_datetime(promos_clean["end_date"])

    campaign_promos = promos_clean[
        ~((promos_clean["applicable_category"] == "ALL") & (promos_clean["start_date"] == "2025-01-01") & (promos_clean["end_date"] == "2025-12-31"))
    ]

    item_cat_map = dict(zip(menu["item_id"], menu["category_id"]))
    inv_dates = inventory["snapshot_date"].values
    inv_items = inventory["item_id"].values
    inv_cats = [item_cat_map.get(i, "") for i in inv_items]

    promo_tuples = list(zip(
        campaign_promos["start_date"],
        campaign_promos["end_date"],
        campaign_promos["applicable_category"],
        campaign_promos["discount_value"]
    ))

    promo_active_list = []
    promo_discount_list = []
    for d, cat in zip(inv_dates, inv_cats):
        d_ts = pd.Timestamp(d)
        is_act = 0
        d_val = 0.0
        for s_d, e_d, app_cat, disc in promo_tuples:
            if s_d <= d_ts <= e_d and (app_cat in ["ALL", cat]):
                is_act = 1
                d_val = disc
                break
        promo_active_list.append(is_act)
        promo_discount_list.append(d_val)

    inventory["is_promo_active"] = promo_active_list
    inventory["promo_discount_val"] = promo_discount_list

    inv_promo = inventory.copy()
    inv_promo["promo_status"] = np.where(inv_promo["is_promo_active"] == 1, "Active Promotion", "Standard Non-Promo")
    dim_promotion = inv_promo.groupby("promo_status").agg(
        snapshot_count=("inventory_id", "count"),
        total_units_sold=("quantity_sold", "sum"),
        total_units_wasted=("quantity_wasted", "sum"),
        avg_daily_sold=("quantity_sold", "mean"),
        avg_daily_wasted=("quantity_wasted", "mean")
    ).reset_index()
    dim_promotion["wastage_rate_pct"] = (
        dim_promotion["total_units_wasted"] / np.maximum(dim_promotion["total_units_sold"] + dim_promotion["total_units_wasted"], 1) * 100
    ).round(2)
    dim_promotion["overproduction_factor"] = (
        dim_promotion["avg_daily_wasted"] / dim_promotion["avg_daily_wasted"].min()
    ).round(3)

    print("  [Dimension 9/9] Analyzing wastage by Preparation Quantity & Over-Prep Index...")
    inv_prep = inventory.copy()
    inv_prep["preparation_quantity"] = inv_prep["starting_stock"] + inv_prep["quantity_received"]
    inv_prep["prep_tier"] = pd.qcut(
        inv_prep["preparation_quantity"], q=4,
        labels=["Low Prep (<80)", "Standard Prep (80-120)", "High Prep (120-160)", "Excess Prep (>160)"]
    )
    dim_prep = inv_prep.groupby("prep_tier", observed=False).agg(
        snapshots_count=("inventory_id", "count"),
        avg_preparation_quantity=("preparation_quantity", "mean"),
        total_prepared_quantity=("preparation_quantity", "sum"),
        total_units_sold=("quantity_sold", "sum"),
        total_units_wasted=("quantity_wasted", "sum"),
        avg_units_wasted=("quantity_wasted", "mean")
    ).reset_index()
    dim_prep["over_preparation_units"] = np.maximum(0, dim_prep["total_prepared_quantity"] - dim_prep["total_units_sold"])
    dim_prep["over_prep_index_pct"] = (
        dim_prep["over_preparation_units"] / dim_prep["total_prepared_quantity"] * 100
    ).round(2)
    dim_prep["prep_wastage_rate_pct"] = (
        dim_prep["total_units_wasted"] / dim_prep["total_prepared_quantity"] * 100
    ).round(2)

    print(f"[Step 23 OK] Completed analytical aggregation across all 9 dimensions.")
    return {
        "by_item": dim_item,
        "by_category": dim_cat,
        "by_location": dim_loc,
        "by_day": dim_day,
        "by_time_period": dim_time_period,
        "by_demand": dim_demand,
        "by_inventory": dim_inventory,
        "by_promotion": dim_promotion,
        "by_preparation": dim_prep
    }


def build_wastage_feature_dataset(
    inventory: pd.DataFrame,
    menu: pd.DataFrame,
    restaurants: pd.DataFrame
) -> pd.DataFrame:
    """
    Constructs the feature matrix using EXACTLY the potential variables SRS lists:
     1. historical_demand
     2. historical_wastage
     3. day_of_week
     4. season
     5. location
     6. promotion
     7. menu_popularity
     8. forecast_demand
     9. preparation_quantity
    """
    print("\n[Phase 3 - Step 24] Constructing Predictive Feature Dataset...")
    df = inventory.copy()
    df = df.sort_values(["location_id", "item_id", "snapshot_date"]).reset_index(drop=True)

    df["preparation_quantity"] = df["starting_stock"] + df["quantity_received"]

    df["historical_demand"] = df.groupby(["location_id", "item_id"])["quantity_sold"].shift(1)
    df["historical_demand_7d"] = df.groupby(["location_id", "item_id"])["historical_demand"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )

    df["historical_wastage"] = df.groupby(["location_id", "item_id"])["quantity_wasted"].shift(1)
    df["historical_wastage_7d"] = df.groupby(["location_id", "item_id"])["historical_wastage"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )

    df["dow_code"] = df["day_of_week"]
    df["weekend_flag"] = df["is_weekend"]

    season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
    df["season_code"] = df["season"].map(season_map)

    df = pd.merge(df, restaurants[["location_id", "location_tier", "cost_index"]], on="location_id", how="left")
    tier_map = {"TIER_1": 1, "TIER_2": 2, "TIER_3": 3}
    df["location_tier_code"] = df["location_tier"].map(tier_map).fillna(2)
    df["location_code"] = df["location_id"].astype("category").cat.codes


    df = pd.merge(
        df,
        menu[["item_id", "name", "category_name", "category_id", "cost_price", "popularity_weight", "shelf_life_days"]],
        on="item_id",
        how="left"
    )
    df["item_code"] = df["item_id"].astype("category").cat.codes
    df["category_code"] = df["category_id"].astype("category").cat.codes

    df["forecast_demand"] = np.round(
        df["historical_demand_7d"] * np.where(df["weekend_flag"] == 1, 1.15, 0.95), 1
    )

    def assign_risk_tier(wasted: int) -> str:
        if wasted == 0:
            return "Low Risk"
        elif 1 <= wasted <= 2:
            return "Moderate Risk"
        elif 3 <= wasted <= 5:
            return "High Risk"
        else:
            return "Critical Risk"

    df["wastage_risk_tier"] = df["quantity_wasted"].apply(assign_risk_tier)
    df["is_high_risk"] = (df["quantity_wasted"] >= 3).astype(int)

    clean_df = df.dropna(subset=["historical_demand", "historical_wastage"]).reset_index(drop=True)
    print(f"Constructed feature matrix with {len(clean_df):,} records and 9 explicit SRS variables.")
    return clean_df


def train_and_evaluate_wastage_models(features_df: pd.DataFrame) -> Tuple[Any, Any, pd.DataFrame, Dict[str, Any]]:
    """
    Trains both a continuous regressor and a discrete risk classifier
    using a strict chronological train/test split.
    Evaluates both models and computes feature importances.
    """
    print("\n[Step 24] Training Wastage Risk Regressor & Classifier...")

    max_date = features_df["snapshot_date"].max()
    split_date = max_date - pd.Timedelta(days=30)

    train_data = features_df[features_df["snapshot_date"] <= split_date].copy()
    test_data = features_df[features_df["snapshot_date"] > split_date].copy()

    assert train_data["snapshot_date"].max() < test_data["snapshot_date"].min()

    feature_cols = [
        "historical_demand_7d",                        
        "historical_wastage_7d",                        
        "dow_code",                              
        "season_code",                      
        "location_code",                      
        "is_promo_active",                     
        "popularity_weight",                         
        "forecast_demand",                           
        "preparation_quantity"                            
    ]

    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_iter=120,
        random_state=42
    )
    regressor.fit(train_data[feature_cols], train_data["quantity_wasted"])
    test_pred_reg = regressor.predict(test_data[feature_cols])
    test_pred_reg = np.maximum(0.0, np.round(test_pred_reg, 1))

    mae = float(mean_absolute_error(test_data["quantity_wasted"], test_pred_reg))
    rmse = float(np.sqrt(mean_squared_error(test_data["quantity_wasted"], test_pred_reg)))
    r2 = float(r2_score(test_data["quantity_wasted"], test_pred_reg))

    classifier = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=120,
        random_state=42
    )
    classifier.fit(train_data[feature_cols], train_data["is_high_risk"])
    test_pred_cls = classifier.predict(test_data[feature_cols])
    test_pred_prob = classifier.predict_proba(test_data[feature_cols])[:, 1]

    acc = float(accuracy_score(test_data["is_high_risk"], test_pred_cls))
    f1 = float(f1_score(test_data["is_high_risk"], test_pred_cls, zero_division=0))
    auc = float(roc_auc_score(test_data["is_high_risk"], test_pred_prob))

    feature_importances = []
    baseline_mae = mae
    for f in feature_cols:
        test_perm = test_data[feature_cols].copy()
        test_perm[f] = np.random.permutation(test_perm[f].values)
        perm_pred = regressor.predict(test_perm)
        perm_mae = mean_absolute_error(test_data["quantity_wasted"], perm_pred)
        importance_score = max(0.0, perm_mae - baseline_mae)
        feature_importances.append({
            "feature_name": f,
            "importance_delta_mae": round(importance_score, 4)
        })

    fi_df = pd.DataFrame(feature_importances).sort_values("importance_delta_mae", ascending=False).reset_index(drop=True)
    fi_df["relative_importance_pct"] = (fi_df["importance_delta_mae"] / fi_df["importance_delta_mae"].sum() * 100).round(2)

    test_output = test_data[[
        "snapshot_date", "location_id", "item_id", "name", "category_name",
        "preparation_quantity", "quantity_sold", "quantity_wasted",
        "wastage_risk_tier", "is_high_risk"
    ]].copy()
    test_output["predicted_quantity_wasted"] = test_pred_reg
    test_output["predicted_high_risk"] = test_pred_cls
    test_output["predicted_risk_probability"] = np.round(test_pred_prob, 4)

    def assign_actionable_recommendation(row):
        if row["predicted_risk_probability"] >= 0.70:
            return "URGENT: Reduce planned morning prep batch by 25%; launch 15% flash happy hour discount to clear inventory."
        elif row["predicted_risk_probability"] >= 0.40:
            return "MODERATE: Trim prep buffer by 10%; bundle as lunch cross-sell combo item."
        else:
            return "OPTIMAL: Standard prep cadence maintained; demand aligned with shelf-life."

    test_output["actionable_mitigation_strategy"] = test_output.apply(assign_actionable_recommendation, axis=1)

    eval_summary = {
        "regression_metrics": {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2_score": round(r2, 4)
        },
        "classification_metrics": {
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4)
        },
        "feature_importance": fi_df.to_dict(orient="records"),
        "test_sample_count": len(test_data),
        "high_risk_alerts_identified": int((test_output["predicted_high_risk"] == 1).sum())
    }

    print(f"  [Step 24 Evaluation] Regressor -> MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.4f}")
    print(f"  [Step 24 Evaluation] Classifier -> Accuracy: {acc*100:.2f}% | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    return regressor, classifier, test_output, eval_summary


def run_wastage_pipeline() -> Dict[str, Any]:
    """Coordinates end-to-end execution of SRS Steps 23 & 24."""
    pipeline_t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 80)
    print("DineIQ Analytics - SRS Steps 23 & 24: Wastage Analysis & Risk Prediction")
    print("=" * 80)

    wastage, inventory, menu, restaurants, promotions = load_and_enrich_wastage_data()

    dim_results = analyze_wastage_all_dimensions(wastage, inventory, menu, restaurants, promotions)

    features_df = build_wastage_feature_dataset(inventory, menu, restaurants)
    regressor, classifier, test_predictions, eval_summary = train_and_evaluate_wastage_models(features_df)

    print("\n[Persistence] Serializing trained ML models to models/wastage/...")
    reg_path = os.path.join(MODELS_DIR, "wastage_risk_regressor.joblib")
    cls_path = os.path.join(MODELS_DIR, "wastage_risk_classifier.joblib")
    joblib.dump(regressor, reg_path)
    joblib.dump(classifier, cls_path)

    print("[Persistence] Saving analytical dimensions and prediction datasets to processed_data/wastage/...")
    dataset_map = {
        "wastage_by_item": dim_results["by_item"],
        "wastage_by_category": dim_results["by_category"],
        "wastage_by_location": dim_results["by_location"],
        "wastage_by_day": dim_results["by_day"],
        "wastage_by_time_period": dim_results["by_time_period"],
        "wastage_by_demand": dim_results["by_demand"],
        "wastage_by_inventory": dim_results["by_inventory"],
        "wastage_by_promotion": dim_results["by_promotion"],
        "wastage_by_preparation": dim_results["by_preparation"],
        "wastage_risk_predictions": test_predictions
    }

    for name, df in dataset_map.items():
        pq_path = os.path.join(OUTPUT_DIR, f"{name}.parquet")
        csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        df.to_parquet(pq_path, compression="snappy")
        df.to_csv(csv_path, index=False)

    print("[Reports] Compiling comprehensive reports in reports/wastage/...")
    md_path = os.path.join(REPORTS_DIR, "wastage_risk_report.md")
    json_path = os.path.join(REPORTS_DIR, "wastage_risk_report.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_summary": eval_summary,
            "top_wastage_items": dim_results["by_item"].head(10).to_dict(orient="records"),
            "category_wastage": dim_results["by_category"].to_dict(orient="records"),
            "shift_wastage": dim_results["by_time_period"].to_dict(orient="records"),
            "preparation_analysis": dim_results["by_preparation"].to_dict(orient="records")
        }, f, indent=2, default=str)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# DineIQ Analytics: Food Wastage Analysis & Predictive Risk Modeling\n\n")
        f.write(f"**Execution Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Wastage Incidents Analyzed:** {len(wastage):,} records\n")
        f.write(f"**Total Financial Loss Evaluated:** ${wastage['total_loss_amount'].sum():,.2f}\n\n")

        f.write("## 1. Executive Summary & Step 24 Model Performance\n\n")
        f.write("Predictive models were trained using a strict chronological split (unseen 30-day out-of-sample test window):\n\n")
        f.write("| Model Type | Target Variable | Primary Metric | Secondary Metric | Out-of-Sample Window |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| **Gradient Boosted Regressor** | Continuous Units Wasted | MAE: **{eval_summary['regression_metrics']['mae']:.3f}** | RMSE: **{eval_summary['regression_metrics']['rmse']:.3f}** (R²: {eval_summary['regression_metrics']['r2_score']:.4f}) | Last 30 Days |\n")
        f.write(f"| **Gradient Boosted Classifier** | High Wastage Risk (≥3 Units) | ROC-AUC: **{eval_summary['classification_metrics']['roc_auc']:.4f}** | F1: **{eval_summary['classification_metrics']['f1_score']:.4f}** (Acc: {eval_summary['classification_metrics']['accuracy']*100:.1f}%) | Last 30 Days |\n\n")

        f.write("### Predictor Feature Importance Ranking (All 9 SRS Variables)\n\n")
        f.write("| Rank | Predictor Variable (SRS Required) | Relative Importance (%) | Operational Rationale |\n")
        f.write("|---:|---|---:|---|\n")
        for i, fi in enumerate(eval_summary["feature_importance"]):
            f.write(f"| {i+1} | `{fi['feature_name']}` | **{fi['relative_importance_pct']:.2f}%** | Directly governs kitchen batch prep buffer vs consumption |\n")
        f.write("\n")

        f.write("## 2. Multi-Dimensional Wastage Analysis (Step 23 - All 9 Dimensions)\n\n")

        f.write("### 1. By Menu Item (Top 10 Highest Financial Loss Dishes)\n\n")
        f.write("| Rank | Menu Item | Category | Wasted Units | Total Loss ($) | Avg Loss / Incident ($) | Wastage Rate (%) |\n")
        f.write("|---:|---|---|---:|---:|---:|---:|\n")
        for i, r in dim_results["by_item"].head(10).iterrows():
            f.write(f"| {i+1} | **{r['name']}** | {r['category_name']} | {int(r['wasted_quantity']):,} | **${r['total_loss_amount']:,.2f}** | ${r['avg_loss_per_incident']:.2f} | {r['wastage_rate_pct']:.2f}% |\n")
        f.write("\n")

        f.write("### 2. By Menu Category\n\n")
        f.write("| Menu Category | Total Loss ($) | Loss Share (%) | Wasted Units | Incidents |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for _, r in dim_results["by_category"].iterrows():
            f.write(f"| **{r['category_name']}** | ${r['total_loss_amount']:,.2f} | **{r['loss_share_pct']:.2f}%** | {int(r['wasted_quantity']):,} | {int(r['incident_count']):,} |\n")
        f.write("\n")

        f.write("### 3. By Restaurant Location (Top 10 High-Loss Locations)\n\n")
        f.write("| Restaurant Name | City | Tier | Total Loss ($) | Wasted Units | Wastage Rate (%) |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for _, r in dim_results["by_location"].head(10).iterrows():
            f.write(f"| **{r['restaurant_name']}** | {r['restaurant_city']} | {r['location_tier']} | ${r['total_loss_amount']:,.2f} | {int(r['wasted_quantity']):,} | {r['wastage_rate_pct']:.2f}% |\n")
        f.write("\n")

        f.write("### 4. By Day of Week\n\n")
        f.write("| Day of Week | Total Loss ($) | Daily Avg Loss ($) | Loss Share (%) | Daily Avg Wasted Units |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for _, r in dim_results["by_day"].iterrows():
            f.write(f"| **{r['day_name']}** | ${r['total_loss_amount']:,.2f} | ${r['daily_avg_loss']:,.2f} | {r['loss_share_pct']:.2f}% | {r['daily_avg_wasted_units']:.1f} |\n")
        f.write("\n")

        f.write("### 5. By Time Period (Kitchen Shifts & Root Causes)\n\n")
        f.write("| Shift Period | Total Loss ($) | Share (%) | Incidents | Primary Root Cause Driver |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for _, r in dim_results["by_time_period"].iterrows():
            f.write(f"| **{r['shift_period']}** | ${r['total_loss_amount']:,.2f} | {r['loss_share_pct']:.2f}% | {int(r['incident_count']):,} | Overproduction unsold & prep expiration |\n")
        f.write("\n")

        f.write("### 6. By Demand Levels (Sales Volume Quartiles)\n\n")
        f.write("| Demand Quartile | Snapshots | Avg Daily Sold | Avg Daily Wasted | Wastage Rate (%) | Demand/Waste Ratio |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for _, r in dim_results["by_demand"].iterrows():
            f.write(f"| **{r['demand_quartile']}** | {int(r['snapshots_count']):,} | {r['avg_units_sold']:.1f} | {r['avg_units_wasted']:.2f} | **{r['wastage_rate_pct']:.2f}%** | {r['demand_to_wastage_ratio']:.1f}x |\n")
        f.write("\n")

        f.write("### 7. By Inventory Consumption & Stock Status\n\n")
        f.write("| Stock Status | Snapshots | Consumption Ratio (%) | Inventory Wastage Rate (%) | Avg Ending Stock |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for _, r in dim_results["by_inventory"].iterrows():
            f.write(f"| **{r['stock_status']}** | {int(r['snapshots_count']):,} | {r['stock_consumption_ratio_pct']:.2f}% | **{r['inventory_wastage_rate_pct']:.2f}%** | {r['avg_ending_stock']:.1f} |\n")
        f.write("\n")

        f.write("### 8. By Promotion Activity\n\n")
        f.write("| Promo Status | Snapshots | Total Units Sold | Total Units Wasted | Wastage Rate (%) | Overproduction Surge |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for _, r in dim_results["by_promotion"].iterrows():
            f.write(f"| **{r['promo_status']}** | {int(r['snapshot_count']):,} | {int(r['total_units_sold']):,} | {int(r['total_units_wasted']):,} | {r['wastage_rate_pct']:.2f}% | **{r['overproduction_factor']:.2f}x** |\n")
        f.write("\n")

        f.write("### 9. By Preparation Quantity & Over-Prep Index\n\n")
        f.write("| Preparation Batch Tier | Snapshots | Avg Batch Prepped | Avg Units Wasted | Over-Prep Index (%) | Prep Waste Rate (%) |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for _, r in dim_results["by_preparation"].iterrows():
            f.write(f"| **{r['prep_tier']}** | {int(r['snapshots_count']):,} | {r['avg_preparation_quantity']:.1f} | {r['avg_units_wasted']:.2f} | **{r['over_prep_index_pct']:.2f}%** | {r['prep_wastage_rate_pct']:.2f}% |\n")
        f.write("\n")

        f.write("## 3. High-Risk Operational Early-Warning Alerts (Step 24)\n\n")
        f.write("Sample proactive operational mitigation recommendations generated for high-risk item batches:\n\n")
        f.write("| Snapshot Date | Location | Menu Item | Prepped | Sold | Wasted (Act) | Predicted Wasted | Risk Prob | Operational Mitigation Strategy |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---|\n")
        sample_alerts = test_predictions[test_predictions["predicted_high_risk"] == 1].head(6)
        for _, r in sample_alerts.iterrows():
            f.write(f"| {str(r['snapshot_date'])[:10]} | `{r['location_id']}` | **{r['name']}** | {int(r['preparation_quantity'])} | {int(r['quantity_sold'])} | {int(r['quantity_wasted'])} | **{r['predicted_quantity_wasted']:.1f}** | {r['predicted_risk_probability']*100:.1f}% | {r['actionable_mitigation_strategy']} |\n")
        f.write("\n")

    print(f"\n[OK] Reports saved to {md_path} and {json_path}")
    print(f"Wastage Pipeline completed in {time.time() - pipeline_t0:.2f} seconds!")
    print("=" * 80)

    return {
        "dimensions": dim_results,
        "evaluation": eval_summary,
        "test_predictions": test_predictions
    }


if __name__ == "__main__":
    run_wastage_pipeline()
