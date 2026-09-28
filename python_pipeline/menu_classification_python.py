"""
DineIQ Analytics - Independent Python Menu Classification Engine (SRS Step 14)
Processes the underlying restaurant records independently in pure Pandas, NumPy,
Scikit-Learn, and XGBoost without relying on Spark models or Spark outputs.

Employs quality-adjusted economic decision boundaries:
- Incorporates repeat-buyer loyalty and customer rating satisfaction into profitability
- Yields independent 4-category classification:
  * Profit Driver
  * Volume Driver
  * Hidden Opportunity
  * Low Performer
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CLEANED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")
MODELS_PYTHON_DIR = os.path.join(PROJECT_ROOT, "models", "python")

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data_loader import load_cleaned_table

def run_independent_python_menu_classification() -> pd.DataFrame:
    """
    Independently aggregates menu sales, costs, ratings, and wastage in pure Python,
    computes Python composite performance scores, and assigns the 4 SRS categories.
    """
    print("[Python Pipeline] Independently loading operational tables for menu classification...")
    menu_items = load_cleaned_table("menu_items")
    order_items = load_cleaned_table("order_items")
    orders = load_cleaned_table("orders")
    wastage = load_cleaned_table("wastage")
    ratings = load_cleaned_table("ratings")

    print("[Python Pipeline] Aggregating item volume, revenue, and prep costs...")
    merged_items = pd.merge(
        order_items[["order_id", "item_id", "quantity", "unit_price", "item_total"]],
        menu_items[["item_id", "name", "category_id", "base_price", "cost_price"]],
        on="item_id",
        how="inner"
    )

    sales_agg = merged_items.groupby("item_id").agg(
        quantity_sold=("quantity", "sum"),
        revenue=("item_total", "sum"),
        cost=("quantity", lambda q: (q * merged_items.loc[q.index, "cost_price"]).sum())
    ).reset_index()

    sales_agg["contribution_margin"] = sales_agg["revenue"] - sales_agg["cost"]
    sales_agg["profit_percentage"] = (sales_agg["contribution_margin"] / np.maximum(sales_agg["revenue"], 1e-5)) * 100

    orders_valid = orders[orders["customer_id"] != "CUST-GUEST"]
    user_item = pd.merge(
        order_items[["order_id", "item_id"]],
        orders_valid[["order_id", "customer_id"]],
        on="order_id",
        how="inner"
    ).drop_duplicates()

    user_order_counts = user_item.groupby(["item_id", "customer_id"]).size().reset_index(name="user_orders")
    repeat_rates = user_order_counts.groupby("item_id").agg(
        total_customers=("customer_id", "nunique"),
        repeat_customers=("user_orders", lambda s: (s > 1).sum())
    ).reset_index()
    repeat_rates["repeat_purchase_rate"] = repeat_rates["repeat_customers"] / np.maximum(repeat_rates["total_customers"], 1)

    ratings_agg = ratings.groupby("item_id")["overall_rating"].agg(["mean", "count"]).reset_index()
    ratings_agg.columns = ["item_id", "customer_rating", "review_count"]

    waste_agg = wastage.groupby("item_id")["quantity_wasted"].sum().reset_index()
    waste_agg.columns = ["item_id", "total_wasted_units"]

    df_py = pd.merge(menu_items[["item_id", "name", "category_id", "base_price", "cost_price"]], sales_agg, on="item_id", how="left")
    df_py = pd.merge(df_py, repeat_rates[["item_id", "repeat_purchase_rate"]], on="item_id", how="left")
    df_py = pd.merge(df_py, ratings_agg[["item_id", "customer_rating"]], on="item_id", how="left")
    df_py = pd.merge(df_py, waste_agg[["item_id", "total_wasted_units"]], on="item_id", how="left")

    df_py["quantity_sold"] = df_py["quantity_sold"].fillna(0)
    df_py["revenue"] = df_py["revenue"].fillna(0.0)
    df_py["cost"] = df_py["cost"].fillna(0.0)
    df_py["contribution_margin"] = df_py["contribution_margin"].fillna(0.0)
    df_py["profit_percentage"] = df_py["profit_percentage"].fillna(0.0)
    df_py["customer_rating"] = df_py["customer_rating"].fillna(df_py["customer_rating"].median())
    df_py["repeat_purchase_rate"] = df_py["repeat_purchase_rate"].fillna(df_py["repeat_purchase_rate"].median())
    df_py["total_wasted_units"] = df_py["total_wasted_units"].fillna(0)
    df_py["wastage_percentage"] = (df_py["total_wasted_units"] / np.maximum(df_py["quantity_sold"] + df_py["total_wasted_units"], 1)) * 100

    scaler = RobustScaler()
    scaled_feats = scaler.fit_transform(df_py[["quantity_sold", "contribution_margin", "customer_rating", "repeat_purchase_rate"]])

    raw_score = (
        0.35 * scaled_feats[:, 0] + 
        0.40 * scaled_feats[:, 1] + 
        0.15 * scaled_feats[:, 2] + 
        0.10 * scaled_feats[:, 3] - 
        0.10 * (df_py["wastage_percentage"].values / 20.0)
    )
    min_s, max_s = raw_score.min(), raw_score.max()
    df_py["python_composite_score"] = np.round((raw_score - min_s) / (max_s - min_s), 4)

    demand_med = df_py["quantity_sold"].median()
    cm_med = df_py["contribution_margin"].median()
    prof_med = df_py["profit_percentage"].median()
    waste_med = df_py["wastage_percentage"].median()
    rating_med = df_py["customer_rating"].median()

    def classify_python(row):
        if row["contribution_margin"] <= 0 or row["wastage_percentage"] >= 19.5:
            return "Low Performer"

        is_high_vol = row["quantity_sold"] >= demand_med
        is_high_prof = (row["contribution_margin"] >= cm_med * 0.98) and (row["profit_percentage"] >= prof_med * 0.96)
        is_acceptable_waste = row["wastage_percentage"] <= (waste_med * 1.6)

        if is_high_vol and is_high_prof and is_acceptable_waste:
            return "Profit Driver"
        elif is_high_vol and not is_high_prof:
            return "Volume Driver"
        elif not is_high_vol and (is_high_prof or (row["customer_rating"] >= rating_med and row["repeat_purchase_rate"] >= 0.05)):
            return "Hidden Opportunity"
        else:
            return "Low Performer"

    df_py["python_classification"] = df_py.apply(classify_python, axis=1)
    print(f"[Python Pipeline OK] Independently classified all {len(df_py)} items in Python:")
    for cat, cnt in df_py["python_classification"].value_counts().items():
        print(f"  - {cat:<20}: {cnt} items ({cnt/len(df_py)*100:.1f}%)")

    return df_py

if __name__ == "__main__":
    df = run_independent_python_menu_classification()
    print(df[["item_id", "name", "python_composite_score", "python_classification"]].head(10).to_string())
