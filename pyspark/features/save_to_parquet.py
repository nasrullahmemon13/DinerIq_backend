"""
DineIQ Analytics - Feature Persistence Engine (SRS Step 7)
Coordinates feature computation across:
 1. menu_features.py (item revenue, cost, contribution margin, profit percentage, order frequency,
                      item popularity, repeat-purchase rate, average rating, rating trend,
                      wastage percentage, price-change percentage)
 2. customer_features.py (basket size, discount percentage, promotion dependency, peak-hour frequency,
                         weekend-order ratio, channel preference, location performance)
 3. rfm_features.py (customer recency, customer frequency, customer monetary value, average order value)

Saves all feature marts to parquet_data/features/ in Snappy-compressed Parquet and CSV formats.
Generates comprehensive Feature Engineering Catalog report in reports/feature_engineering/.
"""
import os
import sys
import time
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PARQUET_FEATURES_DIR = os.path.join(PROJECT_ROOT, "parquet_data", "features")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "feature_engineering")
INGESTION_DIR = os.path.join(PROJECT_ROOT, "spark_jobs", "ingestion")

if INGESTION_DIR not in sys.path:
    sys.path.append(INGESTION_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from spark_compat import get_spark_session
from menu_features import compute_menu_features
from customer_features import compute_customer_features
from rfm_features import compute_rfm_features

SRS_FEATURES = [
    "item_revenue",
    "cost",
    "contribution_margin",
    "profit_percentage",
    "order_frequency",
    "item_popularity",
    "repeat_purchase_rate",
    "average_rating",
    "rating_trend",
    "wastage_percentage",
    "price_change_percentage",
    "basket_size",
    "discount_percentage",
    "promotion_dependency",
    "peak_hour_frequency",
    "weekend_order_ratio",
    "channel_preference",
    "location_performance",
    "customer_recency",
    "customer_frequency",
    "customer_monetary_value",
    "average_order_value"
]

def save_dataframe_to_parquet(df: pd.DataFrame, feature_set_name: str) -> dict:
    """Saves a pandas DataFrame as both Snappy-compressed Parquet and CSV."""
    os.makedirs(PARQUET_FEATURES_DIR, exist_ok=True)
    parquet_path = os.path.join(PARQUET_FEATURES_DIR, f"{feature_set_name}.parquet")
    csv_path = os.path.join(PARQUET_FEATURES_DIR, f"{feature_set_name}.csv")

    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path, compression="snappy")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    parquet_size_kb = round(os.path.getsize(parquet_path) / 1024, 2)
    csv_size_kb = round(os.path.getsize(csv_path) / 1024, 2)

    print(f"  [SAVED] {feature_set_name}: {len(df):,} rows -> {parquet_size_kb} KB (Parquet), {csv_size_kb} KB (CSV)")
    return {
        "feature_set": feature_set_name,
        "rows": len(df),
        "columns": list(df.columns),
        "parquet_path": parquet_path,
        "parquet_size_kb": parquet_size_kb,
        "csv_path": csv_path,
        "csv_size_kb": csv_size_kb
    }

def run_feature_pipeline():
    start_time = time.time()
    os.makedirs(PARQUET_FEATURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 80)
    print("DineIQ Analytics - SRS Step 7: Feature Engineering Pipeline")
    print("=" * 80)

    spark = get_spark_session("DineIQ-FeatureEngineering")

    print("\n>>> Phase 1: Computing Menu Item Features...")
    t0 = time.time()
    menu_df = compute_menu_features(spark)
    t_menu = time.time() - t0
    meta_menu = save_dataframe_to_parquet(menu_df, "menu_features")

    print("\n>>> Phase 2: Computing Customer Behavioral Features...")
    t0 = time.time()
    cust_df = compute_customer_features(spark)
    t_cust = time.time() - t0
    meta_cust = save_dataframe_to_parquet(cust_df, "customer_features")

    print("\n>>> Phase 3: Computing RFM Features...")
    t0 = time.time()
    rfm_df = compute_rfm_features(spark)
    t_rfm = time.time() - t0
    meta_rfm = save_dataframe_to_parquet(rfm_df, "rfm_features")

    print("\n>>> Phase 4: Merging into Consolidated Customer Feature Store...")
    t0 = time.time()
    customer_master_df = pd.merge(
        cust_df,
        rfm_df[["customer_id", "customer_recency", "customer_frequency", "customer_monetary_value", "average_order_value", "r_score", "f_score", "m_score", "rfm_segment"]],
        on="customer_id",
        how="inner"
    )
    t_master = time.time() - t0
    meta_master = save_dataframe_to_parquet(customer_master_df, "customer_master_features")

    print("\n" + "=" * 80)
    print("VERIFYING ALL SRS FEATURES EXISTENCE & STATISTICAL INTEGRITY")
    print("=" * 80)

    feature_catalog = []
    all_engineered_cols = set(menu_df.columns).union(set(customer_master_df.columns))

    for feat in SRS_FEATURES:
        exists = feat in all_engineered_cols
        source = "menu_features" if feat in menu_df.columns else "customer_master_features"
        series = menu_df[feat] if feat in menu_df.columns else customer_master_df[feat]

        null_count = int(series.isnull().sum())
        dtype_str = str(series.dtype)
        if pd.api.types.is_numeric_dtype(series):
            min_val = round(float(series.min()), 2)
            max_val = round(float(series.max()), 2)
            mean_val = round(float(series.mean()), 2)
        else:
            min_val, max_val, mean_val = "N/A", "N/A", f"{series.nunique()} unique"

        feature_catalog.append({
            "feature_name": feat,
            "domain": "Menu Item" if source == "menu_features" else "Customer",
            "source_dataset": source,
            "data_type": dtype_str,
            "null_count": null_count,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "verified": exists and (null_count == 0)
        })
        status_tag = "[PASS]" if exists and null_count == 0 else "[FAIL]"
        print(f"  {status_tag} {feat:<26} | {source:<24} | Min: {min_val:<8} | Max: {max_val:<8} | Mean: {mean_val}")

    catalog_json_path = os.path.join(REPORTS_DIR, "feature_catalog.json")
    with open(catalog_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generation_time_seconds": round(time.time() - start_time, 2),
            "features_verified_count": len(feature_catalog),
            "feature_details": feature_catalog,
            "storage_manifest": [meta_menu, meta_cust, meta_rfm, meta_master]
        }, f, indent=2)

    catalog_md_path = os.path.join(REPORTS_DIR, "feature_catalog.md")
    with open(catalog_md_path, "w", encoding="utf-8") as f:
        f.write("# DineIQ Analytics - Step 7: Feature Engineering Catalog\n\n")
        f.write(f"**Execution Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Engine:** PySpark 4.2.0 & PyArrow Parquet\n")
        f.write(f"**Total Features Engineered:** {len(feature_catalog)} exact SRS features\n\n")
        f.write("## 1. Feature Definition & Distribution Summary\n\n")
        f.write("| Feature Name | Domain | Storage Mart | Data Type | Min | Max | Mean / Top | Status |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for fc in feature_catalog:
            status = "Verified" if fc["verified"] else "Issue"
            f.write(f"| `{fc['feature_name']}` | {fc['domain']} | `{fc['source_dataset']}` | `{fc['data_type']}` | {fc['min']} | {fc['max']} | {fc['mean']} | {status} |\n")

        f.write("\n## 2. Parquet Storage Outputs\n\n")
        f.write("| Feature Store File | Format | Row Count | File Size (KB) |\n")
        f.write("|---|---|---:|---:|\n")
        for sm in [meta_menu, meta_cust, meta_rfm, meta_master]:
            f.write(f"| `{sm['feature_set']}.parquet` | Snappy Columnar Parquet | {sm['rows']:,} | {sm['parquet_size_kb']} KB |\n")
            f.write(f"| `{sm['feature_set']}.csv` | UTF-8 Delimited CSV | {sm['rows']:,} | {sm['csv_size_kb']} KB |\n")

    print("\n" + "=" * 80)
    print(f"Feature Engineering Pipeline Completed in {time.time() - start_time:.2f} seconds!")
    print(f"Parquet Output: {PARQUET_FEATURES_DIR}")
    print(f"Catalog Reports: {catalog_md_path} and {catalog_json_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_feature_pipeline()
