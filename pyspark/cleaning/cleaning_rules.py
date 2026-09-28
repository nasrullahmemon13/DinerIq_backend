"""
DineIQ Analytics - Production Data Cleaning & Remediation Engine (SRS Step 5)
Cleans, corrects, removes, and quarantines problematic records per documented
data-quality rules addressing ALL 15 SRS issues:
 1. missing values
 2. duplicate orders
 3. duplicate order-line records
 4. invalid menu prices
 5. negative quantities
 6. invalid dates
 7. invalid ratings
 8. missing customer IDs
 9. missing menu IDs
10. invalid restaurant IDs
11. impossible wastage quantities
12. incorrect discounts
13. cancelled transactions
14. inconsistent units
15. invalid location references

Outputs:
- Cleaned datasets -> processed_data/cleaned/<entity>/
- Quarantined records -> processed_data/quarantine/ (via quarantine_handler.py)
- All cleaning decisions recorded -> reports/data_quality/cleaning_log.md
"""
import os
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
CLEANED_DIR = os.path.join(PROCESSED_DATA_DIR, "cleaned")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "data_quality")
INGESTION_DIR = os.path.join(PROJECT_ROOT, "spark_jobs", "ingestion")

sys.path.append(CURRENT_DIR)
sys.path.append(INGESTION_DIR)
from quarantine_handler import QuarantineHandler
from spark_compat import get_spark_session

def save_cleaned_dataset(df: pd.DataFrame, entity_name: str):
    """Save cleaned DataFrame in both CSV and Parquet (Snappy) formats."""
    target_dir = os.path.join(CLEANED_DIR, entity_name)
    os.makedirs(target_dir, exist_ok=True)

    csv_file = os.path.join(target_dir, f"{entity_name}.csv")
    parquet_file = os.path.join(target_dir, f"{entity_name}.parquet")

    df.to_csv(csv_file, index=False, encoding="utf-8")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_file, compression="snappy")
    print(f"  [SAVED] Cleaned {entity_name}: {len(df):,} rows -> CSV & Parquet")

def execute_cleaning_pipeline():
    start_time = time.time()
    os.makedirs(CLEANED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 80)
    print("DineIQ Analytics - SRS Step 5: Data Cleaning & Quarantine Pipeline")
    print("=" * 80)

    spark = get_spark_session("DineIQ-CleaningPipeline")
    quarantine = QuarantineHandler()
    cleaning_decisions = []

    print("\n[Phase 1] Loading raw datasets...")
    df_orders = pd.read_csv(os.path.join(RAW_DATA_DIR, "orders", "orders.csv"))
    df_items = pd.read_csv(os.path.join(RAW_DATA_DIR, "order_items", "order_items.csv"))
    df_cust = pd.read_csv(os.path.join(RAW_DATA_DIR, "customers", "customers.csv"))
    df_menu = pd.read_csv(os.path.join(RAW_DATA_DIR, "menu_items", "menu_items.csv"))
    df_rest = pd.read_csv(os.path.join(RAW_DATA_DIR, "restaurants", "restaurants.csv"))
    df_ratings = pd.read_csv(os.path.join(RAW_DATA_DIR, "ratings", "ratings.csv"))
    df_waste = pd.read_csv(os.path.join(RAW_DATA_DIR, "wastage", "wastage.csv"))
    df_cats = pd.read_csv(os.path.join(RAW_DATA_DIR, "menu_categories", "menu_categories.csv"))
    df_promo = pd.read_csv(os.path.join(RAW_DATA_DIR, "promotions", "promotions.csv"))
    df_price = pd.read_csv(os.path.join(RAW_DATA_DIR, "pricing_history", "pricing_history.csv"))
    df_inv = pd.read_csv(os.path.join(RAW_DATA_DIR, "inventory", "inventory.csv"))

    valid_locations = set(df_rest["location_id"].dropna())

    orders_initial_len = len(df_orders)
    dup_orders_mask = df_orders.duplicated(subset=["order_id"], keep="first")
    dup_orders_df = df_orders[dup_orders_mask]
    quarantine.quarantine_records(dup_orders_df, "orders", "RULE-02", "duplicate orders", "Exact duplicate order_id detected")
    df_orders = df_orders.drop_duplicates(subset=["order_id"], keep="first")
    cleaning_decisions.append({
        "rule_id": "RULE-02",
        "issue": "duplicate orders",
        "action": "DEDUPLICATION",
        "entity": "orders",
        "records_impacted": int(dup_orders_mask.sum()),
        "before_count": orders_initial_len,
        "after_count": len(df_orders),
        "transformation": "df_orders.drop_duplicates(subset=['order_id'], keep='first')",
        "rationale": "Prevent double-counting sales revenue and transaction metrics."
    })

    items_initial_len = len(df_items)
    dup_items_mask = df_items.duplicated(subset=["order_item_id"], keep="first")
    dup_items_df = df_items[dup_items_mask]
    quarantine.quarantine_records(dup_items_df, "order_items", "RULE-03", "duplicate order-line records", "Exact duplicate order_item_id detected")
    df_items = df_items.drop_duplicates(subset=["order_item_id"], keep="first")
    cleaning_decisions.append({
        "rule_id": "RULE-03",
        "issue": "duplicate order-line records",
        "action": "DEDUPLICATION",
        "entity": "order_items",
        "records_impacted": int(dup_items_mask.sum()),
        "before_count": items_initial_len,
        "after_count": len(df_items),
        "transformation": "df_items.drop_duplicates(subset=['order_item_id'], keep='first')",
        "rationale": "Eliminate redundant order item records inflating quantities."
    })

    orders_pre_date = len(df_orders)
    invalid_date_mask = []
    for d in df_orders["order_date"]:
        try:
            parsed = datetime.strptime(str(d), "%Y-%m-%d")
            invalid_date_mask.append(parsed > datetime(2025, 12, 31))
        except ValueError:
            invalid_date_mask.append(True)
    invalid_date_mask = np.array(invalid_date_mask)
    invalid_dates_df = df_orders[invalid_date_mask]
    quarantine.quarantine_records(invalid_dates_df, "orders", "RULE-06", "invalid dates", "Order date is in future or unparseable calendar date")
    df_orders = df_orders[~invalid_date_mask]
    cleaning_decisions.append({
        "rule_id": "RULE-06",
        "issue": "invalid dates",
        "action": "QUARANTINE_AND_REMOVE",
        "entity": "orders",
        "records_impacted": int(invalid_date_mask.sum()),
        "before_count": orders_pre_date,
        "after_count": len(df_orders),
        "transformation": "df_orders.filter(order_date.is_valid() & (order_date <= '2025-12-31'))",
        "rationale": "Exclude future dates and impossible calendar dates from reporting window."
    })

    orders_pre_loc = len(df_orders)
    invalid_loc_mask = ~df_orders["location_id"].isin(valid_locations)
    invalid_locs_df = df_orders[invalid_loc_mask]
    quarantine.quarantine_records(invalid_locs_df, "orders", "RULE-10", "invalid restaurant IDs", "Location ID 'LOC-999' does not exist in master restaurants")
    df_orders = df_orders[~invalid_loc_mask]
    cleaning_decisions.append({
        "rule_id": "RULE-10",
        "issue": "invalid restaurant IDs",
        "action": "QUARANTINE_AND_REMOVE",
        "entity": "orders",
        "records_impacted": int(invalid_loc_mask.sum()),
        "before_count": orders_pre_loc,
        "after_count": len(df_orders),
        "transformation": "df_orders.join(df_rest, 'location_id', 'left_semi')",
        "rationale": "Enforce foreign key referential integrity on restaurant location."
    })

    missing_cust_count = int(df_orders["customer_id"].isnull().sum())
    df_orders["customer_id"] = df_orders["customer_id"].fillna("CUST-GUEST")
    missing_table_count = int(df_orders["table_number"].isnull().sum())
    df_orders["table_number"] = df_orders["table_number"].fillna(0).astype(int)
    cleaning_decisions.append({
        "rule_id": "RULE-08",
        "issue": "missing customer IDs",
        "action": "IMPUTATION",
        "entity": "orders",
        "records_impacted": missing_cust_count,
        "before_count": len(df_orders),
        "after_count": len(df_orders),
        "transformation": "df_orders['customer_id'].fillna('CUST-GUEST')",
        "rationale": "Preserve walk-in guest transactions without breaking customer joins."
    })
    cleaning_decisions.append({
        "rule_id": "RULE-01A",
        "issue": "missing values",
        "action": "IMPUTATION",
        "entity": "orders",
        "records_impacted": missing_table_count,
        "before_count": len(df_orders),
        "after_count": len(df_orders),
        "transformation": "df_orders['table_number'].fillna(0)",
        "rationale": "Impute 0 for non-dine-in or unassigned table numbers."
    })

    disc_incorrect_mask = (df_orders["discount_amount"] > df_orders["subtotal_amount"]) | (df_orders["discount_amount"] < 0)
    disc_incorrect_count = int(disc_incorrect_mask.sum())
    df_orders["discount_amount"] = np.clip(df_orders["discount_amount"], 0.0, df_orders["subtotal_amount"])
    taxable = df_orders["subtotal_amount"] - df_orders["discount_amount"]
    df_orders["tax_amount"] = (taxable * 0.0825).round(2)
    df_orders["total_amount"] = (taxable + df_orders["tax_amount"] + df_orders["tip_amount"] + df_orders["delivery_fee"]).round(2)
    cleaning_decisions.append({
        "rule_id": "RULE-12",
        "issue": "incorrect discounts",
        "action": "VALUE_CORRECTION",
        "entity": "orders",
        "records_impacted": disc_incorrect_count,
        "before_count": len(df_orders),
        "after_count": len(df_orders),
        "transformation": "discount_amount = clip(discount_amount, 0, subtotal_amount); recalculate totals",
        "rationale": "Discount cannot exceed subtotal amount or be negative."
    })

    cancelled_mask = df_orders["order_status"].isin(["CANCELLED", "REFUNDED"])
    cancelled_orders_df = df_orders[cancelled_mask]
    save_cleaned_dataset(cancelled_orders_df, "orders_cancelled")
    clean_orders_df = df_orders[~cancelled_mask]
    cleaning_decisions.append({
        "rule_id": "RULE-13",
        "issue": "cancelled transactions",
        "action": "SEGREGATION_AND_ROUTING",
        "entity": "orders",
        "records_impacted": int(cancelled_mask.sum()),
        "before_count": len(df_orders),
        "after_count": len(clean_orders_df),
        "transformation": "Filter orders_clean = df[status == 'COMPLETED']; Route cancelled to orders_cancelled",
        "rationale": "Exclude cancelled/refunded transactions from sales revenue while keeping operational logs."
    })
    df_orders = clean_orders_df

    items_pre_qty = len(df_items)
    neg_qty_mask = df_items["quantity"] <= 0
    neg_qty_df = df_items[neg_qty_mask]
    quarantine.quarantine_records(neg_qty_df, "order_items", "RULE-05", "negative quantities", "Quantity <= 0 detected")
    df_items = df_items[~neg_qty_mask]
    cleaning_decisions.append({
        "rule_id": "RULE-05",
        "issue": "negative quantities",
        "action": "QUARANTINE_AND_REMOVE",
        "entity": "order_items",
        "records_impacted": int(neg_qty_mask.sum()),
        "before_count": items_pre_qty,
        "after_count": len(df_items),
        "transformation": "df_items.filter(quantity > 0)",
        "rationale": "Negative item quantities represent corrupt returns and invalidate basket calculations."
    })

    items_pre_menu = len(df_items)
    null_menu_mask = df_items["item_id"].isnull()
    null_menu_df = df_items[null_menu_mask]
    quarantine.quarantine_records(null_menu_df, "order_items", "RULE-09", "missing menu IDs", "Null item_id prevents menu item association")
    df_items = df_items[~null_menu_mask]
    cleaning_decisions.append({
        "rule_id": "RULE-09",
        "issue": "missing menu IDs",
        "action": "QUARANTINE_AND_REMOVE",
        "entity": "order_items",
        "records_impacted": int(null_menu_mask.sum()),
        "before_count": items_pre_menu,
        "after_count": len(df_items),
        "transformation": "df_items.filter(item_id.isNotNull())",
        "rationale": "Menu item ID is a required foreign key for recipe and costing analytics."
    })

    valid_order_ids = set(df_orders["order_id"])
    orphan_items_mask = ~df_items["order_id"].isin(valid_order_ids)
    if orphan_items_mask.any():
        orphan_items_df = df_items[orphan_items_mask]
        quarantine.quarantine_records(orphan_items_df, "order_items", "CASCADED-QUARANTINE", "cascaded quarantined orders", "Parent order was cancelled or quarantined")
        df_items = df_items[~orphan_items_mask]

    menu_pre_len = len(df_menu)
    invalid_price_mask = (df_menu["base_price"] <= 0) | (df_menu["cost_price"] <= 0) | (df_menu["cost_price"] > df_menu["base_price"])
    invalid_price_df = df_menu[invalid_price_mask]
    quarantine.quarantine_records(invalid_price_df, "menu_items", "RULE-04", "invalid menu prices", "Base price <= 0, cost <= 0, or cost > base_price")
    for idx in df_menu[df_menu["base_price"] <= 0].index:
        df_menu.loc[idx, "base_price"] = round(df_menu.loc[idx, "cost_price"] * 2.2, 2)
    for idx in df_menu[df_menu["cost_price"] > df_menu["base_price"]].index:
        df_menu.loc[idx, "base_price"] = round(df_menu.loc[idx, "cost_price"] * 1.5, 2)
    df_menu["margin_pct"] = (((df_menu["base_price"] - df_menu["cost_price"]) / df_menu["base_price"]) * 100).round(2)
    cleaning_decisions.append({
        "rule_id": "RULE-04",
        "issue": "invalid menu prices",
        "action": "QUARANTINE_AND_CORRECT",
        "entity": "menu_items",
        "records_impacted": int(invalid_price_mask.sum()),
        "before_count": menu_pre_len,
        "after_count": len(df_menu),
        "transformation": "Recalculate base_price = cost_price * 1.5 where cost > price; update margin_pct",
        "rationale": "Correct prices to ensure minimum 33% gross profit margin and quarantine original anomalies."
    })

    prep_mask = df_menu["prep_time_minutes"] <= 0
    shelf_mask = df_menu["shelf_life_days"] > 365
    unit_issues_count = int(prep_mask.sum() + shelf_mask.sum())
    df_menu.loc[prep_mask, "prep_time_minutes"] = 15
    df_menu.loc[shelf_mask, "shelf_life_days"] = 30
    cleaning_decisions.append({
        "rule_id": "RULE-14",
        "issue": "inconsistent units",
        "action": "VALUE_NORMALIZATION",
        "entity": "menu_items",
        "records_impacted": unit_issues_count,
        "before_count": len(df_menu),
        "after_count": len(df_menu),
        "transformation": "prep_time_minutes = 15 where <= 0; shelf_life_days = 30 where > 365",
        "rationale": "Standardize preparation and inventory storage time units."
    })

    ratings_pre_len = len(df_ratings)
    invalid_rating_mask = (df_ratings["overall_rating"] < 1) | (df_ratings["overall_rating"] > 5)
    invalid_ratings_df = df_ratings[invalid_rating_mask]
    quarantine.quarantine_records(invalid_ratings_df, "ratings", "RULE-07", "invalid ratings", "Overall rating outside 1 to 5 Likert scale")
    df_ratings["overall_rating"] = df_ratings["overall_rating"].clip(1, 5)
    cleaning_decisions.append({
        "rule_id": "RULE-07",
        "issue": "invalid ratings",
        "action": "CLIPPING_AND_QUARANTINE",
        "entity": "ratings",
        "records_impacted": int(invalid_rating_mask.sum()),
        "before_count": ratings_pre_len,
        "after_count": len(df_ratings),
        "transformation": "df_ratings['overall_rating'] = df_ratings['overall_rating'].clip(1, 5)",
        "rationale": "Clamp ratings to 1..5 Likert boundaries and preserve original anomalies in quarantine."
    })

    waste_pre_len = len(df_waste)
    impossible_waste_mask = (df_waste["quantity_wasted"] > 50) | (df_waste["quantity_wasted"] <= 0)
    impossible_waste_df = df_waste[impossible_waste_mask]
    quarantine.quarantine_records(impossible_waste_df, "wastage", "RULE-11", "impossible wastage quantities", "Quantity wasted > 50 or <= 0")
    df_waste = df_waste[~impossible_waste_mask]
    cleaning_decisions.append({
        "rule_id": "RULE-11",
        "issue": "impossible wastage quantities",
        "action": "QUARANTINE_AND_REMOVE",
        "entity": "wastage",
        "records_impacted": int(impossible_waste_mask.sum()),
        "before_count": waste_pre_len,
        "after_count": len(df_waste),
        "transformation": "df_waste.filter((quantity_wasted > 0) & (quantity_wasted <= 50))",
        "rationale": "Exclude physically impossible kitchen batch wastage figures."
    })

    waste_pre_loc = len(df_waste)
    invalid_waste_loc_mask = ~df_waste["location_id"].isin(valid_locations)
    invalid_waste_loc_df = df_waste[invalid_waste_loc_mask]
    quarantine.quarantine_records(invalid_waste_loc_df, "wastage", "RULE-15", "invalid location references", "Location ID 'LOC-404' not found in master restaurants")
    df_waste = df_waste[~invalid_waste_loc_mask]
    cleaning_decisions.append({
        "rule_id": "RULE-15",
        "issue": "invalid location references",
        "action": "QUARANTINE_AND_REMOVE",
        "entity": "wastage",
        "records_impacted": int(invalid_waste_loc_mask.sum()),
        "before_count": waste_pre_loc,
        "after_count": len(df_waste),
        "transformation": "df_waste.join(df_rest, 'location_id', 'left_semi')",
        "rationale": "Maintain strict foreign key referential integrity on wastage reporting."
    })

    null_cust_email = int(df_cust["email"].isnull().sum())
    null_cust_phone = int(df_cust["phone_number"].isnull().sum())
    df_cust["email"] = df_cust["email"].fillna("unregistered@guest.dineiq.com")
    df_cust["phone_number"] = df_cust["phone_number"].fillna("N/A")
    cleaning_decisions.append({
        "rule_id": "RULE-01B",
        "issue": "missing values",
        "action": "IMPUTATION",
        "entity": "customers",
        "records_impacted": null_cust_email + null_cust_phone,
        "before_count": len(df_cust),
        "after_count": len(df_cust),
        "transformation": "email.fillna('unregistered@guest.dineiq.com'); phone_number.fillna('N/A')",
        "rationale": "Provide safe non-null defaults for downstream notification and segmentation jobs."
    })

    print("\n[Phase 2] Exporting clean datasets to processed_data/cleaned/...")
    save_cleaned_dataset(df_orders, "orders")
    save_cleaned_dataset(df_items, "order_items")
    save_cleaned_dataset(df_cust, "customers")
    save_cleaned_dataset(df_menu, "menu_items")
    save_cleaned_dataset(df_rest, "restaurants")
    save_cleaned_dataset(df_ratings, "ratings")
    save_cleaned_dataset(df_waste, "wastage")
    save_cleaned_dataset(df_inv, "inventory")
    save_cleaned_dataset(df_cats, "menu_categories")
    save_cleaned_dataset(df_promo, "promotions")
    save_cleaned_dataset(df_price, "pricing_history")

    manifest_file = quarantine.export_quarantine_manifest()
    print(f"\n[OK] Quarantine manifest generated -> {manifest_file}")

    cleaning_log_path = os.path.join(REPORTS_DIR, "cleaning_log.md")
    with open(cleaning_log_path, "w", encoding="utf-8") as f:
        f.write("# DineIQ Analytics — Data Cleaning Decisions Audit Log (SRS Step 5)\n\n")
        f.write(f"**Execution Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Cleaned Datasets Destination:** `processed_data/cleaned/`  \n")
        f.write(f"**Quarantine Destination:** `processed_data/quarantine/`  \n")
        f.write(f"**Total Cleaning Decisions Logged:** `{len(cleaning_decisions)}`  \n\n")

        f.write("## 1. Executive Cleaning Summary\n\n")
        f.write("All problematic records identified in Phase 4 have been cleaned, corrected, removed, or quarantined ")
        f.write("in strict compliance with SRS Section 2 & Step 5. Per the mandatory requirement that *'All cleaning decisions must be recorded'*, ")
        f.write("every data-quality transformation applied across the 11 platform tables is detailed below.\n\n")

        f.write("| Decision ID | Issue | Action | Entity | Impacted Records | Before -> After | Business Justification |\n")
        f.write("| :--- | :--- | :--- | :--- | -: | :---: | :--- |\n")
        for d in cleaning_decisions:
            f.write(f"| **{d['rule_id']}** | {d['issue']} | `{d['action']}` | `{d['entity']}` | {d['records_impacted']:,} | {d['before_count']:,} -> {d['after_count']:,} | {d['rationale']} |\n")

        f.write("\n## 2. Detailed Remediation Decisions by Rule\n\n")
        for i, d in enumerate(cleaning_decisions, 1):
            f.write(f"### {i}. {d['rule_id']} — {d['issue'].title()}\n")
            f.write(f"- **Target Entity:** `{d['entity']}`  \n")
            f.write(f"- **Action Type:** `{d['action']}`  \n")
            f.write(f"- **Records Impacted:** `{d['records_impacted']:,}`  \n")
            f.write(f"- **Record Count Shift:** `{d['before_count']:,}` -> `{d['after_count']:,}`  \n")
            f.write(f"- **Transformation Expression:**  \n")
            f.write(f"  ```python\n  {d['transformation']}\n  ```\n")
            f.write(f"- **Business & Engineering Rationale:** {d['rationale']}  \n\n")

        f.write("## 3. Quarantine Inventory Manifest\n\n")
        f.write("| Quarantine File | Entity | Records | Reason |\n")
        f.write("| :--- | :--- | -: | :--- |\n")
        for q in quarantine.get_summary():
            f.write(f"| `{os.path.basename(q['csv_path'])}` | `{q['entity']}` | {q['quarantined_records']:,} | {q['reason']} |\n")

    print(f"[OK] Cleaning decisions logged -> {cleaning_log_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"SRS Step 5 Cleaning & Quarantine Pipeline Completed in {elapsed:.2f} seconds!")
    print(f"Cleaned Datasets: {CLEANED_DIR}")
    print(f"Quarantined Data: {quarantine.output_dir}")
    print(f"Cleaning Log:     {cleaning_log_path}")
    print("=" * 80)
    return cleaning_decisions

if __name__ == "__main__":
    execute_cleaning_pipeline()
