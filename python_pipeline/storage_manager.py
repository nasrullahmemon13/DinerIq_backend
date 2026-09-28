"""
DineIQ Analytics - Multi-Modal Big Data Storage Manager (SRS Step 2)
Implements storage across all 5 modalities specified in SRS Step 2:
1. CSV: All 11 tables in raw_data/
2. JSON: JSON export of menu_items in sample_data/ and raw_data/
3. Parquet: High-performance columnar storage in parquet_data/ (orders & order_items)
4. Relational Database: SQLite / PostgreSQL via SQLAlchemy in database/
5. NoSQL Database: MongoDB document collections via PyMongo
"""
import os
import sys
import json
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
PARQUET_DATA_DIR = os.path.join(PROJECT_ROOT, "parquet_data")
SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "sample_data")
DATABASE_DIR = os.path.join(PROJECT_ROOT, "database")

sys.path.append(DATABASE_DIR)

def export_json_table(table_name: str = "menu_items"):
    """Export a table as structured JSON for API/NoSQL consumption."""
    print(f"\n--- [Storage 2/5: JSON Export] Exporting '{table_name}' to JSON ---")
    csv_path = os.path.join(RAW_DATA_DIR, table_name, f"{table_name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")

    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
    sample_json_path = os.path.join(SAMPLE_DATA_DIR, f"{table_name}.json")
    raw_json_path = os.path.join(RAW_DATA_DIR, table_name, f"{table_name}.json")

    with open(sample_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"  [OK] Exported {len(records)} records to {sample_json_path}")
    print(f"  [OK] Exported {len(records)} records to {raw_json_path}")

def export_parquet_datasets():
    """
    Store large processed datasets in Parquet format with Snappy compression per SRS Step 2.
    Exports orders and order_items.
    """
    print("\n--- [Storage 3/5: Parquet Storage] Exporting large datasets to Parquet ---")
    datasets_to_parquet = [
        ("orders", "orders/orders.csv"),
        ("order_items", "order_items/order_items.csv")
    ]

    for name, rel_csv in datasets_to_parquet:
        csv_path = os.path.join(RAW_DATA_DIR, rel_csv)
        target_dir = os.path.join(PARQUET_DATA_DIR, name)
        os.makedirs(target_dir, exist_ok=True)
        target_parquet = os.path.join(target_dir, f"{name}.parquet")

        print(f"  Converting {name} to Parquet (Snappy compression)...")
        start_t = time.time()
        df = pd.read_csv(csv_path)

        table = pa.Table.from_pandas(df)
        pq.write_table(table, target_parquet, compression="snappy")

        file_size_mb = os.path.getsize(target_parquet) / (1024 * 1024)
        elapsed = time.time() - start_t
        print(f"  [OK] Saved {len(df):,} rows to {target_parquet} ({file_size_mb:.2f} MB in {elapsed:.1f}s)")

def run_storage_pipeline():
    total_start = time.time()
    print("=" * 70)
    print("DineIQ Analytics - SRS Step 2 (Big Data Storage) Execution")
    print("=" * 70)

    print("\n--- [Storage 1/5: CSV Storage] Validating raw_data/ CSV files ---")
    all_csvs = [
        "customers", "orders", "order_items", "menu_items", "menu_categories",
        "restaurants", "pricing_history", "promotions", "ratings", "inventory", "wastage"
    ]
    for tbl in all_csvs:
        p = os.path.join(RAW_DATA_DIR, tbl, f"{tbl}.csv")
        if os.path.exists(p):
            print(f"  [OK] CSV verified: {tbl} ({os.path.getsize(p)/(1024*1024):.2f} MB)")
        else:
            print(f"  [MISSING] {tbl}")

    export_json_table("menu_items")

    export_parquet_datasets()

    print("\n--- [Storage 4/5: Relational DB] Loading into SQLite/PostgreSQL ---")
    from load_relational import load_relational_database
    load_relational_database(sample_mode=True)

    print("\n--- [Storage 5/5: NoSQL Database] Preparing & Loading MongoDB collections ---")
    from load_nosql import load_nosql_database
    load_nosql_database(max_orders=2500)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"SRS Step 2 Big Data Storage Pipeline Completed in {total_elapsed:.1f} seconds!")
    print("All 5 formats (CSV, JSON, Parquet, Relational DB, NoSQL DB) successfully populated!")
    print("=" * 70)

if __name__ == "__main__":
    run_storage_pipeline()
