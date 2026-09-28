"""
DineIQ Analytics - Relational Database Loader (SQLAlchemy)
Loads raw/processed datasets into Relational Database (SQLite / PostgreSQL)
per SRS Section 2 & Step 2 requirements.
"""
import os
import sys
import time
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
sys.path.append(CURRENT_DIR)

from models import Base

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

def get_engine():
    """Create SQLAlchemy engine using DATABASE_URL if available, else local SQLite."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        db_path = os.path.join(CURRENT_DIR, "dineiq.db")
        db_url = f"sqlite:///{db_path}"
    print(f"Connecting to database: {db_url}")
    return create_engine(db_url, echo=False)

def load_relational_database(sample_mode: bool = False):
    """
    Creates tables and loads data from raw_data into relational DB.
    If sample_mode=True, loads a representative sample for fast verification.
    """
    start_time = time.time()
    engine = get_engine()

    print("Creating relational schema tables via SQLAlchemy metadata...")
    Base.metadata.create_all(engine)
    print("[OK] Schema created.")

    table_files = [
        ("restaurants", "restaurants/restaurants.csv", 1000),
        ("menu_categories", "menu_categories/menu_categories.csv", 1000),
        ("menu_items", "menu_items/menu_items.csv", 1000),
        ("customers", "customers/customers.csv", 10000),
        ("promotions", "promotions/promotions.csv", 1000),
        ("pricing_history", "pricing_history/pricing_history.csv", 2000),
        ("orders", "orders/orders.csv", 10000),
        ("order_items", "order_items/order_items.csv", 20000),
        ("ratings", "ratings/ratings.csv", 10000),
        ("inventory", "inventory/inventory.csv", 5000),
        ("wastage", "wastage/wastage.csv", 10000)
    ]

    with engine.connect() as conn:
        for tbl_name, rel_csv, chunksize in table_files:
            csv_path = os.path.join(RAW_DATA_DIR, rel_csv)
            if not os.path.exists(csv_path):
                print(f"[SKIP] {csv_path} does not exist.")
                continue

            print(f"Loading {tbl_name} from {rel_csv}...")
            df = pd.read_csv(csv_path)
            if sample_mode and len(df) > 5000:
                df = df.head(5000)
                print(f"  [Sample Mode] Truncated to {len(df)} rows.")

            pk_col = df.columns[0]
            if df[pk_col].duplicated().any():
                dup_count = df[pk_col].duplicated().sum()
                df = df.drop_duplicates(subset=[pk_col])
                print(f"  Deduplicated {dup_count} records for relational primary key constraint.")

            df.to_sql(tbl_name, con=engine, if_exists="replace", index=False, chunksize=chunksize)
            print(f"  [OK] Ingested {len(df):,} rows into '{tbl_name}'.")

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Relational database loaded in {elapsed:.1f} seconds!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Load sample data for rapid test")
    args = parser.parse_args()
    load_relational_database(sample_mode=args.sample)
