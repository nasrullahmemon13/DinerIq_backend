"""
DineIQ Analytics - Independent Python Data Ingestion Engine (SRS Step 13)
Enforces SRS explicit rule:
"Spark-generated predictions must not simply be exported and reused as Python results"
Reads strictly from processed_data/cleaned/ and parquet_data/ operational stores.
Never ingests or relies upon Spark model artifacts or Spark predictions.
"""
import os
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CLEANED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")

def load_cleaned_table(table_name: str) -> pd.DataFrame:
    """
    Loads a cleaned entity table directly from Parquet or CSV.
    Guarantees independent Python processing without relying on Spark output.
    """
    parquet_path = os.path.join(CLEANED_DIR, table_name, f"{table_name}.parquet")
    csv_path = os.path.join(CLEANED_DIR, table_name, f"{table_name}.csv")

    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        raw_csv = os.path.join(RAW_DATA_DIR, table_name, f"{table_name}.csv")
        if os.path.exists(raw_csv):
            return pd.read_csv(raw_csv)
        raise FileNotFoundError(f"Cleaned table '{table_name}' not found in {CLEANED_DIR}")

def load_operational_data():
    """Loads all operational entities required for independent Python modeling."""
    orders = load_cleaned_table("orders")
    order_items = load_cleaned_table("order_items")
    customers = load_cleaned_table("customers")
    menu_items = load_cleaned_table("menu_items")
    restaurants = load_cleaned_table("restaurants")
    pricing_history = load_cleaned_table("pricing_history")

    return {
        "orders": orders,
        "order_items": order_items,
        "customers": customers,
        "menu_items": menu_items,
        "restaurants": restaurants,
        "pricing_history": pricing_history
    }
