"""
DineIQ Analytics - Automated Dataset Minimum Sizes Test Suite (SRS Step 1 & Step 50)
Validates that the generated and processed operational restaurant datasets satisfy all minimum volume requirements.
"""
import os
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
CLEANED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")


def test_order_line_records_volume():
    """Verify minimum 1,000,000 order-line records in raw data feed."""
    raw_csv = os.path.join(RAW_DATA_DIR, "order_items", "order_items.csv")
    cleaned_parquet = os.path.join(CLEANED_DIR, "order_items", "order_items.parquet")
    
    assert os.path.exists(raw_csv), f"Raw order_items CSV missing at {raw_csv}"
    with open(raw_csv, "rb") as f:
        raw_count = sum(1 for _ in f) - 1
    assert raw_count >= 1_000_000, f"Raw order items ({raw_count:,}) below 1,000,000 minimum"

    if os.path.exists(cleaned_parquet):
        df_cleaned = pd.read_parquet(cleaned_parquet)
        assert len(df_cleaned) >= 800_000, f"Cleaned order items ({len(df_cleaned):,}) unexpectedly low"


def test_unique_orders_volume():
    """Verify at least 100,000 unique orders generated."""
    raw_csv = os.path.join(RAW_DATA_DIR, "orders", "orders.csv")
    assert os.path.exists(raw_csv), f"Raw orders CSV missing at {raw_csv}"
    with open(raw_csv, "rb") as f:
        raw_count = sum(1 for _ in f) - 1
    assert raw_count >= 100_000, f"Raw orders ({raw_count:,}) below 100,000 minimum"


def test_customers_volume():
    """Verify at least 50,000 customers."""
    pq_path = os.path.join(CLEANED_DIR, "customers", "customers.parquet")
    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path)
    else:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "customers", "customers.csv"))
    assert len(df) >= 50_000, f"Customer count ({len(df):,}) below 50,000 minimum"


def test_menu_items_volume():
    """Verify at least 150 distinct menu items."""
    pq_path = os.path.join(CLEANED_DIR, "menu_items", "menu_items.parquet")
    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path)
    else:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "menu_items", "menu_items.csv"))
    assert len(df) >= 150, f"Menu item count ({len(df)}) below 150 minimum"


def test_menu_categories_volume():
    """Verify at least 10 menu categories."""
    pq_path = os.path.join(CLEANED_DIR, "menu_categories", "menu_categories.parquet")
    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path)
    else:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "menu_categories", "menu_categories.csv"))
    assert len(df) >= 10, f"Category count ({len(df)}) below 10 minimum"


def test_restaurants_locations_volume():
    """Verify at least 20 restaurant locations."""
    pq_path = os.path.join(CLEANED_DIR, "restaurants", "restaurants.parquet")
    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path)
    else:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "restaurants", "restaurants.csv"))
    assert len(df) >= 20, f"Restaurant count ({len(df)}) below 20 minimum"


def test_transaction_history_duration():
    """Verify at least 12 months of transaction history."""
    pq_path = os.path.join(CLEANED_DIR, "orders", "orders.parquet")
    df = pd.read_parquet(pq_path, columns=["order_date"])
    dates = pd.to_datetime(df["order_date"])
    min_date = dates.min()
    max_date = dates.max()
    days_span = (max_date - min_date).days
    assert days_span >= 360, f"Transaction history ({days_span} days) is less than 12 months"


def test_ratings_volume():
    """Verify at least 100,000 rating records."""
    pq_path = os.path.join(CLEANED_DIR, "ratings", "ratings.parquet")
    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path)
    else:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "ratings", "ratings.csv"))
    assert len(df) >= 100_000, f"Rating records ({len(df):,}) below 100,000 minimum"


def test_wastage_records_volume():
    """Verify at least 50,000 wastage records in raw data feed."""
    raw_csv = os.path.join(RAW_DATA_DIR, "wastage", "wastage.csv")
    with open(raw_csv, "rb") as f:
        raw_count = sum(1 for _ in f) - 1
    assert raw_count >= 50_000, f"Raw wastage records ({raw_count:,}) below 50,000 minimum"


def test_pricing_history_and_promotions_volume():
    """Verify multiple historical pricing records and multiple promotion campaigns."""
    pricing_df = pd.read_parquet(os.path.join(CLEANED_DIR, "pricing_history", "pricing_history.parquet"))
    promos_df = pd.read_parquet(os.path.join(CLEANED_DIR, "promotions", "promotions.parquet"))
    assert len(pricing_df) >= 100, f"Pricing history ({len(pricing_df)}) below expected minimum"
    assert len(promos_df) >= 5, f"Promotion campaigns ({len(promos_df)}) below expected minimum"
