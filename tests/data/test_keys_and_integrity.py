"""
DineIQ Analytics - Automated Primary and Foreign Key Integrity Test Suite (SRS Step 1 & Step 3)
Verifies:
- Uniqueness of primary keys across operational entity tables
- Referential integrity of all foreign key relationships in cleaned datasets
"""
import os
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEANED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")


def load_clean(name: str, cols=None) -> pd.DataFrame:
    pq_path = os.path.join(CLEANED_DIR, name, f"{name}.parquet")
    return pd.read_parquet(pq_path, columns=cols)


def test_primary_key_uniqueness():
    """Verify primary keys have 0 duplicates across cleaned entity master tables."""
    pk_checks = [
        ("customers", "customer_id"),
        ("orders", "order_id"),
        ("order_items", "order_item_id"),
        ("menu_items", "item_id"),
        ("menu_categories", "category_id"),
        ("restaurants", "restaurant_id"),
        ("wastage", "wastage_id"),
        ("pricing_history", "price_history_id"),
        ("promotions", "promotion_id"),
    ]
    for table_name, pk_col in pk_checks:
        df = load_clean(table_name, [pk_col])
        dup_count = df[pk_col].duplicated().sum()
        assert dup_count == 0, f"Table {table_name} has {dup_count} duplicate primary keys in {pk_col}"

                                          
    ratings = load_clean("ratings", ["rating_id"])
    assert ratings["rating_id"].nunique() >= 100_000, "Unique rating_ids below 100,000 threshold"


def test_order_to_customer_foreign_key_integrity():
    """Verify all registered orders map to valid customers (excluding guest accounts)."""
    orders = load_clean("orders", ["customer_id"])
    customers = load_clean("customers", ["customer_id"])
    valid_customers = set(customers["customer_id"].dropna().unique())
    
                                     
    registered_order_custs = set(orders[orders["customer_id"] != "CUST-GUEST"]["customer_id"].dropna().unique())
    orphan_custs = registered_order_custs - valid_customers
    assert len(orphan_custs) == 0, f"Found {len(orphan_custs)} orphan customer references in orders"


def test_order_to_restaurant_foreign_key_integrity():
    """Verify all orders map to valid restaurant locations."""
    orders = load_clean("orders", ["location_id"])
    restaurants = load_clean("restaurants", ["location_id"])
    valid_locations = set(restaurants["location_id"].unique())
    
    order_locs = set(orders["location_id"].unique())
    orphan_locs = order_locs - valid_locations
    assert len(orphan_locs) == 0, f"Found orphan restaurant location references: {orphan_locs}"


def test_order_items_to_order_and_menu_foreign_key_integrity():
    """Verify all order line items map to valid orders and valid menu items."""
    order_items = load_clean("order_items", ["order_id", "item_id"])
    orders = load_clean("orders", ["order_id"])
    menu_items = load_clean("menu_items", ["item_id"])

    valid_orders = set(orders["order_id"].unique())
    valid_items = set(menu_items["item_id"].unique())

    orphan_orders = set(order_items["order_id"].unique()) - valid_orders
    assert len(orphan_orders) == 0, f"Found {len(orphan_orders)} orphan order references in order_items"

    orphan_items = set(order_items["item_id"].unique()) - valid_items
    assert len(orphan_items) == 0, f"Found {len(orphan_items)} orphan item references in order_items"


def test_ratings_foreign_key_integrity():
    """Verify ratings connect to valid customers, menu items, and locations."""
    ratings = load_clean("ratings", ["customer_id", "item_id", "location_id"])
    customers = load_clean("customers", ["customer_id"])
    menu_items = load_clean("menu_items", ["item_id"])
    restaurants = load_clean("restaurants", ["location_id"])

    valid_customers = set(customers["customer_id"].unique())
    valid_items = set(menu_items["item_id"].unique())
    valid_locs = set(restaurants["location_id"].unique())

    rating_custs = set(ratings[ratings["customer_id"] != "CUST-GUEST"]["customer_id"].dropna().unique())
    rating_items = set(ratings["item_id"].dropna().unique())
    rating_locs = set(ratings["location_id"].dropna().unique())

    assert len(rating_custs - valid_customers) == 0, "Orphan customer references in ratings"
    assert len(rating_items - valid_items) == 0, "Orphan item references in ratings"
    assert len(rating_locs - valid_locs) == 0, "Orphan location references in ratings"


def test_wastage_foreign_key_integrity():
    """Verify wastage logs connect to valid menu items and restaurant locations."""
    wastage = load_clean("wastage", ["item_id", "location_id"])
    menu_items = load_clean("menu_items", ["item_id"])
    restaurants = load_clean("restaurants", ["location_id"])

    valid_items = set(menu_items["item_id"].unique())
    valid_locs = set(restaurants["location_id"].unique())

    waste_items = set(wastage["item_id"].dropna().unique())
    waste_locs = set(wastage["location_id"].dropna().unique())

    assert len(waste_items - valid_items) == 0, "Orphan item references in wastage"
    assert len(waste_locs - valid_locs) == 0, "Orphan location references in wastage"


def test_menu_items_to_categories_foreign_key_integrity():
    """Verify all menu items map to valid categories."""
    menu_items = load_clean("menu_items", ["category_id"])
    categories = load_clean("menu_categories", ["category_id"])

    valid_cats = set(categories["category_id"].unique())
    item_cats = set(menu_items["category_id"].unique())

    assert len(item_cats - valid_cats) == 0, "Orphan category references in menu_items"
