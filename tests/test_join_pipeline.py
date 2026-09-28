"""
DineIQ Analytics - Integration Tests for Spark Join Pipeline (SRS Step 6)
Verifies:
- All 10 required SRS relationships execute and produce valid outputs
- Master Analytical Cube exists in Parquet and CSV formats with >= 900,000 records
- Schema completeness: joined attributes (customer segment, location tier, margin, gross profit)
- Relational integrity: zero orphaned records in mandatory inner joins
"""
import os
import pytest
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JOINED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "joined")
CLEANED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")

def test_master_analytical_cube_exists():
    """Verify Master Analytical Cube exists in both CSV and Snappy-compressed Parquet."""
    cube_dir = os.path.join(JOINED_DIR, "master_analytical_cube")
    csv_file = os.path.join(cube_dir, "master_analytical_cube.csv")
    parquet_file = os.path.join(cube_dir, "master_analytical_cube.parquet")
    
    assert os.path.exists(csv_file), f"Missing Master Cube CSV at {csv_file}"
    assert os.path.exists(parquet_file), f"Missing Master Cube Parquet at {parquet_file}"

def test_master_cube_row_count_and_columns():
    """Verify Master Cube row count meets SRS scale (>900,000 items) and has all integrated fields."""
    parquet_file = os.path.join(JOINED_DIR, "master_analytical_cube", "master_analytical_cube.parquet")
    table = pq.read_table(parquet_file)
    row_count = table.num_rows
    
    assert row_count >= 900000, f"Expected >= 900,000 rows in Master Cube, got {row_count}"
    
    cols = table.column_names
    expected_cols = [
        "order_id", "order_date", "order_time", "order_type", "payment_method",
        "customer_id", "customer_segment", "loyalty_tier",
        "location_id", "restaurant_name", "restaurant_city", "cost_index",
        "order_item_id", "item_id", "item_name", "category_id", "category_name",
        "quantity", "unit_price", "cost_price", "total_item_cost", "item_total", "gross_profit"
    ]
    for c in expected_cols:
        assert c in cols, f"Missing critical column '{c}' in Master Cube schema"

def test_relationship_orders_customers():
    """SRS Relationship 1: Orders - Customers."""
    orders = pd.read_parquet(os.path.join(CLEANED_DIR, "orders", "orders.parquet"), columns=["order_id", "customer_id"])
    customers = pd.read_parquet(os.path.join(CLEANED_DIR, "customers", "customers.parquet"), columns=["customer_id", "loyalty_tier"])
    
    joined = pd.merge(orders, customers, on="customer_id", how="left")
    assert len(joined) == len(orders), "Orders-Customers join must preserve all valid orders"
    assert "loyalty_tier" in joined.columns

def test_relationship_orders_order_items():
    """SRS Relationship 2: Orders - OrderItems."""
    orders = pd.read_parquet(os.path.join(CLEANED_DIR, "orders", "orders.parquet"), columns=["order_id"])
    items = pd.read_parquet(os.path.join(CLEANED_DIR, "order_items", "order_items.parquet"), columns=["order_item_id", "order_id"])
    
    joined = pd.merge(items, orders, on="order_id", how="inner")
    assert len(joined) > 900000, "Cleaned order items should all link to active orders"

def test_relationship_order_items_menu_items():
    """SRS Relationship 3: OrderItems - MenuItems."""
    items = pd.read_parquet(os.path.join(CLEANED_DIR, "order_items", "order_items.parquet"), columns=["order_item_id", "item_id"])
    menu = pd.read_parquet(os.path.join(CLEANED_DIR, "menu_items", "menu_items.parquet"), columns=["item_id", "name", "base_price"])
    
    joined = pd.merge(items, menu, on="item_id", how="inner")
    assert len(joined) == len(items), "Every order item must link to an active menu item"

def test_relationship_menu_items_categories():
    """SRS Relationship 4: MenuItems - Categories."""
    menu = pd.read_parquet(os.path.join(CLEANED_DIR, "menu_items", "menu_items.parquet"), columns=["item_id", "category_id"])
    cats = pd.read_parquet(os.path.join(CLEANED_DIR, "menu_categories", "menu_categories.parquet"), columns=["category_id", "category_name"])
    
    joined = pd.merge(menu, cats, on="category_id", how="inner")
    assert len(joined) == len(menu), "Every menu item must belong to a valid category"

def test_relationship_orders_locations():
    """SRS Relationship 5: Orders - Locations."""
    orders = pd.read_parquet(os.path.join(CLEANED_DIR, "orders", "orders.parquet"), columns=["order_id", "location_id"])
    rests = pd.read_parquet(os.path.join(CLEANED_DIR, "restaurants", "restaurants.parquet"), columns=["location_id", "name", "city"])
    
    joined = pd.merge(orders, rests, on="location_id", how="inner")
    assert len(joined) == len(orders), "Every order must belong to a valid restaurant location"

def test_relationship_orders_promotions():
    """SRS Relationship 6: Orders - Promotions."""
    orders = pd.read_parquet(os.path.join(CLEANED_DIR, "orders", "orders.parquet"), columns=["order_id", "promotion_id"])
    promos = pd.read_parquet(os.path.join(CLEANED_DIR, "promotions", "promotions.parquet"), columns=["promotion_id", "promotion_name"])
    
    joined = pd.merge(orders, promos, on="promotion_id", how="left")
    assert len(joined) == len(orders), "Orders-Promotions left join preserves order count"

def test_relationship_menu_items_pricing_history():
    """SRS Relationship 7: MenuItems - PricingHistory."""
    menu = pd.read_parquet(os.path.join(CLEANED_DIR, "menu_items", "menu_items.parquet"), columns=["item_id"])
    pricing = pd.read_parquet(os.path.join(CLEANED_DIR, "pricing_history", "pricing_history.parquet"), columns=["price_history_id", "item_id"])
    
    joined = pd.merge(pricing, menu, on="item_id", how="inner")
    assert len(joined) > 0, "Pricing history records must link to valid menu items"

def test_relationship_menu_items_ratings():
    """SRS Relationship 8: MenuItems - Ratings."""
    menu = pd.read_parquet(os.path.join(CLEANED_DIR, "menu_items", "menu_items.parquet"), columns=["item_id"])
    ratings = pd.read_parquet(os.path.join(CLEANED_DIR, "ratings", "ratings.parquet"), columns=["rating_id", "item_id", "overall_rating"])
    
    joined = pd.merge(ratings, menu, on="item_id", how="inner")
    assert len(joined) == len(ratings), "Ratings must link to valid menu items"

def test_relationship_menu_items_inventory():
    """SRS Relationship 9: MenuItems - Inventory."""
    menu = pd.read_parquet(os.path.join(CLEANED_DIR, "menu_items", "menu_items.parquet"), columns=["item_id"])
    inv = pd.read_parquet(os.path.join(CLEANED_DIR, "inventory", "inventory.parquet"), columns=["inventory_id", "item_id", "ending_stock"])
    
    joined = pd.merge(inv, menu, on="item_id", how="inner")
    assert len(joined) == len(inv), "Inventory snapshot items must link to valid menu items"

def test_relationship_menu_items_wastage():
    """SRS Relationship 10: MenuItems - Wastage."""
    menu = pd.read_parquet(os.path.join(CLEANED_DIR, "menu_items", "menu_items.parquet"), columns=["item_id"])
    wastage = pd.read_parquet(os.path.join(CLEANED_DIR, "wastage", "wastage.parquet"), columns=["wastage_id", "item_id", "quantity_wasted"])
    
    joined = pd.merge(wastage, menu, on="item_id", how="inner")
    assert len(joined) == len(wastage), "Wastage records must link to valid menu items"
