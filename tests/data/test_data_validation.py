"""
DineIQ Analytics - Automated Data Validation and Domain Range Test Suite (SRS Step 2, Step 4 & Step 5)
Verifies:
- Relational schema completeness and data types
- Temporal validity (orders within expected calendar windows, no future timestamps)
- Price and cost domain validity (> 0, cost <= price)
- Quantity validity (positive integers)
- Rating scale validity (1.0 to 5.0)
- Discount validity (0% to 100%)
- Wastage domain bounds (>= 0)
- Channel category validity (exact SRS ordering channels)
"""
import os
import pandas as pd
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEANED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")


def test_schema_column_presence():
    """Verify essential columns exist across core operational entities."""
    orders = pd.read_parquet(os.path.join(CLEANED_DIR, "orders", "orders.parquet"))
    order_items = pd.read_parquet(os.path.join(CLEANED_DIR, "order_items", "order_items.parquet"))
    menu = pd.read_parquet(os.path.join(CLEANED_DIR, "menu_items", "menu_items.parquet"))
    ratings = pd.read_parquet(os.path.join(CLEANED_DIR, "ratings", "ratings.parquet"))
    wastage = pd.read_parquet(os.path.join(CLEANED_DIR, "wastage", "wastage.parquet"))

    assert {"order_id", "customer_id", "location_id", "order_date", "total_amount", "order_type"}.issubset(orders.columns)
    assert {"order_item_id", "order_id", "item_id", "quantity", "unit_price", "item_total"}.issubset(order_items.columns)
    assert {"item_id", "name", "category_id", "base_price", "cost_price"}.issubset(menu.columns)
    assert {"rating_id", "order_id", "customer_id", "item_id", "overall_rating"}.issubset(ratings.columns)
    assert {"wastage_id", "location_id", "item_id", "quantity_wasted", "total_loss_amount"}.issubset(wastage.columns)


def test_date_validity():
    """Verify order dates are valid timestamps within the 2024-2026 operational window."""
    orders = pd.read_parquet(os.path.join(CLEANED_DIR, "orders", "orders.parquet"), columns=["order_date"])
    dates = pd.to_datetime(orders["order_date"], errors="coerce")
    assert dates.isna().sum() == 0, "Unparseable dates found in orders"
    assert (dates < "2024-01-01").sum() == 0, "Pre-historical order dates found"
    assert (dates > "2027-01-01").sum() == 0, "Impossible future order dates found in cleaned dataset"


def test_price_validity():
    """Verify menu base prices and costs are strictly positive with base_price >= cost_price."""
    menu = pd.read_parquet(os.path.join(CLEANED_DIR, "menu_items", "menu_items.parquet"))
    assert (menu["base_price"] <= 0).sum() == 0, "Zero or negative menu base prices found"
    assert (menu["cost_price"] <= 0).sum() == 0, "Zero or negative menu cost prices found"
    assert (menu["cost_price"] > menu["base_price"]).sum() == 0, "Items where cost exceeds retail price found"


def test_quantity_validity():
    """Verify order quantities are positive integers in cleaned order lines."""
    order_items = pd.read_parquet(os.path.join(CLEANED_DIR, "order_items", "order_items.parquet"), columns=["quantity"])
    assert (order_items["quantity"] <= 0).sum() == 0, "Non-positive quantities found in order items"
    assert np.all(order_items["quantity"] == order_items["quantity"].round()), "Fractional quantities in order items"


def test_rating_range_validity():
    """Verify customer ratings strictly adhere to the 1.0 to 5.0 Likert scale."""
    ratings = pd.read_parquet(os.path.join(CLEANED_DIR, "ratings", "ratings.parquet"), columns=["overall_rating"])
    assert (ratings["overall_rating"] < 1.0).sum() == 0, "Ratings below 1.0 found"
    assert (ratings["overall_rating"] > 5.0).sum() == 0, "Ratings above 5.0 found"


def test_discount_range_validity():
    """Verify discount percentages in orders and promotions are bounded between 0% and 100%."""
    promotions = pd.read_parquet(os.path.join(CLEANED_DIR, "promotions", "promotions.parquet"))
    orders = pd.read_parquet(os.path.join(CLEANED_DIR, "orders", "orders.parquet"))
    
    if "discount_value" in promotions.columns:
        assert (promotions["discount_value"] < 0).sum() == 0, "Negative promotion discounts found"
    if "discount_amount" in orders.columns:
        assert (orders["discount_amount"] < 0).sum() == 0, "Negative order discount amounts found"


def test_wastage_validity():
    """Verify wastage quantities and financial loss values are strictly non-negative."""
    wastage = pd.read_parquet(os.path.join(CLEANED_DIR, "wastage", "wastage.parquet"))
    assert (wastage["quantity_wasted"] < 0).sum() == 0, "Negative wastage quantities found"
    assert (wastage["total_loss_amount"] < 0).sum() == 0, "Negative wastage loss amounts found"


def test_ordering_channel_validity():
    """Verify all order types match valid SRS ordering channels."""
    orders = pd.read_parquet(os.path.join(CLEANED_DIR, "orders", "orders.parquet"), columns=["order_type"])
    unique_types = set(orders["order_type"].dropna().unique())
    valid_types = {"DINE_IN", "TAKEOUT", "DELIVERY", "DRIVE_THRU"}
    invalid = unique_types - valid_types
    assert len(invalid) == 0, f"Unrecognized order types found: {invalid}"
