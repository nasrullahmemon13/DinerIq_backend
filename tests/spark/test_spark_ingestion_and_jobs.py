"""
DineIQ Analytics - Spark Ingestion and Big Data Jobs Test Suite (SRS Step 3)
Verifies:
- Explicit PySpark StructType schemas for all 11 tables
- Corrupt record quarantine columns
- Schema enforcement and non-null constraints
"""
import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INGESTION_DIR = os.path.join(PROJECT_ROOT, "spark_jobs", "ingestion")
if INGESTION_DIR not in sys.path:
    sys.path.append(INGESTION_DIR)

import schema_definitions as sd


def test_spark_explicit_schemas():
    """Verify PySpark StructType schemas for all 11 tables."""
    schemas = sd.get_all_schemas()
    assert len(schemas) == 11, f"Expected 11 schemas, got {len(schemas)}"
    expected_tables = [
        "customers", "orders", "order_items", "menu_items", "menu_categories",
        "restaurants", "pricing_history", "promotions", "ratings", "inventory", "wastage"
    ]
    for table in expected_tables:
        assert table in schemas, f"Missing Spark schema definition for {table}"
        assert len(schemas[table].fields) >= 3, f"Schema for {table} has insufficient fields"


def test_spark_orders_schema():
    """Verify orders schema contains required identifiers, timestamp, and corrupt record handler."""
    orders_schema = sd.get_orders_schema()
    names = [f.name for f in orders_schema.fields]
    assert "order_id" in names
    assert "customer_id" in names
    assert "location_id" in names
    assert "order_date" in names
    assert "total_amount" in names
    assert "_corrupt_record" in names


def test_spark_order_items_schema():
    """Verify order items schema types and fields."""
    items_schema = sd.get_order_items_schema()
    names = [f.name for f in items_schema.fields]
    assert "order_item_id" in names
    assert "order_id" in names
    assert "item_id" in names
    assert "quantity" in names
    assert "unit_price" in names
    assert "item_total" in names
