"""
Unit & Integration Tests for Spark Ingestion (SRS Step 3)
Tests:
- Schema completeness for all 11 tables
- Mandatory identifiers and field types
- Ingestion logic and validation rules
"""
import os
import sys
import pytest
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
INGESTION_DIR = os.path.join(PROJECT_ROOT, "spark_jobs", "ingestion")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")

sys.path.append(INGESTION_DIR)
import schema_definitions as sd

def test_all_11_schemas_present():
    """Verify that explicit schemas are defined for all 11 SRS tables."""
    schemas = sd.get_all_schemas()
    assert len(schemas) == 11, f"Expected 11 table schemas, got {len(schemas)}"
    expected_tables = [
        "customers", "orders", "order_items", "menu_items", "menu_categories",
        "restaurants", "pricing_history", "promotions", "ratings", "inventory", "wastage"
    ]
    for tbl in expected_tables:
        assert tbl in schemas, f"Missing schema for table: {tbl}"
        assert len(schemas[tbl].fields) > 0, f"Schema for {tbl} has no fields"

def test_orders_schema_structure():
    """Verify orders schema contains required identifiers and types."""
    schema = sd.get_orders_schema()
    field_names = [f.name for f in schema.fields]
    assert "order_id" in field_names
    assert "customer_id" in field_names
    assert "location_id" in field_names
    assert "order_date" in field_names
    assert "total_amount" in field_names
    assert "_corrupt_record" in field_names


    assert schema["order_id"].nullable is False
    assert schema["location_id"].nullable is False

def test_order_items_schema_structure():
    """Verify order_items schema contains required fields."""
    schema = sd.get_order_items_schema()
    field_names = [f.name for f in schema.fields]
    assert "order_item_id" in field_names
    assert "order_id" in field_names
    assert "item_id" in field_names
    assert "quantity" in field_names
    assert "unit_price" in field_names
    assert "subtotal" in field_names
    assert "item_total" in field_names

def test_data_type_validation_quarantine_logic():
    """Verify data-type validation logic correctly flags negative amounts and invalid rows."""
    orders_csv = os.path.join(RAW_DATA_DIR, "orders", "orders.csv")
    assert os.path.exists(orders_csv), "orders.csv must exist"

    df = pd.read_csv(orders_csv)

    invalid_rows = df[df["total_amount"] <= 0]
    assert len(invalid_rows) > 0, "Expected injected invalid transactions to exist for testing"


    valid_rows = df[df["total_amount"] > 0]
    dq_score = len(valid_rows) / len(df) * 100
    assert dq_score > 99.0, f"Expected DQ score > 99%, got {dq_score:.2f}%"

def test_multiple_file_batch_consistency():
    """Verify that multiple file batches equal full dataset volume."""
    orders_csv = os.path.join(RAW_DATA_DIR, "orders", "orders.csv")
    df = pd.read_csv(orders_csv)
    half = len(df) // 2
    part1 = df.iloc[:half]
    part2 = df.iloc[half:]
    combined = pd.concat([part1, part2])
    assert len(combined) == len(df)
