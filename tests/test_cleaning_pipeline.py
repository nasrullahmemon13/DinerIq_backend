"""
Unit & Integration Tests for Cleaning Pipeline (SRS Step 5)
Verifies:
- All cleaned datasets exist in processed_data/cleaned/ (CSV & Parquet)
- Zero duplicates in cleaned orders and order_items
- Zero negative quantities, missing menu IDs, or invalid dates
- Quarantine directory integrity and manifest
- Audit log completeness in cleaning_log.md
"""
import os
import json
import pytest
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEANED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")
QUARANTINE_DIR = os.path.join(PROJECT_ROOT, "processed_data", "quarantine")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "data_quality")

def test_cleaned_datasets_exist():
    """Verify all major cleaned tables exist in both CSV and Parquet format."""
    required_entities = ["orders", "order_items", "customers", "menu_items", "restaurants", "ratings", "wastage"]
    for entity in required_entities:
        csv_path = os.path.join(CLEANED_DIR, entity, f"{entity}.csv")
        parquet_path = os.path.join(CLEANED_DIR, entity, f"{entity}.parquet")
        assert os.path.exists(csv_path), f"Missing cleaned CSV: {csv_path}"
        assert os.path.exists(parquet_path), f"Missing cleaned Parquet: {parquet_path}"

def test_cleaned_orders_integrity():
    """Verify cleaned orders have 0 duplicates, valid dates, and valid restaurant IDs."""
    orders_csv = os.path.join(CLEANED_DIR, "orders", "orders.csv")
    df = pd.read_csv(orders_csv)
    
                  
    assert df["order_id"].duplicated().sum() == 0, "Cleaned orders must have 0 duplicate IDs"
                              
    assert (~df["location_id"].str.startswith("LOC-")).sum() == 0
    assert "LOC-999" not in set(df["location_id"])
                     
    assert (df["order_date"] > "2025-12-31").sum() == 0
                          
    assert (df["discount_amount"] < 0).sum() == 0
                                                         
    assert df["customer_id"].isnull().sum() == 0

def test_cleaned_order_items_integrity():
    """Verify cleaned order_items have 0 duplicates, positive quantities, and valid item IDs."""
    items_csv = os.path.join(CLEANED_DIR, "order_items", "order_items.csv")
    df = pd.read_csv(items_csv)
    
                  
    assert df["order_item_id"].duplicated().sum() == 0, "Cleaned items must have 0 duplicate IDs"
                              
    assert (df["quantity"] <= 0).sum() == 0, "Cleaned items must have positive quantities"
                           
    assert df["item_id"].isnull().sum() == 0, "Cleaned items must have no null item_id"

def test_quarantine_manifest_and_files():
    """Verify quarantine manifest exists and references valid quarantine files."""
    manifest_path = os.path.join(QUARANTINE_DIR, "quarantine_manifest.json")
    assert os.path.exists(manifest_path), f"Missing {manifest_path}"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest["total_quarantined_batches"] > 0
    assert manifest["total_quarantined_records"] > 0
    for batch in manifest["batches"]:
        assert os.path.exists(batch["csv_path"])
        assert os.path.exists(batch["parquet_path"])

def test_cleaning_log_exists_and_complete():
    """Verify cleaning_log.md records all cleaning decisions."""
    log_path = os.path.join(REPORTS_DIR, "cleaning_log.md")
    assert os.path.exists(log_path), f"Missing {log_path}"
    
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "Data Cleaning Decisions Audit Log" in content
    assert "RULE-02" in content
    assert "RULE-03" in content
    assert "RULE-06" in content
    assert "RULE-10" in content
    assert "RULE-05" in content
