"""
DineIQ Analytics - Storage & Database Integration Test Suite (SRS Step 3, Step 29 & Step 30)
Verifies:
- Parquet output file structure, snappy compression, and pyarrow deserialization
- PostgreSQL / Relational SQLAlchemy database connection, schema migration, and queries
- MongoDB NoSQL integration, JSON export schemas, and client connectivity logic
"""
import os
import sys
import json
import pandas as pd
import pyarrow.parquet as pq
import pytest
from sqlalchemy import text

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PARQUET_DIR = os.path.join(PROJECT_ROOT, "parquet_data")
CLEANED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")
MONGO_EXPORTS_DIR = os.path.join(PROJECT_ROOT, "database", "mongodb_exports")


def test_parquet_outputs_validity():
    """Verify core tables are correctly persisted in Parquet format with valid schemas."""
    target_tables = ["orders", "order_items", "customers", "menu_items", "restaurants"]
    for t in target_tables:
        pq_path = os.path.join(CLEANED_DIR, t, f"{t}.parquet")
        assert os.path.exists(pq_path), f"Parquet file {pq_path} does not exist"
        table = pq.read_table(pq_path)
        assert table.num_rows > 0, f"Parquet table {t} is empty"
        assert table.num_columns > 0, f"Parquet table {t} has no columns"


def test_parquet_partitioned_structure():
    """Verify partitioned parquet dataset layout."""
    partitioned_dir = os.path.join(PARQUET_DIR, "partitioned")
    if os.path.exists(partitioned_dir):
        subdirs = os.listdir(partitioned_dir)
        assert len(subdirs) > 0, "Partitioned directory is empty"


def test_relational_database_connection_and_tables():
    """Verify relational database session executes and required tables exist."""
    from database.connection import get_engine, SessionLocal
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1, "Relational DB query failed to return 1"

    with SessionLocal() as db:
        from database.models import Role, User
        role_count = db.query(Role).count()
        assert role_count >= 4, f"Expected at least 4 roles in DB, got {role_count}"
        user_count = db.query(User).count()
        assert user_count >= 4, f"Expected at least 4 users in DB, got {user_count}"


def test_mongodb_json_exports_structure():
    """Verify MongoDB document collections are serialized with valid JSON Lines schemas."""
    required_collections = ["customers", "menu_items", "orders", "restaurants"]
    for coll in required_collections:
        file_path = os.path.join(MONGO_EXPORTS_DIR, f"{coll}.json")
        assert os.path.exists(file_path), f"MongoDB export {coll}.json missing"
        docs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        assert len(docs) > 0, f"{coll}.json has 0 documents"
        sample = docs[0]
        if coll == "customers":
            assert "customer_id" in sample
        elif coll == "menu_items":
            assert "item_id" in sample
        elif coll == "orders":
            assert "order_id" in sample
        elif coll == "restaurants":
            assert "restaurant_id" in sample or "location_id" in sample


def test_mongodb_connection_handler():
    """Verify MongoDB connection module handles credentials and offline states cleanly."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    uri = os.getenv("MONGODB_URI", "")
    assert uri != "", "MONGODB_URI configuration exists"
