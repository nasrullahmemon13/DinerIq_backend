"""
DineIQ Analytics - NoSQL Database Loader (MongoDB / PyMongo)
Per SRS Section 2 & Step 2 requirements:
- Ingests structured and nested document models into MongoDB collections:
  * restaurants, menu_categories, menu_items, customers
  * orders (hierarchical nested documents with embedded items array)
  * order_items, ratings, wastage, inventory, promotions, pricing_history
  * customer_segments, menu_classification, quarantine_manifest, model_evidence, strategic_recommendations
- Supports live Atlas connection or exports mongoimport-ready JSON documents.
"""
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from mongodb_manager import sync_all_data_to_mongodb, check_mongo_connection


def load_nosql_database(max_orders: int = 10000):
    """
    Executes complete NoSQL database ingestion and JSON Line export for all entities.
    """
    return sync_all_data_to_mongodb(order_limit=max_orders)


if __name__ == "__main__":
    load_nosql_database()
