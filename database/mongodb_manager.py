"""
DineIQ Analytics — Enterprise MongoDB & NoSQL Database Manager (SRS Step 2)
Integrates Apache Spark / Big Data outputs into MongoDB Atlas / local MongoDB:
1. Operational Collections (11 Tables):
   - restaurants, menu_categories, menu_items, customers, orders (hierarchical nested),
   - order_items, ratings, wastage, inventory, promotions, pricing_history
2. Analytical Intelligence Collections:
   - customer_segments (RFM scores & 6 segments)
   - menu_classification (BCG matrix 4 categories & 10 tricky flags)
   - quarantine_manifest (audit records from data cleaning)
   - model_evidence (benchmarks & champion model metadata)
   - strategic_recommendations (ROI-ranked prescriptions)
3. Full Indexing on query keys (order_id, customer_id, location_id, item_id, order_date)
4. Offline / Online Dual Persistence: Always writes mongoimport-ready JSON exports,
   and uploads directly to MongoDB Atlas when credentials are configured.
"""
import os
import sys
import json
import time
from typing import Dict, Any, List, Optional
import pandas as pd
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
CLEANED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
MONGO_EXPORT_DIR = os.path.join(CURRENT_DIR, "mongodb_exports")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def get_mongo_client(uri: Optional[str] = None, timeout_ms: int = 15000):
    """
    Returns an authenticated MongoClient with certifi SSL context,
    ServerApi('1'), and generous timeouts for Atlas Free Tier cluster wake-up.
    """
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi
    try:
        import certifi
        ca_file = certifi.where()
    except ImportError:
        ca_file = None

    connection_uri = uri or os.getenv("MONGODB_URI", "")
    if not connection_uri:
        raise ValueError("MONGODB_URI is not set in environment or .env file.")

    kwargs = {
        "server_api": ServerApi('1'),
        "serverSelectionTimeoutMS": timeout_ms,
        "connectTimeoutMS": timeout_ms,
        "socketTimeoutMS": timeout_ms * 2
    }
    if ca_file:
        kwargs["tlsCAFile"] = ca_file

    return MongoClient(connection_uri, **kwargs)


def check_mongo_connection() -> Dict[str, Any]:
    """
    Checks if MongoDB Atlas connection is configured and responsive.
    """
    uri = os.getenv("MONGODB_URI", "")
    db_name = os.getenv("MONGODB_DATABASE", "dineiq_analytics")

    if not uri:
        return {"status": "unconfigured", "message": "MONGODB_URI is not set in .env"}

    if "<db_password>" in uri:
        return {
            "status": "pending_credentials",
            "message": "MONGODB_URI contains '<db_password>' placeholder. Please replace it with your actual Atlas password.",
            "uri_masked": uri.split("@")[-1] if "@" in uri else uri
        }

    try:
        client = get_mongo_client(uri, timeout_ms=4000)
        client.admin.command("ping")
        server_info = client.server_info()
        db = client[db_name]
        colls = db.list_collection_names()
        return {
            "status": "connected",
            "database": db_name,
            "server_version": server_info.get("version", "N/A"),
            "existing_collections": colls,
            "cluster_host": uri.split("@")[-1].split("/")[0] if "@" in uri else "local"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e),
            "cluster_host": uri.split("@")[-1].split("/")[0] if "@" in uri else "local"
        }


def read_table_dataframe(table_name: str) -> pd.DataFrame:
    """Reads table from processed_data/cleaned parquet if available, else raw CSV."""
    pq_path = os.path.join(CLEANED_DIR, table_name, f"{table_name}.parquet")
    if os.path.exists(pq_path):
        return pd.read_parquet(pq_path)

    clean_csv = os.path.join(CLEANED_DIR, table_name, f"{table_name}.csv")
    if os.path.exists(clean_csv):
        return pd.read_csv(clean_csv)

    raw_csv = os.path.join(RAW_DATA_DIR, table_name, f"{table_name}.csv")
    if os.path.exists(raw_csv):
        return pd.read_csv(raw_csv)

    return pd.DataFrame()


def prepare_nested_orders(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Builds MongoDB NoSQL hierarchical documents:
    Orders with embedded line items array for fast single-document retrieval.
    """
    print("  [NoSQL Aggregator] Building hierarchical nested orders with embedded items...")
    df_orders = read_table_dataframe("orders")
    df_items = read_table_dataframe("order_items")

    if limit and limit > 0:
        df_orders = df_orders.head(limit)
        order_ids = set(df_orders["order_id"])
        df_items = df_items[df_items["order_id"].isin(order_ids)]

    items_by_order: Dict[str, List[Dict[str, Any]]] = {}
    for item in df_items.to_dict(orient="records"):
        o_id = str(item.get("order_id"))
        clean_item = {k: (None if pd.isna(v) else v) for k, v in item.items()}
        items_by_order.setdefault(o_id, []).append(clean_item)

    order_docs = []
    for order in df_orders.to_dict(orient="records"):
        o_id = str(order.get("order_id"))
        doc = {k: (None if pd.isna(v) else v) for k, v in order.items()}
        doc["items"] = items_by_order.get(o_id, [])
        doc["item_count"] = len(doc["items"])
        order_docs.append(doc)

    print(f"  [NoSQL Aggregator] Successfully assembled {len(order_docs):,} nested order documents.")
    return order_docs


def export_collection_json(collection_name: str, records: List[Dict[str, Any]]):
    """Exports records to newline-delimited JSON for offline mongoimport."""
    os.makedirs(MONGO_EXPORT_DIR, exist_ok=True)
    out_file = os.path.join(MONGO_EXPORT_DIR, f"{collection_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"  [JSON Export] {collection_name:<22}: {len(records):>8,} docs -> {out_file} ({size_mb:.2f} MB)")


def upload_batch_to_mongo(db, collection_name: str, records: List[Dict[str, Any]], batch_size: int = 5000):
    """Uploads records to MongoDB in chunks to respect network limits."""
    col = db[collection_name]
    col.delete_many({})                                   

    total = len(records)
    inserted = 0
    for i in range(0, total, batch_size):
        chunk = records[i:i + batch_size]
        col.insert_many(chunk, ordered=False)
        inserted += len(chunk)
    print(f"  [MongoDB Upload] {collection_name:<20}: {inserted:>8,} docs committed to database.")


def create_mongodb_indexes(db):
    """Creates indexes on common query fields for maximum read performance."""
    print("\n--- Creating MongoDB Indexes ---")
    try:
        db.orders.create_index("order_id", unique=True)
        db.orders.create_index([("location_id", 1), ("order_date", -1)])
        db.orders.create_index("customer_id")
        db.orders.create_index("order_status")

        db.order_items.create_index("order_item_id", unique=True)
        db.order_items.create_index([("order_id", 1), ("item_id", 1)])

        db.menu_items.create_index("item_id", unique=True)
        db.menu_items.create_index("category_id")

        db.customers.create_index("customer_id", unique=True)
        db.customers.create_index("customer_segment")

        db.restaurants.create_index("location_id", unique=True)
        db.ratings.create_index([("item_id", 1), ("overall_rating", -1)])
        db.wastage.create_index([("location_id", 1), ("wastage_date", -1)])

        print("  [OK] Successfully established indexes on Orders, OrderItems, Customers, Menu, Ratings, and Wastage.")
    except Exception as e:
        print(f"  [Index Notice] Could not create all indexes: {e}")


def sync_all_data_to_mongodb(
    uri: Optional[str] = None,
    order_limit: Optional[int] = 10000,
    force_upload: bool = True
) -> Dict[str, Any]:
    """
    Main orchestration routine:
    1. Loads all 11 operational entities
    2. Loads analytical deliverables (segments, menu matrix, benchmarks, quarantine)
    3. Exports all collections to JSON files in database/mongodb_exports/
    4. If MongoDB is connected, performs live upsert into MongoDB Atlas.
    """
    start_time = time.time()
    print("=" * 80)
    print("DineIQ Analytics — Complete MongoDB & NoSQL Database Synchronization")
    print("=" * 80)

    conn_status = check_mongo_connection()
    is_live = (conn_status.get("status") == "connected")
    db = None
    if is_live:
        client = get_mongo_client(uri)
        db_name = os.getenv("MONGODB_DATABASE", "dineiq_analytics")
        db = client[db_name]
        print(f"[LIVE MODE] Connected to MongoDB Atlas cluster: {conn_status.get('cluster_host')}")
        print(f"Target Database: '{db_name}'")
    else:
        print(f"[OFFLINE/EXPORT MODE] {conn_status.get('message')}")
        print(f"Generating full mongoimport-ready JSON collections in: {MONGO_EXPORT_DIR}")

    sync_summary = {}

    print("\n--- Phase 1: Operational Platform Entities ---")
    operational_tables = [
        "restaurants", "menu_categories", "menu_items", "customers",
        "order_items", "ratings", "wastage", "inventory", "promotions", "pricing_history"
    ]

    for tbl in operational_tables:
        df = read_table_dataframe(tbl)
        records = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in df.to_dict(orient="records")]
        export_collection_json(tbl, records)
        if is_live and db is not None:
            upload_batch_to_mongo(db, tbl, records)
        sync_summary[tbl] = len(records)

    print("\n--- Phase 2: Hierarchical Nested Orders Document Model ---")
    nested_orders = prepare_nested_orders(limit=order_limit)
    export_collection_json("orders_nested", nested_orders)
    if is_live and db is not None:
        upload_batch_to_mongo(db, "orders_nested", nested_orders)
    sync_summary["orders_nested"] = len(nested_orders)

    df_orders = read_table_dataframe("orders")
    if order_limit and order_limit > 0:
        df_orders = df_orders.head(order_limit)
    orders_flat = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in df_orders.to_dict(orient="records")]
    export_collection_json("orders", orders_flat)
    if is_live and db is not None:
        upload_batch_to_mongo(db, "orders", orders_flat)
    sync_summary["orders"] = len(orders_flat)

    print("\n--- Phase 3: Analytical Intelligence & ML Deliverables ---")

    seg_pq = os.path.join(PROJECT_ROOT, "processed_data", "customer_segmentation", "customer_segments.parquet")
    if os.path.exists(seg_pq):
        df_seg = pd.read_parquet(seg_pq)
        records_seg = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in df_seg.to_dict(orient="records")]
        export_collection_json("customer_segments", records_seg)
        if is_live and db is not None:
            upload_batch_to_mongo(db, "customer_segments", records_seg)
        sync_summary["customer_segments"] = len(records_seg)

    menu_cls_pq = os.path.join(PROJECT_ROOT, "processed_data", "menu_classification", "menu_classification.parquet")
    if os.path.exists(menu_cls_pq):
        df_mc = pd.read_parquet(menu_cls_pq)
        records_mc = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in df_mc.to_dict(orient="records")]
        export_collection_json("menu_classification", records_mc)
        if is_live and db is not None:
            upload_batch_to_mongo(db, "menu_classification", records_mc)
        sync_summary["menu_classification"] = len(records_mc)

    q_manifest_file = os.path.join(PROJECT_ROOT, "processed_data", "quarantine", "quarantine_manifest.json")
    if os.path.exists(q_manifest_file):
        with open(q_manifest_file, "r", encoding="utf-8") as f:
            q_data = json.load(f)
        q_records = q_data.get("batches", []) if isinstance(q_data, dict) else q_data
        export_collection_json("quarantine_manifest", q_records)
        if is_live and db is not None:
            upload_batch_to_mongo(db, "quarantine_manifest", q_records)
        sync_summary["quarantine_manifest"] = len(q_records)

    rec_file = os.path.join(REPORTS_DIR, "recommendations", "recommendation_engine_summary.json")
    if os.path.exists(rec_file):
        with open(rec_file, "r", encoding="utf-8") as f:
            rec_data = json.load(f)
        rec_records = rec_data.get("category_distribution", []) if isinstance(rec_data, dict) else rec_data
        export_collection_json("strategic_recommendations", rec_records)
        if is_live and db is not None:
            upload_batch_to_mongo(db, "strategic_recommendations", rec_records)
        sync_summary["strategic_recommendations"] = len(rec_records)

    evidence_file = os.path.join(REPORTS_DIR, "model_comparison", "spark_model_evidence.json")
    if os.path.exists(evidence_file):
        with open(evidence_file, "r", encoding="utf-8") as f:
            ev_data = json.load(f)
        ev_records = ev_data.get("classification_benchmarks", [])
        export_collection_json("model_evidence", ev_records)
        if is_live and db is not None:
            upload_batch_to_mongo(db, "model_evidence", ev_records)
        sync_summary["model_evidence"] = len(ev_records)

    if is_live and db is not None:
        create_mongodb_indexes(db)

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"MongoDB Synchronization Completed in {elapsed} seconds!")
    print(f"Total Collections Prepared: {len(sync_summary)}")
    total_docs = sum(sync_summary.values())
    print(f"Total Documents: {total_docs:,}")
    print(f"Local Exports Directory: {MONGO_EXPORT_DIR}")
    print("=" * 80)

    return {
        "status": "success",
        "is_live_atlas": is_live,
        "elapsed_seconds": elapsed,
        "total_documents": total_docs,
        "collections": sync_summary
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DineIQ Analytics MongoDB Sync Manager")
    parser.add_argument("--limit-orders", type=int, default=10000, help="Order limit for nested documents")
    parser.add_argument("--test-only", action="store_true", help="Only test connection")
    args = parser.parse_args()

    if args.test_only:
        res = check_mongo_connection()
        print(json.dumps(res, indent=2))
    else:
        sync_all_data_to_mongodb(order_limit=args.limit_orders)
