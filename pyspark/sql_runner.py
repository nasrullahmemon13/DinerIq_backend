"""
DineIQ Analytics — PySpark SQL Execution Engine (SRS Step 6)
Loads raw/cleaned tables as TempViews and executes production Spark SQL queries.
"""
import os
import sys
import time
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CLEANED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")
QUERIES_DIR = os.path.join(CURRENT_DIR, "queries")

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from session import get_spark_session


def register_all_temp_views(spark):
    """Loads cleaned Parquet tables and registers Spark SQL Temporary Views."""
    tables = [
        "orders", "order_items", "menu_items", "menu_categories",
        "customers", "restaurants", "ratings", "wastage", "promotions"
    ]
    for tbl in tables:
        pq_path = os.path.join(CLEANED_DIR, tbl, f"{tbl}.parquet")
        if os.path.exists(pq_path):
            df = spark.read.parquet(pq_path)
            df.createOrReplaceTempView(tbl)
            print(f"  [Registered TempView] {tbl} ({df.count():,} rows)")
        else:
            print(f"  [Warning] Missing parquet table: {pq_path}")


def run_all_queries(spark):
    """Executes all 8 production SQL queries and displays preview results."""
    print("=" * 80)
    print("DineIQ Analytics — Executing Production Spark SQL Query Suite")
    print("=" * 80)

    register_all_temp_views(spark)

    sql_files = sorted([f for f in os.listdir(QUERIES_DIR) if f.endswith(".sql")])
    results = {}

    for sql_file in sql_files:
        query_path = os.path.join(QUERIES_DIR, sql_file)
        with open(query_path, "r", encoding="utf-8") as f:
            query_str = f.read()

        print(f"\n--- Executing: {sql_file} ---")
        t0 = time.time()
        df_res = spark.sql(query_str).toPandas()
        elapsed = round(time.time() - t0, 3)
        print(f"Executed in {elapsed}s | Rows Returned: {len(df_res)}")
        print(df_res.head(3))
        results[sql_file] = df_res

    return results


if __name__ == "__main__":
    sp = get_spark_session("DineIQ-PySpark-SQLRunner")
    try:
        run_all_queries(sp)
    finally:
        sp.stop()
        print("\nPySpark SQL Engine terminated cleanly.")
