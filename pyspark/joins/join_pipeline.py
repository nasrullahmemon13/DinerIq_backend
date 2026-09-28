"""
DineIQ Analytics - Data Integration & Join Pipeline (SRS Step 6)
Implements EXACTLY the 10 Relationships listed in SRS:
 1. Orders - Customers
 2. Orders - OrderItems
 3. OrderItems - MenuItems
 4. MenuItems - Categories
 5. Orders - Locations
 6. Orders - Promotions
 7. MenuItems - PricingHistory
 8. MenuItems - Ratings
 9. MenuItems - Inventory
10. MenuItems - Wastage

Demonstrates:
- PySpark DataFrame API joins (inner, left, left_anti)
- Spark SQL declarative relational queries with registered TempViews
- Integrated Master Analytical Order Cube export
- Columnar Parquet & CSV persistence in processed_data/joined/
"""
import os
import sys
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
CLEANED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")
JOINED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data", "joined")
INGESTION_DIR = os.path.join(PROJECT_ROOT, "spark_jobs", "ingestion")

sys.path.append(INGESTION_DIR)
from spark_compat import get_spark_session

def save_joined_view(df, name: str):
    """Save joined output in both CSV and Snappy-compressed Parquet."""
    out_dir = os.path.join(JOINED_DATA_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    pdf = df.toPandas() if hasattr(df, "toPandas") else df

    csv_path = os.path.join(out_dir, f"{name}.csv")
    parquet_path = os.path.join(out_dir, f"{name}.parquet")

    pdf.to_csv(csv_path, index=False, encoding="utf-8")
    table = pa.Table.from_pandas(pdf)
    pq.write_table(table, parquet_path, compression="snappy")
    print(f"  [SAVED] '{name}': {len(pdf):,} rows -> CSV & Parquet")

def execute_join_pipeline():
    start_total = time.time()
    os.makedirs(JOINED_DATA_DIR, exist_ok=True)

    print("=" * 80)
    print("DineIQ Analytics - SRS Step 6: Spark SQL & PySpark Relational Joins")
    print("=" * 80)

    spark = get_spark_session("DineIQ-JoinPipeline")

    print("\n[Phase 1] Ingesting cleaned datasets from processed_data/cleaned/...")
    t0 = time.time()
    def load_table(name):
        pq_path = os.path.join(CLEANED_DATA_DIR, name, f"{name}.parquet")
        csv_path = os.path.join(CLEANED_DATA_DIR, name, f"{name}.csv")
        if os.path.exists(pq_path):
            return spark.read.parquet(pq_path)
        else:
            return spark.read.option("header", "true").option("inferSchema", "true").csv(csv_path)

    df_orders = load_table("orders")
    df_items = load_table("order_items")
    df_cust = load_table("customers")
    df_menu = load_table("menu_items")
    df_rest = load_table("restaurants")
    df_cats = load_table("menu_categories")
    df_promo = load_table("promotions")
    df_price = load_table("pricing_history")
    df_ratings = load_table("ratings")
    df_inv = load_table("inventory")
    df_waste = load_table("wastage")

    print("Registering Spark SQL TempViews...")
    df_orders.createOrReplaceTempView("orders")
    df_items.createOrReplaceTempView("order_items")
    df_cust.createOrReplaceTempView("customers")
    df_menu.createOrReplaceTempView("menu_items")
    df_rest.createOrReplaceTempView("restaurants")
    df_cats.createOrReplaceTempView("menu_categories")
    df_promo.createOrReplaceTempView("promotions")
    df_price.createOrReplaceTempView("pricing_history")
    df_ratings.createOrReplaceTempView("ratings")
    df_inv.createOrReplaceTempView("inventory")
    df_waste.createOrReplaceTempView("wastage")
    print(f"[OK] Ingested and registered 11 TempViews in {time.time() - t0:.2f}s")

    join_metrics = []

    print("\n" + "=" * 80)
    print("EXECUTING THE EXACT 10 SRS RELATIONSHIPS")
    print("=" * 80)

    print("\n[Relationship 1/10] Orders - Customers")
    t_start = time.perf_counter()
    join_1 = spark.sql("""
        SELECT 
            o.order_id,
            o.customer_id,
            c.customer_segment,
            c.loyalty_tier,
            c.loyalty_points,
            c.churn_risk_score,
            o.order_date,
            o.total_amount,
            o.order_status
        FROM orders o
        LEFT JOIN customers c ON o.customer_id = c.customer_id
    """)
    elapsed_1 = time.perf_counter() - t_start
    c_1 = join_1.count()
    print(f"  Joined: {c_1:,} rows in {elapsed_1:.3f}s")
    join_metrics.append(("Orders - Customers", "LEFT JOIN", "customer_id", c_1, elapsed_1))

    print("\n[Relationship 2/10] Orders - OrderItems")
    t_start = time.perf_counter()
    join_2 = spark.sql("""
        SELECT 
            o.order_id,
            o.location_id,
            o.order_date,
            o.order_time,
            oi.order_item_id,
            oi.item_id,
            oi.quantity,
            oi.unit_price,
            oi.item_total
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
    """)
    elapsed_2 = time.perf_counter() - t_start
    c_2 = join_2.count()
    print(f"  Joined: {c_2:,} rows in {elapsed_2:.3f}s")
    join_metrics.append(("Orders - OrderItems", "INNER JOIN", "order_id", c_2, elapsed_2))

    print("\n[Relationship 3/10] OrderItems - MenuItems")
    t_start = time.perf_counter()
    join_3 = spark.sql("""
        SELECT 
            oi.order_item_id,
            oi.order_id,
            oi.item_id,
            m.name AS item_name,
            m.category_id,
            m.base_price,
            m.cost_price,
            m.margin_pct,
            m.complexity_profile,
            oi.quantity,
            oi.item_total
        FROM order_items oi
        JOIN menu_items m ON oi.item_id = m.item_id
    """)
    elapsed_3 = time.perf_counter() - t_start
    c_3 = join_3.count()
    print(f"  Joined: {c_3:,} rows in {elapsed_3:.3f}s")
    join_metrics.append(("OrderItems - MenuItems", "INNER JOIN", "item_id", c_3, elapsed_3))

    print("\n[Relationship 4/10] MenuItems - Categories")
    t_start = time.perf_counter()
    join_4 = spark.sql("""
        SELECT 
            m.item_id,
            m.name AS item_name,
            m.category_id,
            c.category_name,
            c.target_margin_pct,
            m.base_price,
            m.cost_price,
            m.margin_pct
        FROM menu_items m
        JOIN menu_categories c ON m.category_id = c.category_id
    """)
    elapsed_4 = time.perf_counter() - t_start
    c_4 = join_4.count()
    print(f"  Joined: {c_4:,} rows in {elapsed_4:.3f}s")
    join_metrics.append(("MenuItems - Categories", "INNER JOIN", "category_id", c_4, elapsed_4))

    print("\n[Relationship 5/10] Orders - Locations")
    t_start = time.perf_counter()
    join_5 = spark.sql("""
        SELECT 
            o.order_id,
            o.order_date,
            o.total_amount,
            r.location_id,
            r.name AS restaurant_name,
            r.city,
            r.state,
            r.location_tier,
            r.cost_index
        FROM orders o
        JOIN restaurants r ON o.location_id = r.location_id
    """)
    elapsed_5 = time.perf_counter() - t_start
    c_5 = join_5.count()
    print(f"  Joined: {c_5:,} rows in {elapsed_5:.3f}s")
    join_metrics.append(("Orders - Locations", "INNER JOIN", "location_id", c_5, elapsed_5))

    print("\n[Relationship 6/10] Orders - Promotions")
    t_start = time.perf_counter()
    join_6 = spark.sql("""
        SELECT 
            o.order_id,
            o.order_date,
            o.subtotal_amount,
            o.discount_amount,
            o.total_amount,
            p.promotion_id,
            p.promotion_name,
            p.discount_type,
            p.discount_value,
            p.is_misleading
        FROM orders o
        LEFT JOIN promotions p ON o.promotion_id = p.promotion_id
    """)
    elapsed_6 = time.perf_counter() - t_start
    c_6 = join_6.count()
    print(f"  Joined: {c_6:,} rows in {elapsed_6:.3f}s")
    join_metrics.append(("Orders - Promotions", "LEFT JOIN", "promotion_id", c_6, elapsed_6))

    print("\n[Relationship 7/10] MenuItems - PricingHistory")
    t_start = time.perf_counter()
    join_7 = spark.sql("""
        SELECT 
            m.item_id,
            m.name AS item_name,
            m.base_price AS current_base_price,
            ph.price_history_id,
            ph.location_id,
            ph.base_price AS historical_price,
            ph.cost_price AS historical_cost,
            ph.effective_start_date,
            ph.effective_end_date,
            ph.change_reason
        FROM menu_items m
        JOIN pricing_history ph ON m.item_id = ph.item_id
    """)
    elapsed_7 = time.perf_counter() - t_start
    c_7 = join_7.count()
    print(f"  Joined: {c_7:,} rows in {elapsed_7:.3f}s")
    join_metrics.append(("MenuItems - PricingHistory", "INNER JOIN", "item_id", c_7, elapsed_7))

    print("\n[Relationship 8/10] MenuItems - Ratings")
    t_start = time.perf_counter()
    join_8 = spark.sql("""
        SELECT 
            m.item_id,
            m.name AS item_name,
            m.category_id,
            r.rating_id,
            r.overall_rating,
            r.food_rating,
            r.service_rating,
            r.review_text,
            r.review_date
        FROM menu_items m
        JOIN ratings r ON m.item_id = r.item_id
    """)
    elapsed_8 = time.perf_counter() - t_start
    c_8 = join_8.count()
    print(f"  Joined: {c_8:,} rows in {elapsed_8:.3f}s")
    join_metrics.append(("MenuItems - Ratings", "INNER JOIN", "item_id", c_8, elapsed_8))

    print("\n[Relationship 9/10] MenuItems - Inventory")
    t_start = time.perf_counter()
    join_9 = spark.sql("""
        SELECT 
            m.item_id,
            m.name AS item_name,
            inv.inventory_id,
            inv.location_id,
            inv.snapshot_date,
            inv.starting_stock,
            inv.quantity_received,
            inv.quantity_sold,
            inv.quantity_wasted,
            inv.ending_stock,
            inv.stock_status
        FROM menu_items m
        JOIN inventory inv ON m.item_id = inv.item_id
    """)
    elapsed_9 = time.perf_counter() - t_start
    c_9 = join_9.count()
    print(f"  Joined: {c_9:,} rows in {elapsed_9:.3f}s")
    join_metrics.append(("MenuItems - Inventory", "INNER JOIN", "item_id", c_9, elapsed_9))

    print("\n[Relationship 10/10] MenuItems - Wastage")
    t_start = time.perf_counter()
    join_10 = spark.sql("""
        SELECT 
            m.item_id,
            m.name AS item_name,
            m.complexity_profile,
            w.wastage_id,
            w.location_id,
            w.wastage_date,
            w.quantity_wasted,
            w.unit_cost,
            w.total_loss_amount,
            w.wastage_reason
        FROM menu_items m
        JOIN wastage w ON m.item_id = w.item_id
    """)
    elapsed_10 = time.perf_counter() - t_start
    c_10 = join_10.count()
    print(f"  Joined: {c_10:,} rows in {elapsed_10:.3f}s")
    join_metrics.append(("MenuItems - Wastage", "INNER JOIN", "item_id", c_10, elapsed_10))

    print("\n" + "=" * 80)
    print("BUILDING MASTER DENORMALIZED ANALYTICAL ORDER CUBE")
    print("=" * 80)
    t_cube = time.perf_counter()
    master_cube = spark.sql("""
        SELECT 
            o.order_id,
            o.order_date,
            o.order_time,
            o.order_type,
            o.payment_method,
            o.order_status,
            o.customer_id,
            c.customer_segment,
            c.loyalty_tier,
            c.churn_risk_score,
            o.location_id,
            r.name AS restaurant_name,
            r.city AS restaurant_city,
            r.state AS restaurant_state,
            r.location_tier,
            r.cost_index,
            oi.order_item_id,
            oi.item_id,
            m.name AS item_name,
            m.category_id,
            cat.category_name,
            m.complexity_profile,
            oi.quantity,
            oi.unit_price,
            m.cost_price,
            (oi.quantity * m.cost_price) AS total_item_cost,
            oi.subtotal AS item_subtotal,
            oi.item_total,
            (oi.item_total - (oi.quantity * m.cost_price)) AS gross_profit,
            o.promotion_id,
            p.promotion_name,
            p.is_misleading AS promo_is_misleading
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN menu_items m ON oi.item_id = m.item_id
        JOIN menu_categories cat ON m.category_id = cat.category_id
        JOIN restaurants r ON o.location_id = r.location_id
        LEFT JOIN customers c ON o.customer_id = c.customer_id
        LEFT JOIN promotions p ON o.promotion_id = p.promotion_id
    """)
    elapsed_cube = time.perf_counter() - t_cube
    print(f"Master Analytical Cube created: {master_cube.count():,} rows in {elapsed_cube:.2f}s")
    save_joined_view(master_cube, "master_analytical_cube")

    save_joined_view(join_1.limit(5000), "orders_customers_sample")
    save_joined_view(join_10.limit(5000), "menu_wastage_sample")

    print("\n" + "=" * 80)
    print("SRS STEP 6 — ALL 10 RELATIONSHIPS EXECUTION SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame(join_metrics, columns=["Relationship", "Join Type", "Key", "Joined Rows", "Execution Time (s)"])
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print(f"Entire Join Pipeline Completed in {time.time() - start_total:.2f} seconds!")
    print(f"Output Destination: {JOINED_DATA_DIR}")
    print("=" * 80)
    return summary_df

if __name__ == "__main__":
    execute_join_pipeline()
