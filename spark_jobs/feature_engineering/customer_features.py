"""
DineIQ Analytics - Customer Behavioral Feature Engineering (SRS Step 7)
Computes customer-level behavioral and preference features specified in SRS:
 1. basket_size: Average number of item units per order for the customer
 2. discount_percentage: Average percentage discount applied to customer's orders
 3. promotion_dependency: Ratio of orders utilizing a promotion code vs total orders
 4. peak_hour_frequency: Proportion of customer's orders placed during lunch/dinner peak hours
 5. weekend_order_ratio: Ratio of orders placed on Saturday/Sunday vs total orders
 6. channel_preference: Dominant dining channel (Dine-in, Takeout, Delivery)
 7. location_performance: Relative performance / spend index at customer's primary restaurant
"""
import os
import sys
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
CLEANED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")
INGESTION_DIR = os.path.join(PROJECT_ROOT, "spark_jobs", "ingestion")

if INGESTION_DIR not in sys.path:
    sys.path.append(INGESTION_DIR)
from spark_compat import get_spark_session


def compute_customer_features(spark=None) -> pd.DataFrame:
    """
    Computes all 7 Customer Behavioral features using Spark SQL / PySpark DataFrames.
    Returns a pandas DataFrame with one row per customer.
    """
    if spark is None:
        spark = get_spark_session("DineIQ-CustomerFeatures")

    print("[CustomerFeatures] Loading cleaned datasets into Spark...")

    def load_clean(name):
        pq = os.path.join(CLEANED_DIR, name, f"{name}.parquet")
        if os.path.exists(pq):
            return spark.read.parquet(pq)
        csv = os.path.join(CLEANED_DIR, name, f"{name}.csv")
        return spark.read.option("header", "true").option("inferSchema", "true").csv(csv)

    df_customers = load_clean("customers")
    df_orders = load_clean("orders")
    df_items = load_clean("order_items")
    df_restaurants = load_clean("restaurants")

    df_customers.createOrReplaceTempView("customers")
    df_orders.createOrReplaceTempView("orders")
    df_items.createOrReplaceTempView("order_items")
    df_restaurants.createOrReplaceTempView("restaurants")

    print("[CustomerFeatures] Computing per-order basket sizes...")
    order_basket_df = spark.sql("""
        SELECT 
            order_id,
            SUM(quantity) AS order_total_quantity,
            COUNT(DISTINCT item_id) AS distinct_item_count
        FROM order_items
        GROUP BY order_id
    """)
    order_basket_df.createOrReplaceTempView("order_basket_sizes")

    print("[CustomerFeatures] Computing customer temporal, basket, discount, and promotion patterns...")
    cust_metrics_df = spark.sql("""
        SELECT 
            o.customer_id,
            ROUND(AVG(COALESCE(b.order_total_quantity, 1)), 2) AS basket_size,
            ROUND(AVG((o.discount_amount / NULLIF(o.subtotal_amount, 0)) * 100), 2) AS discount_percentage,
            ROUND(
                COUNT(CASE WHEN o.promotion_id IS NOT NULL AND o.promotion_id != '' THEN 1 END) / 
                NULLIF(COUNT(o.order_id), 0), 
                4
            ) AS promotion_dependency,
            ROUND(
                COUNT(CASE 
                    WHEN (HOUR(CAST(o.order_time AS STRING)) BETWEEN 12 AND 14) OR 
                         (HOUR(CAST(o.order_time AS STRING)) BETWEEN 18 AND 21) THEN 1 
                END) / NULLIF(COUNT(o.order_id), 0), 
                4
            ) AS peak_hour_frequency,
            ROUND(
                COUNT(CASE 
                    WHEN DAYOFWEEK(CAST(o.order_date AS DATE)) IN (1, 7) THEN 1 
                END) / NULLIF(COUNT(o.order_id), 0), 
                4
            ) AS weekend_order_ratio
        FROM orders o
        LEFT JOIN order_basket_sizes b ON o.order_id = b.order_id
        WHERE o.customer_id != 'CUST-GUEST'
        GROUP BY o.customer_id
    """)
    cust_metrics_df.createOrReplaceTempView("cust_metrics")

    print("[CustomerFeatures] Determining dominant channel preference per customer...")
    channel_df = spark.sql("""
        WITH channel_counts AS (
            SELECT 
                customer_id,
                order_type,
                COUNT(*) as cnt,
                ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY COUNT(*) DESC) as rank
            FROM orders
            WHERE customer_id != 'CUST-GUEST'
            GROUP BY customer_id, order_type
        )
        SELECT customer_id, order_type AS channel_preference
        FROM channel_counts
        WHERE rank = 1
    """)
    channel_df.createOrReplaceTempView("channel_metrics")

    print("[CustomerFeatures] Computing preferred location performance index...")
    loc_df = spark.sql("""
        WITH cust_pref_loc AS (
            SELECT 
                customer_id,
                location_id,
                COUNT(*) as visit_count,
                ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY COUNT(*) DESC) as rk
            FROM orders
            WHERE customer_id != 'CUST-GUEST'
            GROUP BY customer_id, location_id
        ),
        loc_totals AS (
            SELECT 
                location_id,
                COUNT(order_id) as total_loc_orders,
                ROUND(AVG(total_amount), 2) as loc_avg_spend
            FROM orders
            GROUP BY location_id
        )
        SELECT 
            c.customer_id,
            c.location_id AS primary_location_id,
            ROUND(l.loc_avg_spend, 2) AS location_performance
        FROM cust_pref_loc c
        JOIN loc_totals l ON c.location_id = l.location_id
        WHERE c.rk = 1
    """)
    loc_df.createOrReplaceTempView("location_metrics")

    print("[CustomerFeatures] Assembling unified Customer Feature Mart...")
    final_spark_df = spark.sql("""
        SELECT 
            c.customer_id,
            c.customer_segment,
            c.loyalty_tier,
            c.loyalty_points,
            c.churn_risk_score,
            COALESCE(m.basket_size, 0.0) AS basket_size,
            COALESCE(m.discount_percentage, 0.0) AS discount_percentage,
            COALESCE(m.promotion_dependency, 0.0) AS promotion_dependency,
            COALESCE(m.peak_hour_frequency, 0.0) AS peak_hour_frequency,
            COALESCE(m.weekend_order_ratio, 0.0) AS weekend_order_ratio,
            COALESCE(ch.channel_preference, 'Dine-in') AS channel_preference,
            COALESCE(loc.primary_location_id, c.preferred_location_id) AS primary_location_id,
            COALESCE(loc.location_performance, 0.0) AS location_performance
        FROM customers c
        LEFT JOIN cust_metrics m ON c.customer_id = m.customer_id
        LEFT JOIN channel_metrics ch ON c.customer_id = ch.customer_id
        LEFT JOIN location_metrics loc ON c.customer_id = loc.customer_id
        WHERE c.customer_id != 'CUST-GUEST'
    """)

    pdf = final_spark_df.toPandas()
    pdf["basket_size"] = pdf["basket_size"].fillna(1.0)
    pdf["discount_percentage"] = pdf["discount_percentage"].fillna(0.0)
    pdf["promotion_dependency"] = pdf["promotion_dependency"].fillna(0.0)
    pdf["peak_hour_frequency"] = pdf["peak_hour_frequency"].fillna(0.0)
    pdf["weekend_order_ratio"] = pdf["weekend_order_ratio"].fillna(0.0)
    pdf["location_performance"] = pdf["location_performance"].fillna(0.0)
    print(f"[CustomerFeatures] Successfully calculated 7 customer features for {len(pdf):,} customers!")
    return pdf

if __name__ == "__main__":
    df = compute_customer_features()
    print(df.head(10).to_string())
