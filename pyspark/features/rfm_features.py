"""
DineIQ Analytics - RFM Customer Value Feature Engineering (SRS Step 7)
Computes the exact RFM and monetary features specified in SRS:
 1. customer_recency: Days elapsed since customer's most recent completed order
 2. customer_frequency: Lifetime count of orders placed by customer
 3. customer_monetary_value: Lifetime cumulative spend ($) of customer
 4. average_order_value (AOV): Mean spend per transaction (Monetary / Frequency)
 5. rfm_score: Composite RFM tier score (e.g., Champions, Loyal, At-Risk, Hibernating)
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


def compute_rfm_features(spark=None) -> pd.DataFrame:
    """
    Computes Customer Recency, Frequency, Monetary Value, AOV, and RFM segmentation.
    Returns a pandas DataFrame with one row per customer.
    """
    if spark is None:
        spark = get_spark_session("DineIQ-RFMFeatures")

    print("[RFMFeatures] Loading cleaned customers and orders into Spark...")

    def load_clean(name):
        pq = os.path.join(CLEANED_DIR, name, f"{name}.parquet")
        if os.path.exists(pq):
            return spark.read.parquet(pq)
        csv = os.path.join(CLEANED_DIR, name, f"{name}.csv")
        return spark.read.option("header", "true").option("inferSchema", "true").csv(csv)

    df_customers = load_clean("customers")
    df_orders = load_clean("orders")

    df_customers.createOrReplaceTempView("customers")
    df_orders.createOrReplaceTempView("orders")

    print("[RFMFeatures] Extracting reference cutoff date and customer aggregates...")
    rfm_raw_df = spark.sql("""
        WITH max_date AS (
            SELECT MAX(CAST(order_date AS DATE)) AS max_order_date FROM orders
        ),
        cust_aggregates AS (
            SELECT 
                o.customer_id,
                MAX(CAST(o.order_date AS DATE)) AS last_order_date,
                COUNT(DISTINCT o.order_id) AS customer_frequency,
                ROUND(SUM(o.total_amount), 2) AS customer_monetary_value,
                ROUND(AVG(o.total_amount), 2) AS average_order_value
            FROM orders o
            WHERE o.customer_id != 'CUST-GUEST'
            GROUP BY o.customer_id
        )
        SELECT 
            c.customer_id,
            COALESCE(DATEDIFF(m.max_order_date, ca.last_order_date), 365) AS customer_recency,
            COALESCE(ca.customer_frequency, 0) AS customer_frequency,
            COALESCE(ca.customer_monetary_value, 0.0) AS customer_monetary_value,
            COALESCE(ca.average_order_value, 0.0) AS average_order_value,
            ca.last_order_date
        FROM customers c
        CROSS JOIN max_date m
        LEFT JOIN cust_aggregates ca ON c.customer_id = ca.customer_id
        WHERE c.customer_id != 'CUST-GUEST'
    """)

    pdf = rfm_raw_df.toPandas()

    print("[RFMFeatures] Computing statistical RFM quintiles and business segments...")
    r_labels = [5, 4, 3, 2, 1]
    f_labels = [1, 2, 3, 4, 5]
    m_labels = [1, 2, 3, 4, 5]

    try:
        pdf["r_score"] = pd.qcut(pdf["customer_recency"], q=5, labels=r_labels, duplicates="drop").astype(int)
    except Exception:
        pdf["r_score"] = 3

    try:
        pdf["f_score"] = pd.qcut(pdf["customer_frequency"].rank(method="first"), q=5, labels=f_labels).astype(int)
    except Exception:
        pdf["f_score"] = 3

    try:
        pdf["m_score"] = pd.qcut(pdf["customer_monetary_value"].rank(method="first"), q=5, labels=m_labels).astype(int)
    except Exception:
        pdf["m_score"] = 3

    pdf["rfm_combined_score"] = pdf["r_score"].astype(str) + pdf["f_score"].astype(str) + pdf["m_score"].astype(str)

    def assign_segment(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "Recent First-Time"
        elif r <= 2 and f >= 3:
            return "At Risk - High Value"
        elif r <= 2 and f <= 2:
            return "Hibernating / Churned"
        else:
            return "Potential Loyalists"

    pdf["rfm_segment"] = pdf.apply(assign_segment, axis=1)

    print(f"[RFMFeatures] Successfully generated RFM features for {len(pdf):,} customers!")
    return pdf

if __name__ == "__main__":
    df = compute_rfm_features()
    print(df.head(10).to_string())
