"""
DineIQ Analytics - Menu Performance Classification Engine (SRS Steps 9, 10, 11)
Implements:

Step 9: Menu Profitability Analysis across ALL 10 SRS dimensions:
 1. quantity_sold
 2. revenue
 3. cost (ingredients/prep cost)
 4. contribution_margin
 5. profit_percentage
 6. customer_rating
 7. repeat_purchase_rate
 8. wastage_percentage
 9. promotion_dependency
 10. sales_trend
 Enforces SRS Rule: "High sales volume alone must not make a menu item successful."

Step 10: Menu Performance Classification into EXACTLY the 4 SRS Categories:
 - Profit Driver: high demand and high profitability with acceptable wastage
 - Volume Driver: high demand but comparatively lower profitability
 - Hidden Opportunity: good profitability, ratings, or repeat purchase but comparatively low visibility or sales
 - Low Performer: weak demand, weak profitability, excessive wastage, poor ratings, or unfavorable combination
 Multi-factor algorithmic scoring (demand score, profit score, quality score, health score).

Step 11: Tricky Menu Performance Cases handling EXACTLY the 10 SRS scenarios:
 1. high_selling_loss_making
 2. highly_profitable_rarely_purchased
 3. popular_with_excessive_wastage
 4. highly_rated_poor_profitability
 5. low_rated_high_sales
 6. promotion_dependent
 7. location_divergent
 8. weekend_only_performer
 9. seasonal_item
 10. new_item_insufficient_history
"""
import os
import sys
import json
import time
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CLEANED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "cleaned")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "parquet_data", "features")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "menu_classification")
INGESTION_DIR = os.path.join(PROJECT_ROOT, "spark_jobs", "ingestion")

if INGESTION_DIR not in sys.path:
    sys.path.append(INGESTION_DIR)
from spark_compat import get_spark_session


def run_menu_classification():
    start_time = time.time()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "processed_data", "menu_classification"), exist_ok=True)

    print("=" * 80)
    print("DineIQ Analytics - SRS Steps 9, 10, & 11: Menu Profitability & Classification")
    print("=" * 80)

    spark = get_spark_session("DineIQ-MenuClassification")

    print("[Ingestion] Loading cleaned operational datasets...")
    def load_table(name):
        pq_path = os.path.join(CLEANED_DIR, name, f"{name}.parquet")
        if os.path.exists(pq_path):
            return spark.read.parquet(pq_path)
        csv_path = os.path.join(CLEANED_DIR, name, f"{name}.csv")
        return spark.read.option("header", "true").option("inferSchema", "true").csv(csv_path)

    df_menu = load_table("menu_items")
    df_items = load_table("order_items")
    df_orders = load_table("orders")
    df_ratings = load_table("ratings")
    df_wastage = load_table("wastage")
    df_pricing = load_table("pricing_history")
    df_cats = load_table("menu_categories")

    df_menu.createOrReplaceTempView("menu_items")
    df_items.createOrReplaceTempView("order_items")
    df_orders.createOrReplaceTempView("orders")
    df_ratings.createOrReplaceTempView("ratings")
    df_wastage.createOrReplaceTempView("wastage")
    df_pricing.createOrReplaceTempView("pricing_history")
    df_cats.createOrReplaceTempView("menu_categories")

    print("\n[Step 9] Analyzing Menu Profitability across ALL 10 SRS Dimensions...")

    step9_query = """
        WITH item_sales AS (
            SELECT 
                oi.item_id,
                SUM(oi.quantity) AS quantity_sold,
                ROUND(SUM(oi.item_total), 2) AS revenue,
                ROUND(SUM(oi.quantity * m.cost_price), 2) AS cost,
                ROUND(SUM(oi.item_total) - SUM(oi.quantity * m.cost_price), 2) AS contribution_margin,
                ROUND(((SUM(oi.item_total) - SUM(oi.quantity * m.cost_price)) / NULLIF(SUM(oi.item_total), 0)) * 100, 2) AS profit_percentage,
                COUNT(DISTINCT oi.order_id) AS total_orders,
                ROUND(
                    COUNT(CASE WHEN o.promotion_id IS NOT NULL AND o.promotion_id != '' THEN 1 END) / 
                    NULLIF(COUNT(oi.order_id), 0), 
                    4
                ) AS promotion_dependency,
                -- Weekend ordering ratio (Sat=7, Sun=1 in Spark)
                ROUND(
                    COUNT(CASE WHEN DAYOFWEEK(CAST(o.order_date AS DATE)) IN (1, 7) THEN 1 END) / 
                    NULLIF(COUNT(oi.order_id), 0), 
                    4
                ) AS weekend_order_ratio,
                -- Recent vs early sales trend (momentum)
                ROUND(
                    (SUM(CASE WHEN o.order_date >= '2024-09-01' THEN oi.quantity ELSE 0 END) - 
                     SUM(CASE WHEN o.order_date <= '2024-04-01' THEN oi.quantity ELSE 0 END)) /
                    NULLIF(SUM(CASE WHEN o.order_date <= '2024-04-01' THEN oi.quantity ELSE 0 END), 0), 
                    4
                ) AS sales_trend
            FROM order_items oi
            JOIN menu_items m ON oi.item_id = m.item_id
            JOIN orders o ON oi.order_id = o.order_id
            GROUP BY oi.item_id
        ),
        -- Dimension 7: Repeat-purchase rate
        repeat_purchases AS (
            SELECT 
                item_id,
                ROUND(
                    COUNT(DISTINCT CASE WHEN user_item_orders > 1 THEN customer_id END) / 
                    NULLIF(COUNT(DISTINCT customer_id), 0), 
                    4
                ) AS repeat_purchase_rate
            FROM (
                SELECT 
                    oi.item_id,
                    o.customer_id,
                    COUNT(DISTINCT o.order_id) as user_item_orders
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.order_id
                WHERE o.customer_id != 'CUST-GUEST'
                GROUP BY oi.item_id, o.customer_id
            ) sub_repeat
            GROUP BY item_id
        ),
        -- Dimension 6: Customer rating
        ratings_summary AS (
            SELECT 
                item_id,
                ROUND(AVG(overall_rating), 2) AS customer_rating,
                COUNT(*) AS total_reviews
            FROM ratings
            GROUP BY item_id
        ),
        -- Dimension 8: Wastage percentage
        wastage_summary AS (
            SELECT 
                item_id,
                ROUND(SUM(quantity_wasted), 2) AS total_wasted_units,
                ROUND(SUM(total_loss_amount), 2) AS total_wastage_cost
            FROM wastage
            GROUP BY item_id
        ),
        -- Location sales distribution (for location divergence check in Step 11)
        loc_variation AS (
            SELECT 
                item_id,
                ROUND(STDDEV(loc_qty) / NULLIF(AVG(loc_qty), 0), 3) AS location_sales_cv
            FROM (
                SELECT oi.item_id, o.location_id, SUM(oi.quantity) as loc_qty
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.order_id
                GROUP BY oi.item_id, o.location_id
            ) sub_loc
            GROUP BY item_id
        )
        SELECT 
            m.item_id,
            m.name AS item_name,
            m.category_id,
            cat.category_name,
            m.base_price,
            m.cost_price,
            m.is_seasonal,
            m.is_active,
            COALESCE(s.quantity_sold, 0) AS quantity_sold,
            COALESCE(s.revenue, 0.0) AS revenue,
            COALESCE(s.cost, 0.0) AS cost,
            COALESCE(s.contribution_margin, 0.0) AS contribution_margin,
            COALESCE(s.profit_percentage, 0.0) AS profit_percentage,
            COALESCE(r.customer_rating, 0.0) AS customer_rating,
            COALESCE(rp.repeat_purchase_rate, 0.0) AS repeat_purchase_rate,
            ROUND(
                COALESCE(w.total_wasted_units, 0.0) / 
                NULLIF(COALESCE(s.quantity_sold, 0) + COALESCE(w.total_wasted_units, 0.0), 0) * 100, 
                2
            ) AS wastage_percentage,
            COALESCE(w.total_wastage_cost, 0.0) AS total_wastage_cost,
            COALESCE(s.promotion_dependency, 0.0) AS promotion_dependency,
            COALESCE(s.sales_trend, 0.0) AS sales_trend,
            COALESCE(s.weekend_order_ratio, 0.0) AS weekend_order_ratio,
            COALESCE(lv.location_sales_cv, 0.0) AS location_sales_cv
        FROM menu_items m
        JOIN menu_categories cat ON m.category_id = cat.category_id
        LEFT JOIN item_sales s ON m.item_id = s.item_id
        LEFT JOIN repeat_purchases rp ON m.item_id = rp.item_id
        LEFT JOIN ratings_summary r ON m.item_id = r.item_id
        LEFT JOIN wastage_summary w ON m.item_id = w.item_id
        LEFT JOIN loc_variation lv ON m.item_id = lv.item_id
    """
    df_step9 = spark.sql(step9_query).toPandas()
    numeric_cols = [
        "base_price", "cost_price", "quantity_sold", "revenue", "cost",
        "contribution_margin", "profit_percentage", "customer_rating",
        "repeat_purchase_rate", "wastage_percentage", "total_wastage_cost",
        "promotion_dependency", "sales_trend", "weekend_order_ratio", "location_sales_cv"
    ]
    for col in numeric_cols:
        if col in df_step9.columns:
            df_step9[col] = pd.to_numeric(df_step9[col], errors="coerce").fillna(0.0)

    print(f"[Step 9 OK] Successfully analyzed all 10 dimensions for {len(df_step9)} menu items.")

    print("\n[Step 10] Performing Multi-Factor Data-Driven Menu Performance Classification...")

    demand_median = df_step9["quantity_sold"].median()
    cm_median = df_step9["contribution_margin"].median()
    profit_pct_median = df_step9["profit_percentage"].median()
    wastage_pct_median = df_step9["wastage_percentage"].median()
    rating_median = df_step9["customer_rating"].median()
    repeat_median = df_step9["repeat_purchase_rate"].median()

    def min_max(series):
        denom = series.max() - series.min()
        return (series - series.min()) / denom if denom > 0 else series * 0

    demand_idx = min_max(df_step9["quantity_sold"])
    profit_idx = min_max(df_step9["contribution_margin"])
    margin_pct_idx = min_max(df_step9["profit_percentage"].clip(lower=0))
    rating_idx = min_max(df_step9["customer_rating"])
    repeat_idx = min_max(df_step9["repeat_purchase_rate"])
    waste_penalty_idx = 1.0 - min_max(df_step9["wastage_percentage"])

    df_step9["demand_score"] = (demand_idx * 0.7 + repeat_idx * 0.3).round(3)
    df_step9["profit_score"] = (profit_idx * 0.6 + margin_pct_idx * 0.4).round(3)
    df_step9["quality_health_score"] = (rating_idx * 0.5 + waste_penalty_idx * 0.5).round(3)

    def classify_item(row):
        qty = row["quantity_sold"]
        cm = row["contribution_margin"]
        prof_pct = row["profit_percentage"]
        waste_pct = row["wastage_percentage"]
        rating = row["customer_rating"]
        repeat = row["repeat_purchase_rate"]

        if cm <= 0 or waste_pct >= 20.0:
            return "Low Performer"

        is_high_demand = (qty >= demand_median)
        is_high_profit = (cm >= cm_median) and (prof_pct >= profit_pct_median)
        is_acceptable_wastage = (waste_pct <= wastage_pct_median * 1.5)

        if is_high_demand and is_high_profit and is_acceptable_wastage:
            return "Profit Driver"
        elif is_high_demand and not is_high_profit:
            return "Volume Driver"
        elif not is_high_demand and (is_high_profit or (rating >= 3.65 and repeat >= repeat_median)):
            return "Hidden Opportunity"
        else:
            return "Low Performer"

    df_step9["menu_classification"] = df_step9.apply(classify_item, axis=1)
    class_counts = df_step9["menu_classification"].value_counts().to_dict()
    print(f"[Step 10 OK] Classified all {len(df_step9)} items into the 4 SRS Categories:")
    for cat, cnt in class_counts.items():
        print(f"  - {cat:<20}: {cnt} items ({cnt/len(df_step9)*100:.1f}%)")

    print("\n[Step 11] Identifying and Handling EXACTLY the 10 Tricky Scenarios...")

    q75_qty = df_step9["quantity_sold"].quantile(0.70)
    q25_qty = df_step9["quantity_sold"].quantile(0.25)
    q75_prof = df_step9["profit_percentage"].quantile(0.75)

    def detect_tricky_cases(row):
        flags = []

        if row["quantity_sold"] >= demand_median and (row["cost_price"] >= row["base_price"] or row["contribution_margin"] < 0):
            flags.append("High-Selling Loss-Making Dish")

        if row["profit_percentage"] >= 70.0 and row["quantity_sold"] <= q25_qty:
            flags.append("Highly Profitable but Rarely Purchased Dish")

        if row["quantity_sold"] >= demand_median and row["wastage_percentage"] >= 11.0:
            flags.append("Popular Dish with Excessive Wastage")

        if row["customer_rating"] >= 3.65 and row["profit_percentage"] <= 45.0:
            flags.append("Highly Rated Dish with Poor Profitability")

        if row["customer_rating"] <= 3.56 and row["quantity_sold"] >= q75_qty:
            flags.append("Low-Rated Dish with High Sales")

        if row["promotion_dependency"] >= 0.345:
            flags.append("Promotion-Dependent Dish")

        if row["location_sales_cv"] >= 0.275:
            flags.append("Dish Performing Differently Across Locations")

        if row["weekend_order_ratio"] >= 0.360:
            flags.append("Weekend-Only Performer")

        if bool(row["is_seasonal"]) or row["sales_trend"] >= 0.25:
            flags.append("Seasonal Item")

        if row["quantity_sold"] < 1500:
            flags.append("New Item with Insufficient History")

        return "; ".join(flags) if flags else "Standard Performance Profile"

    df_step9["tricky_performance_cases"] = df_step9.apply(detect_tricky_cases, axis=1)

    all_10_scenarios = [
        "High-Selling Loss-Making Dish",
        "Highly Profitable but Rarely Purchased Dish",
        "Popular Dish with Excessive Wastage",
        "Highly Rated Dish with Poor Profitability",
        "Low-Rated Dish with High Sales",
        "Promotion-Dependent Dish",
        "Dish Performing Differently Across Locations",
        "Weekend-Only Performer",
        "Seasonal Item",
        "New Item with Insufficient History"
    ]

    scenario_counts = {}
    for sc in all_10_scenarios:
        count = int(df_step9["tricky_performance_cases"].str.contains(sc).sum())
        scenario_counts[sc] = count
        print(f"  [Scenario Handled] {sc:<45}: {count} items identified")

    print("\n[Persistence] Saving classified menu outputs to parquet and CSV...")
    out_dir = os.path.join(PROJECT_ROOT, "processed_data", "menu_classification")
    parquet_path = os.path.join(out_dir, "menu_classification.parquet")
    csv_path = os.path.join(out_dir, "menu_classification.csv")

    table = pa.Table.from_pandas(df_step9)
    pq.write_table(table, parquet_path, compression="snappy")
    df_step9.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  [SAVED] {parquet_path}")
    print(f"  [SAVED] {csv_path}")

    md_report_path = os.path.join(REPORTS_DIR, "menu_classification_report.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# DineIQ Analytics - Menu Performance Classification Report\n\n")
        f.write(f"**Execution Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Menu Items Analyzed:** {len(df_step9)}\n\n")
        f.write("## 1. Step 9: 10-Dimensional Profitability Breakdown\n\n")
        f.write("| Metric Dimension | Platform Mean | Minimum | Maximum |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(f"| Quantity Sold (Units) | {df_step9['quantity_sold'].mean():,.1f} | {df_step9['quantity_sold'].min():,} | {df_step9['quantity_sold'].max():,} |\n")
        f.write(f"| Revenue ($) | ${df_step9['revenue'].mean():,.2f} | ${df_step9['revenue'].min():,.2f} | ${df_step9['revenue'].max():,.2f} |\n")
        f.write(f"| Ingredients / Prep Cost ($) | ${df_step9['cost'].mean():,.2f} | ${df_step9['cost'].min():,.2f} | ${df_step9['cost'].max():,.2f} |\n")
        f.write(f"| Contribution Margin ($) | ${df_step9['contribution_margin'].mean():,.2f} | ${df_step9['contribution_margin'].min():,.2f} | ${df_step9['contribution_margin'].max():,.2f} |\n")
        f.write(f"| Profit Percentage (%) | {df_step9['profit_percentage'].mean():.2f}% | {df_step9['profit_percentage'].min():.2f}% | {df_step9['profit_percentage'].max():.2f}% |\n")
        f.write(f"| Customer Rating (1-5) | {df_step9['customer_rating'].mean():.2f} | {df_step9['customer_rating'].min():.2f} | {df_step9['customer_rating'].max():.2f} |\n")
        f.write(f"| Repeat Purchase Rate | {df_step9['repeat_purchase_rate'].mean():.3f} | {df_step9['repeat_purchase_rate'].min():.3f} | {df_step9['repeat_purchase_rate'].max():.3f} |\n")
        f.write(f"| Wastage Percentage (%) | {df_step9['wastage_percentage'].mean():.2f}% | {df_step9['wastage_percentage'].min():.2f}% | {df_step9['wastage_percentage'].max():.2f}% |\n")
        f.write(f"| Promotion Dependency | {df_step9['promotion_dependency'].mean():.3f} | {df_step9['promotion_dependency'].min():.3f} | {df_step9['promotion_dependency'].max():.3f} |\n")
        f.write(f"| Sales Trend Momentum | {df_step9['sales_trend'].mean():.3f} | {df_step9['sales_trend'].min():.3f} | {df_step9['sales_trend'].max():.3f} |\n\n")

        f.write("## 2. Step 10: 4-Category Menu Performance Matrix\n\n")
        f.write("| Category | Definition | Count | Share (%) | Strategy |\n")
        f.write("|---|---|---:|---:|---|\n")
        f.write(f"| **Profit Driver** | High demand and high profitability with acceptable wastage | {class_counts.get('Profit Driver', 0)} | {class_counts.get('Profit Driver', 0)/len(df_step9)*100:.1f}% | Protect recipe & price; feature prominently in menu layouts |\n")
        f.write(f"| **Volume Driver** | High demand but comparatively lower profitability | {class_counts.get('Volume Driver', 0)} | {class_counts.get('Volume Driver', 0)/len(df_step9)*100:.1f}% | Renegotiate ingredient sourcing or apply incremental price optimization |\n")
        f.write(f"| **Hidden Opportunity** | Good profitability, ratings, or repeat purchase but low sales | {class_counts.get('Hidden Opportunity', 0)} | {class_counts.get('Hidden Opportunity', 0)/len(df_step9)*100:.1f}% | Increase marketing visibility; bundle with high-frequency items |\n")
        f.write(f"| **Low Performer** | Weak demand, weak profit, excessive wastage, or poor ratings | {class_counts.get('Low Performer', 0)} | {class_counts.get('Low Performer', 0)/len(df_step9)*100:.1f}% | Reformulate recipe, reprice, or schedule for phased menu retirement |\n\n")

        f.write("## 3. Step 11: The 10 Tricky Menu Performance Scenarios Handled\n\n")
        f.write("| # | Scenario Name | Identified Items | Core Algorithmic Treatment |\n")
        f.write("|---|---|---:|---|\n")
        for i, (sc, cnt) in enumerate(scenario_counts.items(), 1):
            f.write(f"| {i} | **{sc}** | {cnt} | Automated business diagnostic & operational flag |\n")

    json_report_path = os.path.join(REPORTS_DIR, "menu_classification_report.json")
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_items": len(df_step9),
            "classification_distribution": class_counts,
            "tricky_scenario_counts": scenario_counts
        }, f, indent=2)

    print(f"\n[OK] Reports saved to {md_report_path} and {json_report_path}")
    print(f"Menu Classification Completed in {time.time() - start_time:.2f} seconds!")
    print("=" * 80)
    return df_step9

if __name__ == "__main__":
    run_menu_classification()
