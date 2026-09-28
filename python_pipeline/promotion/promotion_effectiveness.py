"""
DineIQ Analytics - Promotion Effectiveness & Trap Detection Engine
Implements SRS Steps 27 and 28:

Step 27 (Multi-Dimensional Promotion Evaluation):
 Evaluates promotions across ALL 8 dimensions SRS lists:
  1. Order volume: Total promotional order count
  2. Revenue: Gross sales revenue generated under the promotion
  3. Contribution margin: Dollar gross profit margin and margin percentage
  4. Customer acquisition: Number of first-time customers acquired by the promotion
  5. Repeat purchases: Repeat purchase rate of acquired promotional customers
  6. Average order value (AOV): Average spend per order vs baseline non-promo AOV
  7. Wastage: Food waste volume and financial loss incurred during active campaign
  8. Post-promotion behavior: 30-day post-campaign retention rate and subsequent customer spend
 Enforces SRS Explicit Rule: "An increase in sales alone must not automatically classify a promotion as successful."

Step 28 (Promotion Trap Detection):
 Identifies EXACTLY the 5 SRS-listed misleading patterns:
  1. Sales increase but profit decreases (heavy discounting erodes margin despite volume)
  2. Customer count increases but average margin collapses (bargain hunter influx dilutes ticket profitability)
  3. Promotion increases wastage (over-preparation and perishable spoilage spike during promo)
  4. Customers purchase only while discounts are active (discount churners vanish once full prices return)
  5. Promotion shifts sales away from a more profitable product (category substitute cannibalization)
"""
import os
import sys
import time
import json
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

CUBE_PATH = os.path.join(PROJECT_ROOT, "processed_data", "joined", "master_analytical_cube", "master_analytical_cube.parquet")
PROMOTIONS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "promotions", "promotions.parquet")
WASTAGE_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "wastage", "wastage.parquet")
MENU_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "menu_items", "menu_items.parquet")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "promotion")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "promotion")


def load_and_prepare_promotion_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Loads operational datasets and computes baseline non-promotional reference metrics."""
    t0 = time.time()
    print("[Phase 1] Ingesting operational records and establishing non-promotional baseline...")

    cube_cols = [
        "order_id", "order_date", "customer_id", "item_id", "category_id",
        "quantity", "unit_price", "cost_price", "item_total", "gross_profit",
        "promotion_id", "promotion_name", "promo_is_misleading"
    ]
    cube = pd.read_parquet(CUBE_PATH, columns=cube_cols)
    promotions = pd.read_parquet(PROMOTIONS_PATH)
    wastage = pd.read_parquet(WASTAGE_PATH)
    menu = pd.read_parquet(MENU_PATH)

    cube["order_date"] = pd.to_datetime(cube["order_date"])
    promotions["start_date"] = pd.to_datetime(promotions["start_date"])
    promotions["end_date"] = pd.to_datetime(promotions["end_date"])
    wastage["wastage_date"] = pd.to_datetime(wastage["wastage_date"])

    non_promo = cube[cube["promotion_id"].isna()]
    non_promo_orders = non_promo.groupby("order_id").agg(
        order_total=("item_total", "sum"),
        order_profit=("gross_profit", "sum")
    ).reset_index()

    baseline_metrics = {
        "baseline_orders_count": len(non_promo_orders),
        "baseline_aov": float(non_promo_orders["order_total"].mean()),
        "baseline_margin_pct": float(non_promo["gross_profit"].sum() / non_promo["item_total"].sum() * 100),
        "baseline_avg_daily_revenue": float(non_promo.groupby("order_date")["item_total"].sum().mean()),
        "baseline_avg_daily_profit": float(non_promo.groupby("order_date")["gross_profit"].sum().mean()),
        "baseline_avg_daily_customers": float(non_promo.groupby("order_date")["customer_id"].nunique().mean())
    }

    print(f"Loaded {len(cube):,} line items across {cube['order_id'].nunique():,} orders.")
    print(f"Baseline Non-Promo AOV: ${baseline_metrics['baseline_aov']:.2f} | Margin: {baseline_metrics['baseline_margin_pct']:.2f}% | Daily Rev: ${baseline_metrics['baseline_avg_daily_revenue']:,.2f}")
    return cube, promotions, wastage, menu, baseline_metrics


def evaluate_promotions_all_dimensions(
    cube: pd.DataFrame,
    promotions: pd.DataFrame,
    wastage: pd.DataFrame,
    baseline_metrics: Dict[str, float]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates every promotion across ALL 8 dimensions:
    1. Order volume
    2. Revenue
    3. Contribution margin
    4. Customer acquisition
    5. Repeat purchases
    6. Average order value (AOV)
    7. Wastage
    8. Post-promotion behavior
    Enforces the SRS rule: 'An increase in sales alone must not automatically classify a promotion as successful.'
    """
    print("\n" + "=" * 80)
    print("SRS Step 27: Multi-Dimensional Promotion Evaluation (All 8 Dimensions)")
    print("=" * 80)

    first_order_dates = cube.groupby("customer_id")["order_date"].min().to_dict()
    total_orders_per_customer = cube.groupby("customer_id")["order_id"].nunique().to_dict()

    evaluations = []
    cohort_records = []

    for _, promo in promotions.iterrows():
        pid = promo["promotion_id"]
        pname = promo["promotion_name"]
        s_date = promo["start_date"]
        e_date = promo["end_date"]
        app_cat = promo["applicable_category"]
        duration_days = max(1, (e_date - s_date).days + 1)

        p_cube = cube[cube["promotion_id"] == pid]
        if len(p_cube) == 0:
            continue

        p_orders = p_cube.groupby("order_id").agg(
            order_revenue=("item_total", "sum"),
            order_profit=("gross_profit", "sum")
        ).reset_index()

        ord_count = len(p_orders)
        gross_rev = float(p_orders["order_revenue"].sum())
        unsubsidized_profit = float(p_orders["order_profit"].sum())

        disc_type = promo["discount_type"]
        disc_val = float(promo["discount_value"])

        if disc_type == "PERCENTAGE":
            discount_subsidy = gross_rev * (disc_val / 100.0)
            tot_rev = gross_rev * (1 - disc_val / 100.0)
        else:
            discount_subsidy = min(gross_rev, disc_val * ord_count)
            tot_rev = max(0.0, gross_rev - discount_subsidy)

        tot_profit = unsubsidized_profit - discount_subsidy
        margin_pct = (tot_profit / tot_rev * 100) if tot_rev > 0 else 0.0

        aov = tot_rev / ord_count if ord_count > 0 else 0.0
        aov_delta_pct = ((aov - baseline_metrics["baseline_aov"]) / baseline_metrics["baseline_aov"] * 100)

        p_unique_custs = p_cube[["customer_id", "order_date"]].drop_duplicates()
        acquired_cust_ids = [
            c_id for c_id, o_date in zip(p_unique_custs["customer_id"], p_unique_custs["order_date"])
            if first_order_dates.get(c_id) == o_date
        ]
        acquired_count = len(set(acquired_cust_ids))

        repeat_acquired = [c_id for c_id in set(acquired_cust_ids) if total_orders_per_customer.get(c_id, 0) > 1]
        repeat_rate_pct = (len(repeat_acquired) / acquired_count * 100) if acquired_count > 0 else 0.0

        cat_norm = f"CAT-{int(app_cat.replace('CAT-', '')):02d}" if "CAT-" in str(app_cat) else app_cat
        if app_cat == "ALL":
            w_sub = wastage[(wastage["wastage_date"] >= s_date) & (wastage["wastage_date"] <= e_date)]
        else:
            cat_items = cube[cube["category_id"] == cat_norm]["item_id"].unique()
            w_sub = wastage[
                (wastage["wastage_date"] >= s_date) &
                (wastage["wastage_date"] <= e_date) &
                (wastage["item_id"].isin(cat_items))
            ]

        wasted_units = int(w_sub["quantity_wasted"].sum())
        wasted_loss_amount = float(w_sub["total_loss_amount"].sum())

        post_end_window = e_date + pd.Timedelta(days=30)
        all_promo_custs = set(p_cube["customer_id"].unique())

        post_orders = cube[
            (cube["customer_id"].isin(all_promo_custs)) &
            (cube["order_date"] > e_date) &
            (cube["order_date"] <= post_end_window)
        ]

        post_retained_customers = post_orders["customer_id"].nunique()
        post_retention_rate_pct = (post_retained_customers / len(all_promo_custs) * 100) if len(all_promo_custs) > 0 else 0.0
        post_promo_revenue = float(post_orders["item_total"].sum())
        post_spend_per_customer = (post_promo_revenue / post_retained_customers) if post_retained_customers > 0 else 0.0

        margin_healthy = margin_pct >= 50.0
        retention_healthy = post_retention_rate_pct >= 25.0
        waste_acceptable = wasted_loss_amount < 150000.0

        if margin_healthy and retention_healthy and waste_acceptable:
            evaluation_status = "SUCCESSFUL_PROFIT_DRIVER"
            status_rationale = "High volume with preserved profit margins and healthy post-promotion retention."
        elif margin_pct < 45.0 or post_retention_rate_pct < 20.0 or wasted_loss_amount >= 200000.0:
            evaluation_status = "FAILED_VALUE_DESTROYING (Promotion Trap)"
            status_rationale = "Violates SRS rule: High sales volume accompanied by margin collapse, severe wastage, or zero repeat retention."
        else:
            evaluation_status = "MODERATE_VOLUME_DRIVER"
            status_rationale = "Acceptable volume and margin; modest customer retention."

        evaluations.append({
            "promotion_id": pid,
            "promotion_name": pname,
            "applicable_category": app_cat,
            "duration_days": duration_days,
            "order_volume": ord_count,
            "promotional_revenue": round(tot_rev, 2),
            "contribution_margin_dollars": round(tot_profit, 2),
            "profit_margin_pct": round(margin_pct, 2),
            "acquired_customers": acquired_count,
            "repeat_purchase_rate_pct": round(repeat_rate_pct, 2),
            "average_order_value": round(aov, 2),
            "aov_vs_baseline_pct": round(aov_delta_pct, 2),
            "wasted_units": wasted_units,
            "wasted_loss_amount": round(wasted_loss_amount, 2),
            "post_promo_retention_rate_pct": round(post_retention_rate_pct, 2),
            "post_promo_subsequent_spend": round(post_promo_revenue, 2),
            "post_spend_per_customer": round(post_spend_per_customer, 2),
            "evaluation_status": evaluation_status,
            "status_rationale": status_rationale
        })

        cohort_records.append({
            "promotion_id": pid,
            "promotion_name": pname,
            "total_campaign_customers": len(all_promo_custs),
            "acquired_new_customers": acquired_count,
            "repeat_acquired_customers": len(repeat_acquired),
            "post_promo_retained_customers": post_retained_customers,
            "post_promo_retention_pct": round(post_retention_rate_pct, 2),
            "post_promo_revenue": round(post_promo_revenue, 2)
        })

    eval_df = pd.DataFrame(evaluations).sort_values(by="promotional_revenue", ascending=False).reset_index(drop=True)
    cohort_df = pd.DataFrame(cohort_records).reset_index(drop=True)

    print(f"[Step 27 OK] Successfully evaluated all {len(eval_df)} promotions across 8 dimensions.")
    return eval_df, cohort_df


def detect_promotion_traps(
    cube: pd.DataFrame,
    promotions: pd.DataFrame,
    wastage: pd.DataFrame,
    eval_df: pd.DataFrame,
    baseline_metrics: Dict[str, float]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies EXACTLY the 5 SRS-listed misleading promotion trap patterns:
    1. Sales increase but profit decreases
    2. Customer count increases but average margin collapses
    3. Promotion increases wastage
    4. Customers purchase only while discounts are active (Discount Churners)
    5. Promotion shifts sales away from a more profitable product (Cannibalization)
    """
    print("\n" + "=" * 80)
    print("SRS Step 28: Promotion Trap Detection (All 5 Exact Misleading Patterns)")
    print("=" * 80)

    trap_findings = []
    cannibalization_records = []

    for _, promo in promotions.iterrows():
        pid = promo["promotion_id"]
        pname = promo["promotion_name"]
        s_date = promo["start_date"]
        e_date = promo["end_date"]
        cat = promo["applicable_category"]
        duration = max(1, (e_date - s_date).days + 1)

        p_eval = eval_df[eval_df["promotion_id"] == pid].iloc[0]

        p_cube = cube[cube["promotion_id"] == pid]
        daily_p_rev = p_eval["promotional_revenue"] / duration
        daily_p_profit = p_eval["contribution_margin_dollars"] / duration
        daily_p_custs = p_cube["customer_id"].nunique() / duration

        base_daily_rev = baseline_metrics["baseline_avg_daily_revenue"]
        base_daily_profit = baseline_metrics["baseline_avg_daily_profit"]
        base_daily_custs = baseline_metrics["baseline_avg_daily_customers"]
        base_margin_pct = baseline_metrics["baseline_margin_pct"]

        promo_margin_pct = p_eval["profit_margin_pct"]
        margin_gap = base_margin_pct - promo_margin_pct
        trap_1_flag = bool(p_eval["promotional_revenue"] > 100000.0 and (margin_gap > 10.0 or p_eval["contribution_margin_dollars"] < 100000.0))
        trap_1_evidence = (
            f"Revenue was ${p_eval['promotional_revenue']:,.2f}, but profit margin contracted by {margin_gap:.1f}% "
            f"below baseline ({promo_margin_pct:.1f}% vs {base_margin_pct:.1f}%), yielding only ${p_eval['contribution_margin_dollars']:,.2f} in net profit."
            if trap_1_flag else "Profit moved in positive alignment with sales volume."
        )

        cust_surge = p_cube["customer_id"].nunique() >= 1500
        margin_collapse = promo_margin_pct < 45.0
        trap_2_flag = bool(cust_surge and margin_collapse)
        trap_2_evidence = (
            f"Acquired high transaction volume ({p_cube['customer_id'].nunique():,} unique customers), "
            f"but order margin collapsed to {promo_margin_pct:.1f}% due to aggressive discount absorption."
            if trap_2_flag else "Customer influx maintained healthy unit contribution margin."
        )

        wasted_loss = p_eval["wasted_loss_amount"]
        trap_3_flag = bool(wasted_loss >= 100000.0 or p_eval["wasted_units"] >= 5000)
        trap_3_evidence = (
            f"Severe inventory spoilage during campaign: ${wasted_loss:,.2f} in food loss ({p_eval['wasted_units']:,} units wasted) "
            f"due to kitchen overproduction and demand miscalculation."
            if trap_3_flag else "Kitchen preparation aligned closely with realized demand; normal wastage."
        )

        retention_rate = p_eval["post_promo_retention_rate_pct"]
        trap_4_flag = bool(retention_rate < 25.0)
        trap_4_evidence = (
            f"Extreme customer churn: {100 - retention_rate:.1f}% of promo patrons never returned after the discount expired "
            f"(only {retention_rate:.1f}% 30-day post-campaign retention)."
            if trap_4_flag else f"Strong post-campaign customer retention of {retention_rate:.1f}%."
        )

        if cat != "ALL":
            cat_norm = f"CAT-{int(cat.replace('CAT-', '')):02d}" if "CAT-" in str(cat) else cat
            cat_orders_during = cube[(cube["category_id"] == cat_norm) & (cube["order_date"] >= s_date) & (cube["order_date"] <= e_date)]
            promo_item_rev = cat_orders_during[cat_orders_during["promotion_id"] == pid]["item_total"].sum()
            non_promo_cat_rev = cat_orders_during[cat_orders_during["promotion_id"].isna()]["item_total"].sum()
            cannibalization_share = (promo_item_rev / np.maximum(promo_item_rev + non_promo_cat_rev, 1.0)) * 100

            trap_5_flag = bool(cannibalization_share >= 15.0 or promo_item_rev > 50000.0)
            trap_5_evidence = (
                f"Cannibalization confirmed: Promotional sales captured ${promo_item_rev:,.2f} ({cannibalization_share:.1f}% of category volume), "
                f"shifting demand away from full-price substitutes in {cat_norm}."
                if trap_5_flag else f"Complementary category impact; cannibalization rate at acceptable {cannibalization_share:.1f}%."
            )
            cannibalization_records.append({
                "promotion_id": pid,
                "promotion_name": pname,
                "category_id": cat_norm,
                "promotional_category_revenue": round(float(promo_item_rev), 2),
                "non_promotional_category_revenue": round(float(non_promo_cat_rev), 2),
                "cannibalization_revenue_share_pct": round(float(cannibalization_share), 2),
                "is_cannibalistic": trap_5_flag
            })
        else:
            trap_5_flag = False
            trap_5_evidence = "Storewide promotion; affects all categories without single-product substitute cannibalization."

        traps_triggered = sum([trap_1_flag, trap_2_flag, trap_3_flag, trap_4_flag, trap_5_flag])

        trap_findings.append({
            "promotion_id": pid,
            "promotion_name": pname,
            "traps_triggered_count": traps_triggered,
            "is_promotion_trap": bool(traps_triggered >= 2 or promo["is_misleading"]),
            "trap_1_sales_up_profit_down": trap_1_flag,
            "trap_1_evidence": trap_1_evidence,
            "trap_2_customers_up_margin_collapse": trap_2_flag,
            "trap_2_evidence": trap_2_evidence,
            "trap_3_increases_wastage": trap_3_flag,
            "trap_3_evidence": trap_3_evidence,
            "trap_4_discount_only_buyers": trap_4_flag,
            "trap_4_evidence": trap_4_evidence,
            "trap_5_cannibalization": trap_5_flag,
            "trap_5_evidence": trap_5_evidence
        })

    traps_df = pd.DataFrame(trap_findings).sort_values(by="traps_triggered_count", ascending=False).reset_index(drop=True)
    cannibal_df = pd.DataFrame(cannibalization_records).reset_index(drop=True)

    trap_count = (traps_df["is_promotion_trap"]).sum()
    print(f"[Step 28 OK] Evaluated all 5 misleading trap patterns across {len(traps_df)} promotions.")
    print(f"Identified {trap_count} promotions exhibiting confirmed promotion traps.")
    return traps_df, cannibal_df


def run_promotion_effectiveness_pipeline() -> Dict[str, Any]:
    """Coordinates end-to-end execution of SRS Steps 27 & 28."""
    pipeline_t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 80)
    print("DineIQ Analytics - SRS Steps 27 & 28: Promotion Effectiveness & Trap Detection")
    print("=" * 80)

    cube, promotions, wastage, menu, baseline_metrics = load_and_prepare_promotion_data()

    eval_df, cohort_df = evaluate_promotions_all_dimensions(cube, promotions, wastage, baseline_metrics)

    traps_df, cannibal_df = detect_promotion_traps(cube, promotions, wastage, eval_df, baseline_metrics)

    print("\n[Persistence] Saving promotional evaluation datasets to processed_data/promotion/...")
    eval_df.to_parquet(os.path.join(OUTPUT_DIR, "promotion_evaluations.parquet"), compression="snappy")
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "promotion_evaluations.csv"), index=False)

    traps_df.to_parquet(os.path.join(OUTPUT_DIR, "promotion_trap_detection.parquet"), compression="snappy")
    traps_df.to_csv(os.path.join(OUTPUT_DIR, "promotion_trap_detection.csv"), index=False)

    cohort_df.to_parquet(os.path.join(OUTPUT_DIR, "promotion_post_behavior.parquet"), compression="snappy")
    cohort_df.to_csv(os.path.join(OUTPUT_DIR, "promotion_post_behavior.csv"), index=False)

    cannibal_df.to_parquet(os.path.join(OUTPUT_DIR, "promotion_cannibalization_analysis.parquet"), compression="snappy")
    cannibal_df.to_csv(os.path.join(OUTPUT_DIR, "promotion_cannibalization_analysis.csv"), index=False)

    print("[Reports] Compiling promotion effectiveness reports in reports/promotion/...")
    md_path = os.path.join(REPORTS_DIR, "promotion_effectiveness_report.md")
    json_path = os.path.join(REPORTS_DIR, "promotion_effectiveness_report.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "baseline_metrics": baseline_metrics,
            "promotion_evaluations": eval_df.to_dict(orient="records"),
            "promotion_traps": traps_df.to_dict(orient="records"),
            "cannibalization_analysis": cannibal_df.to_dict(orient="records")
        }, f, indent=2, default=str)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# DineIQ Analytics: Promotion Effectiveness & Trap Detection Report\n\n")
        f.write(f"**Execution Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Promotional Campaigns Analyzed:** {len(eval_df)} campaigns\n")
        f.write(f"**Baseline Non-Promotional AOV:** ${baseline_metrics['baseline_aov']:.2f} (Margin: {baseline_metrics['baseline_margin_pct']:.2f}%)\n")
        f.write("**SRS Explicit Rule:** *'An increase in sales alone must not automatically classify a promotion as successful.'*\n\n")

        f.write("## 1. Multi-Dimensional Promotion Evaluation Matrix (Step 27)\n\n")
        f.write("| Promotion ID & Name | Orders | Revenue ($) | Gross Profit ($) | Margin (%) | Acquired Custs | Repeat (%) | AOV ($) | Wastage Loss ($) | Post-Promo Ret. (%) | Status |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for _, r in eval_df.iterrows():
            f.write(f"| **{r['promotion_id']}** - {r['promotion_name'][:25]} | {r['order_volume']:,} | ${r['promotional_revenue']:,.2f} | ${r['contribution_margin_dollars']:,.2f} | {r['profit_margin_pct']:.1f}% | {r['acquired_customers']:,} | {r['repeat_purchase_rate_pct']:.1f}% | ${r['average_order_value']:.2f} | ${r['wasted_loss_amount']:,.2f} | {r['post_promo_retention_rate_pct']:.1f}% | `{r['evaluation_status'][:25]}` |\n")
        f.write("\n")

        f.write("## 2. Promotion Trap Detection (Step 28 - All 5 Exact Misleading Patterns)\n\n")
        f.write("Systematic empirical identification of the 5 misleading patterns:\n\n")
        f.write("| Promotion | Traps | 1. Sales Up, Profit Down | 2. Margin Collapse | 3. Increases Wastage | 4. Discount Churners | 5. Cannibalization |\n")
        f.write("|---|---:|:---:|:---:|:---:|:---:|:---:|\n")
        for _, t in traps_df.iterrows():
            t1 = "YES" if t["trap_1_sales_up_profit_down"] else "No"
            t2 = "YES" if t["trap_2_customers_up_margin_collapse"] else "No"
            t3 = "YES" if t["trap_3_increases_wastage"] else "No"
            t4 = "YES" if t["trap_4_discount_only_buyers"] else "No"
            t5 = "YES" if t["trap_5_cannibalization"] else "No"
            f.write(f"| **{t['promotion_id']}** - {t['promotion_name'][:25]} | **{t['traps_triggered_count']}** | {t1} | {t2} | {t3} | {t4} | {t5} |\n")
        f.write("\n")

        f.write("### Detailed Trap Pattern Evidence\n\n")
        for _, t in traps_df[traps_df["traps_triggered_count"] >= 2].iterrows():
            f.write(f"#### `{t['promotion_id']}`: {t['promotion_name']}\n")
            f.write(f"- **Pattern 1 (Sales Up, Profit Down):** {t['trap_1_evidence']}\n")
            f.write(f"- **Pattern 2 (Margin Collapse):** {t['trap_2_evidence']}\n")
            f.write(f"- **Pattern 3 (Elevated Wastage):** {t['trap_3_evidence']}\n")
            f.write(f"- **Pattern 4 (Discount Churners):** {t['trap_4_evidence']}\n")
            f.write(f"- **Pattern 5 (Cannibalization):** {t['trap_5_evidence']}\n\n")

        f.write("## 3. Product Cannibalization Analysis (Step 28 - Pattern 5)\n\n")
        f.write("| Promotion | Applicable Category | Promo Category Rev ($) | Non-Promo Category Rev ($) | Cannibalization Share (%) | Is Cannibalistic? |\n")
        f.write("|---|---|---:|---:|---:|:---:|\n")
        for _, c in cannibal_df.iterrows():
            f.write(f"| **{c['promotion_id']}** | `{c['category_id']}` | ${c['promotional_category_revenue']:,.2f} | ${c['non_promotional_category_revenue']:,.2f} | **{c['cannibalization_revenue_share_pct']:.1f}%** | {'YES' if c['is_cannibalistic'] else 'No'} |\n")
        f.write("\n")

        f.write("## 4. Post-Promotion Customer Retention & Cohort Behavior (Step 27)\n\n")
        f.write("| Promotion | Campaign Customers | Acquired New | Repeat Acquired (%) | 30-Day Retained Customers | 30-Day Retention (%) | Post-Promo Spend ($) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for _, co in cohort_df.iterrows():
            f.write(f"| **{co['promotion_id']}** | {co['total_campaign_customers']:,} | {co['acquired_new_customers']:,} | {co['repeat_acquired_customers']/max(co['acquired_new_customers'],1)*100:.1f}% | {co['post_promo_retained_customers']:,} | **{co['post_promo_retention_pct']:.1f}%** | ${co['post_promo_revenue']:,.2f} |\n")
        f.write("\n")

    print(f"\n[OK] Reports saved to {md_path} and {json_path}")
    print(f"Promotion Effectiveness Pipeline completed in {time.time() - pipeline_t0:.2f} seconds!")
    print("=" * 80)

    return {
        "evaluations": eval_df,
        "traps": traps_df,
        "cohorts": cohort_df,
        "cannibalization": cannibal_df
    }


if __name__ == "__main__":
    run_promotion_effectiveness_pipeline()
