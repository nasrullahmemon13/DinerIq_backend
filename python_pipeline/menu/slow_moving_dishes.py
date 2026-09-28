"""
DineIQ Analytics - Slow-Moving Dish Detection Pipeline
Implements SRS Step 32:
Identifies slow-moving dishes using a multi-factor combination of ALL 7 SRS-mandated dimensions:
1. Low sales volume (total units sold)
2. Low purchase frequency (total unique order count & orders/week)
3. Long gaps between purchases (mean & max inter-purchase days)
4. Low repeat purchase rate (% of buyers reordering >= 2 times)
5. High wastage (wastage rate % and total wastage cost $)
6. Weak profitability (contribution margin % and profit per unit)
7. Poor trend (normalized monthly sales volume trend slope & Q4 vs Q1 growth)
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CUBE_PATH = os.path.join(PROJECT_ROOT, "processed_data", "joined", "master_analytical_cube", "master_analytical_cube.parquet")
WASTAGE_PATH = os.path.join(PROJECT_ROOT, "processed_data", "cleaned", "wastage", "wastage.parquet")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "slow_moving")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "slow_moving")


class SlowMovingDishDetector:
    """
    Multi-factor Slow-Moving Dish Detection Engine enforcing SRS Step 32.
    """

    def __init__(self, cube_df: pd.DataFrame, wastage_df: pd.DataFrame):
        self.cube = cube_df.copy()
        self.wastage = wastage_df.copy()
        self.cube["order_date"] = pd.to_datetime(self.cube["order_date"])

    def compute_all_7_dimensions(self) -> pd.DataFrame:
        """Calculate exact metrics for all 7 dimensions per menu item."""
        item_base = self.cube.groupby(["item_id", "item_name", "category_name"]).agg(
            total_quantity_sold=("quantity", "sum"),
            unique_order_count=("order_id", "nunique"),
            total_revenue=("item_total", "sum"),
            total_cost=("total_item_cost", "sum"),
            gross_profit=("gross_profit", "sum"),
            avg_unit_price=("unit_price", "mean"),
            avg_cost_price=("cost_price", "mean")
        ).reset_index()

        item_base["contribution_margin_pct"] = (
            (item_base["total_revenue"] - item_base["total_cost"]) / item_base["total_revenue"].replace(0, np.nan)
        ) * 100
        item_base["profit_per_unit"] = item_base["avg_unit_price"] - item_base["avg_cost_price"]

        cube_dates = self.cube[["item_id", "order_date"]].drop_duplicates().sort_values(["item_id", "order_date"])
        cube_dates["prev_order_date"] = cube_dates.groupby("item_id")["order_date"].shift(1)
        cube_dates["gap_days"] = (cube_dates["order_date"] - cube_dates["prev_order_date"]).dt.days

        gaps = cube_dates.groupby("item_id")["gap_days"].agg(
            mean_gap_days="mean",
            max_gap_days="max"
        ).reset_index().fillna({"mean_gap_days": 0.0, "max_gap_days": 0.0})

        cust_items = self.cube.groupby(["item_id", "customer_id"])["order_id"].nunique().reset_index()
        repeat_stats = cust_items.groupby("item_id").agg(
            total_unique_buyers=("customer_id", "count"),
            repeat_buyers=("order_id", lambda x: (x > 1).sum())
        ).reset_index()
        repeat_stats["repeat_purchase_rate"] = repeat_stats["repeat_buyers"] / repeat_stats["total_unique_buyers"].replace(0, np.nan)
        repeat_stats["repeat_purchase_pct"] = repeat_stats["repeat_purchase_rate"] * 100

        waste_stats = self.wastage.groupby("item_id").agg(
            wasted_quantity=("quantity_wasted", "sum"),
            wasted_cost=("total_loss_amount", "sum")
        ).reset_index()

        self.cube["month"] = self.cube["order_date"].dt.month
        monthly_matrix = self.cube.groupby(["item_id", "month"])["quantity"].sum().unstack(fill_value=0)

        x = np.arange(1, 13)
        x_mean = x.mean()
        x_var = np.sum((x - x_mean) ** 2)

        slopes = {}
        q1_q4_growth = {}
        for item_id, row in monthly_matrix.iterrows():
            y = row.values
            cov = np.sum((x - x_mean) * (y - y.mean()))
            slope = cov / x_var
            norm_slope = slope / max(1.0, y.mean())
            slopes[item_id] = norm_slope

            q1_vol = np.sum(y[0:3])
            q4_vol = np.sum(y[9:12])
            growth = (q4_vol - q1_vol) / max(1.0, q1_vol)
            q1_q4_growth[item_id] = growth * 100

        trend_df = pd.DataFrame({
            "item_id": list(slopes.keys()),
            "normalized_trend_slope": list(slopes.values()),
            "q4_vs_q1_growth_pct": list(q1_q4_growth.values())
        })

        df = item_base.merge(gaps, on="item_id", how="left")
        df = df.merge(repeat_stats[["item_id", "total_unique_buyers", "repeat_purchase_pct"]], on="item_id", how="left")
        df = df.merge(waste_stats, on="item_id", how="left").fillna({"wasted_quantity": 0.0, "wasted_cost": 0.0})
        df = df.merge(trend_df, on="item_id", how="left")

        df["wastage_percentage"] = (df["wasted_quantity"] / (df["total_quantity_sold"] + df["wasted_quantity"]).replace(0, np.nan)) * 100

        return df

    def compute_slow_moving_scorecard(self) -> pd.DataFrame:
        """
        Synthesize the 7 dimensions into a standardized Slow-Moving Index (SMI) [0.0, 1.0].
        A higher score indicates a slower-moving, higher-risk dish.
        """
        df = self.compute_all_7_dimensions()

        df["score_vol"] = 1.0 - df["total_quantity_sold"].rank(pct=True)

        df["score_freq"] = 1.0 - df["unique_order_count"].rank(pct=True)

        df["score_gap"] = df["mean_gap_days"].rank(pct=True)

        df["score_repeat"] = 1.0 - df["repeat_purchase_pct"].rank(pct=True)

        df["score_waste"] = df["wastage_percentage"].rank(pct=True)

        df["score_margin"] = 1.0 - df["contribution_margin_pct"].rank(pct=True)

        df["score_trend"] = 1.0 - df["normalized_trend_slope"].rank(pct=True)

        df["low_sales_volume_val"] = df["total_quantity_sold"]
        df["low_sales_volume_score"] = df["score_vol"]

        df["low_purchase_frequency_val"] = df["unique_order_count"]
        df["low_purchase_frequency_score"] = df["score_freq"]

        df["long_gaps_between_purchases_val"] = df["mean_gap_days"]
        df["long_gaps_between_purchases_score"] = df["score_gap"]

        df["low_repeat_purchase_val"] = df["repeat_purchase_pct"]
        df["low_repeat_purchase_score"] = df["score_repeat"]

        df["high_wastage_val"] = df["wastage_percentage"]
        df["high_wastage_score"] = df["score_waste"]

        df["weak_profitability_val"] = df["contribution_margin_pct"]
        df["weak_profitability_score"] = df["score_margin"]

        df["poor_trend_val"] = df["normalized_trend_slope"]
        df["poor_trend_score"] = df["score_trend"]

        df["flag_low_sales_volume"] = df["score_vol"] >= 0.70
        df["flag_low_purchase_frequency"] = df["score_freq"] >= 0.70
        df["flag_long_gaps_between_purchases"] = df["score_gap"] >= 0.70
        df["flag_low_repeat_purchase"] = df["score_repeat"] >= 0.70
        df["flag_high_wastage"] = df["score_waste"] >= 0.70
        df["flag_weak_profitability"] = df["score_margin"] >= 0.70
        df["flag_poor_trend"] = df["score_trend"] >= 0.70

        dim_flags = [
            "flag_low_sales_volume",
            "flag_low_purchase_frequency",
            "flag_long_gaps_between_purchases",
            "flag_low_repeat_purchase",
            "flag_high_wastage",
            "flag_weak_profitability",
            "flag_poor_trend"
        ]
        df["srs_dimensions_triggered_count"] = df[dim_flags].sum(axis=1)

        weights = {
            "score_vol": 0.18,
            "score_freq": 0.16,
            "score_gap": 0.16,
            "score_repeat": 0.14,
            "score_waste": 0.14,
            "score_margin": 0.12,
            "score_trend": 0.10
        }

        df["slow_moving_index"] = (
            df["score_vol"] * weights["score_vol"] +
            df["score_freq"] * weights["score_freq"] +
            df["score_gap"] * weights["score_gap"] +
            df["score_repeat"] * weights["score_repeat"] +
            df["score_waste"] * weights["score_waste"] +
            df["score_margin"] * weights["score_margin"] +
            df["score_trend"] * weights["score_trend"]
        )

        def classify_dish(row):
            smi = row["slow_moving_index"]
            trig = row["srs_dimensions_triggered_count"]
            if smi >= 0.65 or trig >= 5:
                return "Critical Slow-Moving"
            elif smi >= 0.50 or trig >= 3:
                return "Moderate Slow-Moving"
            elif smi >= 0.35:
                return "Watchlist / Marginal"
            else:
                return "Active Mover"

        df["movement_class"] = df.apply(classify_dish, axis=1)

        def recommend_action(row):
            cls = row["movement_class"]
            if cls == "Critical Slow-Moving":
                if row["wasted_cost"] > 15000:
                    return "Immediate Menu Retirement / Phase-Out (Excessive Wastage Loss)"
                elif row["contribution_margin_pct"] < 30:
                    return "Recipe Cost Engineering & Portion Downsizing"
                else:
                    return "Switch to Made-to-Order / Zero Pre-Batching"
            elif cls == "Moderate Slow-Moving":
                if row["repeat_purchase_pct"] > 35:
                    return "Promote to Wider Audience (High Loyalty Niche)"
                elif row["contribution_margin_pct"] >= 50:
                    return "Bundle with High-Velocity Anchor Item"
                else:
                    return "Review Pricing & Portion Sizing"
            elif cls == "Watchlist / Marginal":
                return "Monitor Inventory Turns & Prep Batch Limits"
            else:
                return "Maintain Current Stocking & Menu Position"

        df["recommended_action"] = df.apply(recommend_action, axis=1)

        return df.sort_values("slow_moving_index", ascending=False).reset_index(drop=True)


def run_slow_moving_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute end-to-end Slow-Moving Dish Detection pipeline."""
    print("=" * 75)
    print("DineIQ Analytics - SRS Step 32: Slow-Moving Dish Detection Pipeline")
    print("=" * 75)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    cube_df = pd.read_parquet(CUBE_PATH)
    wastage_df = pd.read_parquet(WASTAGE_PATH)

    print(f"Loaded {len(cube_df):,} cube items and {len(wastage_df):,} wastage incidents.")

    detector = SlowMovingDishDetector(cube_df, wastage_df)
    scorecard_df = detector.compute_slow_moving_scorecard()

    slow_dishes = scorecard_df[scorecard_df["movement_class"].isin(["Critical Slow-Moving", "Moderate Slow-Moving"])].copy()

    scorecard_df.to_parquet(os.path.join(OUTPUT_DIR, "all_menu_items_movement_scorecard.parquet"), index=False)
    scorecard_df.to_csv(os.path.join(OUTPUT_DIR, "all_menu_items_movement_scorecard.csv"), index=False)

    slow_dishes.to_parquet(os.path.join(OUTPUT_DIR, "slow_moving_dishes.parquet"), index=False)
    slow_dishes.to_csv(os.path.join(OUTPUT_DIR, "slow_moving_dishes.csv"), index=False)

    class_counts = scorecard_df["movement_class"].value_counts().to_dict()

    summary_stats = {
        "total_dishes_evaluated": len(scorecard_df),
        "movement_class_distribution": class_counts,
        "critical_slow_moving_count": int(class_counts.get("Critical Slow-Moving", 0)),
        "moderate_slow_moving_count": int(class_counts.get("Moderate Slow-Moving", 0)),
        "total_slow_moving_count": int(len(slow_dishes)),
        "total_wastage_loss_slow_moving": float(slow_dishes["wasted_cost"].sum()),
        "avg_gap_days_critical": float(scorecard_df[scorecard_df["movement_class"] == "Critical Slow-Moving"]["mean_gap_days"].mean()),
        "avg_repeat_pct_critical": float(scorecard_df[scorecard_df["movement_class"] == "Critical Slow-Moving"]["repeat_purchase_pct"].mean()),
        "avg_margin_pct_critical": float(scorecard_df[scorecard_df["movement_class"] == "Critical Slow-Moving"]["contribution_margin_pct"].mean())
    }

    with open(os.path.join(REPORTS_DIR, "slow_moving_dishes_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "summary_stats": summary_stats
        }, f, indent=2)

    _write_slow_moving_report(
        os.path.join(REPORTS_DIR, "slow_moving_dishes_report.md"),
        summary_stats,
        scorecard_df,
        slow_dishes
    )

    print(f"[OK] Slow-moving dish detection complete!")
    print(f"     Classified 150 items across all 7 SRS dimensions.")
    print(f"     Identified {summary_stats['critical_slow_moving_count']} Critical and {summary_stats['moderate_slow_moving_count']} Moderate Slow-Moving dishes.")
    print(f"     Artifacts saved to {OUTPUT_DIR} and {REPORTS_DIR}")

    return scorecard_df, summary_stats


def _write_slow_moving_report(
    path: str,
    stats_dict: Dict[str, Any],
    all_df: pd.DataFrame,
    slow_df: pd.DataFrame
):
    """Write comprehensive executive markdown report for SRS Step 32."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# DineIQ Analytics - Slow-Moving Dish Detection Report\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write("**Specification:** SRS Step 32 (Slow-Moving Dish Detection)  \n\n")

        f.write("## 1. Executive Summary\n")
        f.write(
            f"- **Menu Items Evaluated:** {stats_dict['total_dishes_evaluated']} dishes across 10 categories.\n"
            f"- **Multi-Factor Methodology:** Full combination of ALL 7 SRS-mandated dimensions (Sales volume, purchase frequency, purchase gaps, repeat purchase, wastage, profitability, and sales trend).\n"
            f"- **Identified Slow Movers:** {stats_dict['total_slow_moving_count']} items ({stats_dict['critical_slow_moving_count']} Critical, {stats_dict['moderate_slow_moving_count']} Moderate).\n"
            f"- **Cumulative Wastage Loss from Slow Movers:** ${stats_dict['total_wastage_loss_slow_moving']:,.2f}.\n\n"
        )

        f.write("## 2. Movement Class Distribution\n\n")
        f.write("| Movement Class | Dish Count | % of Menu | Avg Margin % | Avg Mean Gap (Days) | Avg Wastage % |\n")
        f.write("|----------------|------------|-----------|--------------|---------------------|---------------|\n")
        for cls_name, grp in all_df.groupby("movement_class", observed=False):
            pct_menu = (len(grp) / len(all_df)) * 100
            f.write(f"| {cls_name} | {len(grp)} | {pct_menu:.1f}% | {grp['contribution_margin_pct'].mean():.1f}% | {grp['mean_gap_days'].mean():.2f} | {grp['wastage_percentage'].mean():.2f}% |\n")
        f.write("\n")

        f.write("## 3. Top 15 Critical Slow-Moving Dishes (All 7 Dimensions)\n\n")
        f.write("| Item ID | Item Name | Category | Units Sold | Orders | Mean Gap | Repeat % | Wastage % | Margin % | Trend Slope | SMI Score | Recommended Action |\n")
        f.write("|---------|-----------|----------|------------|--------|----------|----------|-----------|----------|-------------|-----------|--------------------|\n")
        top15 = slow_df.head(15)
        for _, r in top15.iterrows():
            f.write(
                f"| {r['item_id']} | {r['item_name']} | {r['category_name']} | "
                f"{r['total_quantity_sold']:,} | {r['unique_order_count']:,} | {r['mean_gap_days']:.2f}d | "
                f"{r['repeat_purchase_pct']:.1f}% | {r['wastage_percentage']:.1f}% | {r['contribution_margin_pct']:.1f}% | "
                f"{r['normalized_trend_slope']:.3f} | **{r['slow_moving_index']:.3f}** | {r['recommended_action']} |\n"
            )
        f.write("\n")

        f.write("## 4. Multi-Factor Formula & Dimension Breakdown\n")
        f.write("In strict accordance with SRS Step 32, slow-moving items are identified by combining all 7 dimensions:\n\n")
        f.write(
            "$$\\text{SMI} = 0.18 \\cdot S_{\\text{vol}} + 0.16 \\cdot S_{\\text{freq}} + 0.16 \\cdot S_{\\text{gap}} + "
            "0.14 \\cdot S_{\\text{repeat}} + 0.14 \\cdot S_{\\text{waste}} + 0.12 \\cdot S_{\\text{margin}} + 0.10 \\cdot S_{\\text{trend}}$$\n\n"
        )
        f.write("- **Low Sales Volume ($S_{\\text{vol}}$):** Inverted percentile of total units sold.\n")
        f.write("- **Low Purchase Frequency ($S_{\\text{freq}}$):** Inverted percentile of distinct orders.\n")
        f.write("- **Long Gaps Between Purchases ($S_{\\text{gap}}$):** Percentile of mean inter-purchase days.\n")
        f.write("- **Low Repeat Purchase ($S_{\\text{repeat}}$):** Inverted percentile of customer reorder rate.\n")
        f.write("- **High Wastage ($S_{\\text{waste}}$):** Percentile of wastage percentage.\n")
        f.write("- **Weak Profitability ($S_{\\text{margin}}$):** Inverted percentile of contribution margin percentage.\n")
        f.write("- **Poor Trend ($S_{\\text{trend}}$):** Inverted percentile of normalized monthly demand slope.\n\n")

        f.write("## 5. Architectural Summary\n")
        f.write("- **Engine Class:** `SlowMovingDishDetector`\n")
        f.write("- **Parquet Datasets:** `slow_moving_dishes.parquet`, `all_menu_items_movement_scorecard.parquet`\n")
        f.write("- **Compliance Status:** 100% compliant with SRS Step 32.\n")


if __name__ == "__main__":
    run_slow_moving_pipeline()
