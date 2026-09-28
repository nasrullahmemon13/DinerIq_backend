"""
DineIQ Analytics - What-If Scenario Analysis & Scenario Impact Engine
Implements SRS Steps 40 and 41:

Step 40: What-If Scenario Analysis
Users must be able to simulate selected scenarios such as:
1. Increase menu price
2. Reduce item price
3. Change discount percentage
4. Increase promotion frequency
5. Remove a menu item
6. Reduce preparation quantity
7. Increase predicted demand
8. Change wastage assumptions

Step 41: Scenario Impact Analysis
The application should display the estimated impact on selected indicators such as:
1. Revenue
2. Contribution margin
3. Demand
4. Wastage
5. Profitability

MANDATORY SRS Rule (Step 41):
"The application must clearly identify simulated outputs as estimates rather than actual results."
Every simulated output explicitly carries `is_simulation_estimate: True` and the standard disclaimer:
"ESTIMATE ONLY: Simulated outputs represent theoretical projections generated via econometric
and behavioral models and must not be interpreted as actual historical results."
"""

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MENU_CLASS_PATH = os.path.join(PROJECT_ROOT, "processed_data", "menu_classification", "menu_classification.parquet")
PRICING_PATH = os.path.join(PROJECT_ROOT, "processed_data", "pricing", "price_sensitivity_analysis.parquet")
WASTAGE_PATH = os.path.join(PROJECT_ROOT, "processed_data", "wastage", "wastage_by_item.parquet")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "what_if")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "what_if")

SIMULATION_DISCLAIMER = (
    "ESTIMATE ONLY: Simulated outputs represent theoretical projections generated via "
    "econometric and behavioral models and must not be interpreted as actual historical results."
)


@dataclass
class SimulationResult:
    """
    Standardized result structure for What-If scenario simulations.
    Explicitly tracks baseline vs simulated estimates across all 5 SRS indicators.
    """
    scenario_name: str
    scenario_parameters: Dict[str, Any]
    is_simulation_estimate: bool = True
    disclaimer: str = SIMULATION_DISCLAIMER

    revenue: Dict[str, float] = field(default_factory=dict)
    contribution_margin: Dict[str, float] = field(default_factory=dict)
    demand: Dict[str, float] = field(default_factory=dict)
    wastage: Dict[str, float] = field(default_factory=dict)
    profitability: Dict[str, float] = field(default_factory=dict)

    key_findings: List[str] = field(default_factory=list)
    item_level_estimates: Optional[pd.DataFrame] = None

    @property
    def revenue_baseline(self) -> float:
        return float(self.revenue.get("baseline", 0.0))

    @property
    def revenue_simulated(self) -> float:
        return float(self.revenue.get("estimated", 0.0))

    @property
    def revenue_delta(self) -> float:
        return float(self.revenue.get("delta", 0.0))

    @property
    def revenue_pct_change(self) -> float:
        return float(self.revenue.get("pct_change", 0.0))

    @property
    def profit_baseline(self) -> float:
        return float(self.profitability.get("baseline_net_profit", 0.0))

    @property
    def profit_simulated(self) -> float:
        return float(self.profitability.get("estimated_net_profit", 0.0))

    @property
    def profit_delta(self) -> float:
        return float(self.profitability.get("delta_net_profit", 0.0))

    @property
    def profit_pct_change(self) -> float:
        return float(self.profitability.get("pct_change", 0.0))

    @property
    def demand_baseline(self) -> float:
        return float(self.demand.get("baseline_units", 0.0))

    @property
    def demand_simulated(self) -> float:
        return float(self.demand.get("estimated_units", 0.0))

    @property
    def demand_delta(self) -> float:
        return float(self.demand.get("delta_units", 0.0))

    @property
    def demand_pct_change(self) -> float:
        return float(self.demand.get("pct_change", 0.0))

    @property
    def margin_pct_baseline(self) -> float:
        return float(self.contribution_margin.get("baseline_pct", 0.0))

    @property
    def margin_pct_simulated(self) -> float:
        return float(self.contribution_margin.get("estimated_pct", 0.0))

    @property
    def margin_pct_delta(self) -> float:
        return float(self.contribution_margin.get("delta_pct", 0.0))

    @property
    def wastage_baseline(self) -> float:
        return float(self.wastage.get("baseline_cost", 0.0))

    @property
    def wastage_simulated(self) -> float:
        return float(self.wastage.get("estimated_cost", 0.0))

    @property
    def wastage_delta(self) -> float:
        return float(self.wastage.get("delta_cost", 0.0))

    @property
    def wastage_pct_change(self) -> float:
        return float(self.wastage.get("pct_change", 0.0))

    @property
    def target_entity(self) -> str:
        return str(self.scenario_parameters.get("item_id", "ITEM-001"))

    @property
    def business_rationale(self) -> str:
        if isinstance(self.key_findings, list) and len(self.key_findings) > 0:
            return " • ".join(self.key_findings)
        return str(self.key_findings or "Simulation estimated via econometric behavioral models.")

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to JSON-serializable dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "scenario_parameters": self.scenario_parameters,
            "is_simulation_estimate": self.is_simulation_estimate,
            "disclaimer": self.disclaimer,
            "estimated_impact_indicators": {
                "revenue": self.revenue,
                "contribution_margin": self.contribution_margin,
                "demand": self.demand,
                "wastage": self.wastage,
                "profitability": self.profitability
            },
            "key_findings": self.key_findings
        }

    def summary_table(self) -> pd.DataFrame:
        """Returns comparison DataFrame across all 5 SRS indicators."""
        rows = [
            {
                "indicator": "Revenue ($)",
                "baseline_actual": round(self.revenue["baseline"], 2),
                "simulated_estimate": round(self.revenue["estimated"], 2),
                "estimated_delta": round(self.revenue["delta"], 2),
                "estimated_pct_change": round(self.revenue["pct_change"], 2)
            },
            {
                "indicator": "Contribution Margin ($)",
                "baseline_actual": round(self.contribution_margin["baseline_dollars"], 2),
                "simulated_estimate": round(self.contribution_margin["estimated_dollars"], 2),
                "estimated_delta": round(self.contribution_margin["delta_dollars"], 2),
                "estimated_pct_change": round(self.contribution_margin["delta_pct"], 2)
            },
            {
                "indicator": "Contribution Margin (%)",
                "baseline_actual": round(self.contribution_margin["baseline_pct"], 2),
                "simulated_estimate": round(self.contribution_margin["estimated_pct"], 2),
                "estimated_delta": round(self.contribution_margin["estimated_pct"] - self.contribution_margin["baseline_pct"], 2),
                "estimated_pct_change": round(
                    ((self.contribution_margin["estimated_pct"] - self.contribution_margin["baseline_pct"]) /
                     max(0.01, self.contribution_margin["baseline_pct"])) * 100, 2
                )
            },
            {
                "indicator": "Demand (Units Sold)",
                "baseline_actual": round(self.demand["baseline_units"], 0),
                "simulated_estimate": round(self.demand["estimated_units"], 0),
                "estimated_delta": round(self.demand["delta_units"], 0),
                "estimated_pct_change": round(self.demand["pct_change"], 2)
            },
            {
                "indicator": "Wastage Cost ($)",
                "baseline_actual": round(self.wastage["baseline_cost"], 2),
                "simulated_estimate": round(self.wastage["estimated_cost"], 2),
                "estimated_delta": round(self.wastage["delta_cost"], 2),
                "estimated_pct_change": round(self.wastage["pct_change"], 2)
            },
            {
                "indicator": "Net Profitability ($)",
                "baseline_actual": round(self.profitability["baseline_net_profit"], 2),
                "simulated_estimate": round(self.profitability["estimated_net_profit"], 2),
                "estimated_delta": round(self.profitability["delta_net_profit"], 2),
                "estimated_pct_change": round(self.profitability["pct_change"], 2)
            }
        ]
        df = pd.DataFrame(rows)
        df["is_simulation_estimate"] = True
        return df


class WhatIfScenarioEngine:
    """
    Simulation engine implementing SRS Step 40 (the 8 What-If Scenarios)
    and Step 41 (the 5 Scenario Impact Indicators with mandatory estimate disclaimers).
    """

    SUPPORTED_SCENARIOS = [
        "Increase menu price",
        "Reduce item price",
        "Change discount percentage",
        "Increase promotion frequency",
        "Remove a menu item",
        "Reduce preparation quantity",
        "Increase predicted demand",
        "Change wastage assumptions"
    ]

    def __init__(
        self,
        menu_class_df: Optional[pd.DataFrame] = None,
        pricing_df: Optional[pd.DataFrame] = None,
        wastage_df: Optional[pd.DataFrame] = None
    ):
        self.menu_class = menu_class_df if menu_class_df is not None else self._load_df(MENU_CLASS_PATH)
        self.pricing = pricing_df if pricing_df is not None else self._load_df(PRICING_PATH)
        self.wastage = wastage_df if wastage_df is not None else self._load_df(WASTAGE_PATH)

        self.baseline_df = self._construct_unified_baseline()

    @staticmethod
    def _load_df(path: str) -> pd.DataFrame:
        if os.path.exists(path):
            return pd.read_parquet(path)
        return pd.DataFrame()

    def _construct_unified_baseline(self) -> pd.DataFrame:
        """
        Constructs master baseline dataset joining menu items, pricing elasticity,
        and wastage loss amounts.
        """
        cols = [
            "item_id", "item_name", "category_id", "category_name",
            "base_price", "cost_price", "quantity_sold", "revenue",
            "cost", "contribution_margin", "profit_percentage"
        ]
        available_cols = [c for c in cols if c in self.menu_class.columns]
        df = self.menu_class[available_cols].copy()

        if not self.pricing.empty:
            p_cols = ["item_id", "avg_empirical_elasticity", "price_sensitivity_tier"]
            avail_p = [c for c in p_cols if c in self.pricing.columns]
            df = df.merge(self.pricing[avail_p], on="item_id", how="left")

        if not self.wastage.empty:
            w_cols = ["item_id", "wasted_quantity", "total_loss_amount", "wastage_rate_pct"]
            avail_w = [c for c in w_cols if c in self.wastage.columns]
            df = df.merge(self.wastage[avail_w], on="item_id", how="left")

        df["avg_empirical_elasticity"] = df["avg_empirical_elasticity"].fillna(1.2)
        df["price_sensitivity_tier"] = df["price_sensitivity_tier"].fillna("Moderately Price Sensitive")
        df["wasted_quantity"] = df["wasted_quantity"].fillna(0.0)
        df["total_loss_amount"] = df["total_loss_amount"].fillna(0.0)
        df["wastage_rate_pct"] = df["wastage_rate_pct"].fillna(5.0)

        df["baseline_net_profit"] = df["contribution_margin"] - df["total_loss_amount"]

        return df

    def _build_simulation_result(
        self,
        scenario_name: str,
        params: Dict[str, Any],
        simulated_df: pd.DataFrame,
        findings: List[str]
    ) -> SimulationResult:
        """Computes aggregate impact across the 5 SRS indicators."""
        base_rev = float(self.baseline_df["revenue"].sum())
        est_rev = float(simulated_df["simulated_revenue"].sum())
        delta_rev = est_rev - base_rev
        pct_rev = (delta_rev / base_rev) * 100 if base_rev > 0 else 0.0

        base_demand = float(self.baseline_df["quantity_sold"].sum())
        est_demand = float(simulated_df["simulated_quantity"].sum())
        delta_demand = est_demand - base_demand
        pct_demand = (delta_demand / base_demand) * 100 if base_demand > 0 else 0.0

        base_cm = float(self.baseline_df["contribution_margin"].sum())
        est_cm = float(simulated_df["simulated_contribution_margin"].sum())
        delta_cm = est_cm - base_cm
        base_cm_pct = (base_cm / base_rev) * 100 if base_rev > 0 else 0.0
        est_cm_pct = (est_cm / est_rev) * 100 if est_rev > 0 else 0.0
        delta_cm_pct = (delta_cm / base_cm) * 100 if base_cm > 0 else 0.0

        base_waste_qty = float(self.baseline_df["wasted_quantity"].sum())
        est_waste_qty = float(simulated_df["simulated_wasted_quantity"].sum())
        base_waste_cost = float(self.baseline_df["total_loss_amount"].sum())
        est_waste_cost = float(simulated_df["simulated_loss_amount"].sum())
        delta_waste_cost = est_waste_cost - base_waste_cost
        pct_waste_cost = (delta_waste_cost / base_waste_cost) * 100 if base_waste_cost > 0 else 0.0

        base_profit = float(self.baseline_df["baseline_net_profit"].sum())
        est_profit = float(simulated_df["simulated_net_profit"].sum())
        delta_profit = est_profit - base_profit
        pct_profit = (delta_profit / abs(base_profit)) * 100 if base_profit != 0 else 0.0

        return SimulationResult(
            scenario_name=scenario_name,
            scenario_parameters=params,
            is_simulation_estimate=True,
            disclaimer=SIMULATION_DISCLAIMER,
            revenue={
                "baseline": base_rev,
                "estimated": est_rev,
                "delta": delta_rev,
                "pct_change": pct_rev
            },
            contribution_margin={
                "baseline_dollars": base_cm,
                "estimated_dollars": est_cm,
                "delta_dollars": delta_cm,
                "baseline_pct": base_cm_pct,
                "estimated_pct": est_cm_pct,
                "delta_pct": delta_cm_pct
            },
            demand={
                "baseline_units": base_demand,
                "estimated_units": est_demand,
                "delta_units": delta_demand,
                "pct_change": pct_demand
            },
            wastage={
                "baseline_quantity": base_waste_qty,
                "estimated_quantity": est_waste_qty,
                "baseline_cost": base_waste_cost,
                "estimated_cost": est_waste_cost,
                "delta_cost": delta_waste_cost,
                "pct_change": pct_waste_cost
            },
            profitability={
                "baseline_net_profit": base_profit,
                "estimated_net_profit": est_profit,
                "delta_net_profit": delta_profit,
                "pct_change": pct_profit
            },
            key_findings=findings,
            item_level_estimates=simulated_df
        )

    def simulate_increase_menu_price(
        self,
        item_id: Optional[str] = None,
        category_id: Optional[str] = None,
        price_increase_pct: float = 10.0,
        prep_adaptation_pct: float = 1.0
    ) -> SimulationResult:
        """
        Simulates price increase using empirical price elasticity |ε_i|.
        High elasticity items contract in demand proportionally to ε_i * ΔP/P.
        """
        df = self.baseline_df.copy()
        mask = pd.Series(True, index=df.index)
        if item_id:
            mask = mask & (df["item_id"] == item_id)
        if category_id:
            mask = mask & (df["category_id"] == category_id)

        fraction = price_increase_pct / 100.0

        df["simulated_price"] = np.where(mask, df["base_price"] * (1.0 + fraction), df["base_price"])

        elasticity = df["avg_empirical_elasticity"]
        demand_change_pct = np.where(mask, -elasticity * fraction, 0.0)
        df["simulated_quantity"] = np.maximum(0.0, df["quantity_sold"] * (1.0 + demand_change_pct))

        df["simulated_revenue"] = df["simulated_quantity"] * df["simulated_price"]
        df["simulated_cost"] = df["simulated_quantity"] * df["cost_price"]
        df["simulated_contribution_margin"] = df["simulated_revenue"] - df["simulated_cost"]

        unadapted_share = 1.0 - prep_adaptation_pct
        waste_multiplier = np.where(mask, 1.0 + (np.abs(demand_change_pct) * unadapted_share), 1.0)
        df["simulated_wasted_quantity"] = df["wasted_quantity"] * waste_multiplier
        df["simulated_loss_amount"] = df["simulated_wasted_quantity"] * df["cost_price"]
        df["simulated_net_profit"] = df["simulated_contribution_margin"] - df["simulated_loss_amount"]

        findings = [
            f"Simulated price increase of +{price_increase_pct:.1f}% applied.",
            "Higher prices expand gross margin per unit, but trigger elastic volume contraction.",
            "Net profitability impact depends heavily on item price elasticity tier."
        ]

        return self._build_simulation_result(
            scenario_name="Increase menu price",
            params={
                "item_id": item_id,
                "category_id": category_id,
                "price_increase_pct": price_increase_pct,
                "prep_adaptation_pct": prep_adaptation_pct
            },
            simulated_df=df,
            findings=findings
        )

    def simulate_reduce_item_price(
        self,
        item_id: Optional[str] = None,
        category_id: Optional[str] = None,
        price_reduction_pct: float = 10.0,
        prep_adaptation_pct: float = 1.0
    ) -> SimulationResult:
        """
        Simulates price reduction using empirical price elasticity |ε_i|.
        Lower prices stimulate demand according to ε_i * |ΔP/P|.
        """
        df = self.baseline_df.copy()
        mask = pd.Series(True, index=df.index)
        if item_id:
            mask = mask & (df["item_id"] == item_id)
        if category_id:
            mask = mask & (df["category_id"] == category_id)

        fraction = price_reduction_pct / 100.0

        df["simulated_price"] = np.where(mask, df["base_price"] * (1.0 - fraction), df["base_price"])

        elasticity = df["avg_empirical_elasticity"]
        demand_change_pct = np.where(mask, elasticity * fraction, 0.0)
        df["simulated_quantity"] = df["quantity_sold"] * (1.0 + demand_change_pct)

        df["simulated_revenue"] = df["simulated_quantity"] * df["simulated_price"]
        df["simulated_cost"] = df["simulated_quantity"] * df["cost_price"]
        df["simulated_contribution_margin"] = df["simulated_revenue"] - df["simulated_cost"]

        waste_reduction = np.where(mask, np.clip(demand_change_pct * 0.40, 0.0, 0.50), 0.0)
        df["simulated_wasted_quantity"] = df["wasted_quantity"] * (1.0 - waste_reduction)
        df["simulated_loss_amount"] = df["simulated_wasted_quantity"] * df["cost_price"]
        df["simulated_net_profit"] = df["simulated_contribution_margin"] - df["simulated_loss_amount"]

        findings = [
            f"Simulated price reduction of -{price_reduction_pct:.1f}% applied.",
            "Lower price expands customer unit demand, absorbing kitchen prep and reducing spoilage.",
            "Contribution margin per unit contracts; requires sufficient elasticity to remain profit-accretive."
        ]

        return self._build_simulation_result(
            scenario_name="Reduce item price",
            params={
                "item_id": item_id,
                "category_id": category_id,
                "price_reduction_pct": price_reduction_pct,
                "prep_adaptation_pct": prep_adaptation_pct
            },
            simulated_df=df,
            findings=findings
        )

    def simulate_change_discount_percentage(
        self,
        current_discount_pct: float = 10.0,
        new_discount_pct: Optional[float] = None,
        discount_pct: Optional[float] = None,
        item_id: Optional[str] = None,
        category_id: Optional[str] = None
    ) -> SimulationResult:
        """
        Simulates adjusting promotional discount depths.
        Incorporates SRS Step 28 promotion trap mechanics (margin dilution vs volume lift).
        """
        if new_discount_pct is None:
            if discount_pct is not None:
                new_discount_pct = current_discount_pct + float(discount_pct)
            else:
                new_discount_pct = 20.0
        df = self.baseline_df.copy()
        mask = pd.Series(True, index=df.index)
        if item_id:
            mask = mask & (df["item_id"] == item_id)
        if category_id:
            mask = mask & (df["category_id"] == category_id)

        delta_discount = (new_discount_pct - current_discount_pct) / 100.0
        realized_price_factor = 1.0 - (new_discount_pct / 100.0)

        df["simulated_price"] = np.where(mask, df["base_price"] * realized_price_factor, df["base_price"])

        volume_lift_pct = np.where(mask, np.maximum(-0.5, delta_discount * 1.35), 0.0)
        df["simulated_quantity"] = df["quantity_sold"] * (1.0 + volume_lift_pct)

        df["simulated_revenue"] = df["simulated_quantity"] * df["simulated_price"]
        df["simulated_cost"] = df["simulated_quantity"] * df["cost_price"]
        df["simulated_contribution_margin"] = df["simulated_revenue"] - df["simulated_cost"]

        spoilage_factor = 1.0 + np.maximum(0.0, delta_discount * 0.50)
        df["simulated_wasted_quantity"] = df["wasted_quantity"] * np.where(mask, spoilage_factor, 1.0)
        df["simulated_loss_amount"] = df["simulated_wasted_quantity"] * df["cost_price"]
        df["simulated_net_profit"] = df["simulated_contribution_margin"] - df["simulated_loss_amount"]

        findings = [
            f"Simulated discount adjustment from {current_discount_pct:.1f}% to {new_discount_pct:.1f}%.",
            f"Estimated demand volume shifts by {float(volume_lift_pct.mean()*100):+.1f}%.",
            "Evaluates margin dilution to protect against SRS Promotion Trap 2 (margin collapse)."
        ]

        return self._build_simulation_result(
            scenario_name="Change discount percentage",
            params={
                "current_discount_pct": current_discount_pct,
                "new_discount_pct": new_discount_pct,
                "item_id": item_id,
                "category_id": category_id
            },
            simulated_df=df,
            findings=findings
        )

    def simulate_increase_promotion_frequency(
        self,
        promo_frequency_multiplier: float = 1.5,
        campaign_discount_pct: float = 15.0
    ) -> SimulationResult:
        """
        Simulates running promotions more frequently across the network.
        Models volume surge on promo days, discount margin dilution, and post-campaign behavior.
        """
        df = self.baseline_df.copy()

        base_promo_share = 0.20
        new_promo_share = np.clip(base_promo_share * promo_frequency_multiplier, 0.0, 0.60)
        incremental_promo_orders = (new_promo_share - base_promo_share)

        volume_lift = incremental_promo_orders * 0.22
        df["simulated_quantity"] = df["quantity_sold"] * (1.0 + volume_lift)

        blended_discount_pct = (new_promo_share * campaign_discount_pct) / 100.0
        df["simulated_price"] = df["base_price"] * (1.0 - blended_discount_pct)

        df["simulated_revenue"] = df["simulated_quantity"] * df["simulated_price"]
        df["simulated_cost"] = df["simulated_quantity"] * df["cost_price"]
        df["simulated_contribution_margin"] = df["simulated_revenue"] - df["simulated_cost"]

        promo_waste_increase = 1.0 + (incremental_promo_orders * 0.35)
        df["simulated_wasted_quantity"] = df["wasted_quantity"] * promo_waste_increase
        df["simulated_loss_amount"] = df["simulated_wasted_quantity"] * df["cost_price"]
        df["simulated_net_profit"] = df["simulated_contribution_margin"] - df["simulated_loss_amount"]

        findings = [
            f"Simulated promotion frequency multiplier of {promo_frequency_multiplier:.2f}x.",
            f"Effective network promotional order share increases to {new_promo_share*100:.1f}%.",
            "Models demand lift against incremental food cost, margin discount, and overproduction spoilage."
        ]

        return self._build_simulation_result(
            scenario_name="Increase promotion frequency",
            params={
                "promo_frequency_multiplier": promo_frequency_multiplier,
                "campaign_discount_pct": campaign_discount_pct
            },
            simulated_df=df,
            findings=findings
        )

    def simulate_remove_menu_item(
        self,
        item_id: str,
        demand_substitution_rate: float = 0.50
    ) -> SimulationResult:
        """
        Simulates removing a specific dish (e.g. persistent Low Performer or high wastage item).
        Recovers 100% of the item's wastage loss and models category substitution recapture.
        """
        df = self.baseline_df.copy()
        target_mask = (df["item_id"] == item_id)
        if not target_mask.any():
            item_id = df.iloc[0]["item_id"]
            target_mask = (df["item_id"] == item_id)

        target_row = df[target_mask].iloc[0]
        cat_id = target_row["category_id"]
        removed_qty = target_row["quantity_sold"]
        removed_rev = target_row["revenue"]
        removed_waste = target_row["total_loss_amount"]
        item_name = target_row["item_name"]

        sibling_mask = (df["category_id"] == cat_id) & (~target_mask)
        sibling_count = sibling_mask.sum()

        df.loc[target_mask, "simulated_quantity"] = 0.0
        df.loc[target_mask, "simulated_price"] = 0.0
        df.loc[target_mask, "simulated_revenue"] = 0.0
        df.loc[target_mask, "simulated_cost"] = 0.0
        df.loc[target_mask, "simulated_contribution_margin"] = 0.0
        df.loc[target_mask, "simulated_wasted_quantity"] = 0.0
        df.loc[target_mask, "simulated_loss_amount"] = 0.0
        df.loc[target_mask, "simulated_net_profit"] = 0.0

        df.loc[~target_mask, "simulated_price"] = df.loc[~target_mask, "base_price"]
        df.loc[~target_mask, "simulated_quantity"] = df.loc[~target_mask, "quantity_sold"]
        df.loc[~target_mask, "simulated_wasted_quantity"] = df.loc[~target_mask, "wasted_quantity"]
        df.loc[~target_mask, "simulated_loss_amount"] = df.loc[~target_mask, "total_loss_amount"]

        if sibling_count > 0:
            recaptured_units_per_sibling = (removed_qty * demand_substitution_rate) / sibling_count
            df.loc[sibling_mask, "simulated_quantity"] += recaptured_units_per_sibling

        df.loc[~target_mask, "simulated_revenue"] = df.loc[~target_mask, "simulated_quantity"] * df.loc[~target_mask, "simulated_price"]
        df.loc[~target_mask, "simulated_cost"] = df.loc[~target_mask, "simulated_quantity"] * df.loc[~target_mask, "cost_price"]
        df.loc[~target_mask, "simulated_contribution_margin"] = df.loc[~target_mask, "simulated_revenue"] - df.loc[~target_mask, "simulated_cost"]
        df.loc[~target_mask, "simulated_net_profit"] = df.loc[~target_mask, "simulated_contribution_margin"] - df.loc[~target_mask, "simulated_loss_amount"]

        findings = [
            f"Simulated removal of item {item_id} ('{item_name}').",
            f"100% of item wastage loss (${removed_waste:,.2f}) is eliminated immediately.",
            f"{demand_substitution_rate*100:.0f}% of lost demand is recaptured by sibling dishes in category {cat_id}."
        ]

        return self._build_simulation_result(
            scenario_name="Remove a menu item",
            params={
                "item_id": item_id,
                "item_name": item_name,
                "demand_substitution_rate": demand_substitution_rate
            },
            simulated_df=df,
            findings=findings
        )

    def simulate_reduce_preparation_quantity(
        self,
        item_id: Optional[str] = None,
        prep_reduction_pct: float = 20.0
    ) -> SimulationResult:
        """
        Simulates reducing daily kitchen preparation quantities for high-wastage dishes.
        Directly reduces food spoilage cost while protecting sales within demand thresholds.
        """
        df = self.baseline_df.copy()
        mask = pd.Series(True, index=df.index)
        if item_id:
            mask = mask & (df["item_id"] == item_id)
        else:
            mask = mask & (df["wastage_rate_pct"] >= 15.0)

        fraction = prep_reduction_pct / 100.0

        df["simulated_price"] = df["base_price"]
        df["simulated_quantity"] = df["quantity_sold"]
        df["simulated_revenue"] = df["revenue"]
        df["simulated_cost"] = df["cost"]
        df["simulated_contribution_margin"] = df["contribution_margin"]

        waste_reduction_factor = 1.0 - fraction
        df["simulated_wasted_quantity"] = np.where(
            mask,
            df["wasted_quantity"] * waste_reduction_factor,
            df["wasted_quantity"]
        )
        df["simulated_loss_amount"] = df["simulated_wasted_quantity"] * df["cost_price"]
        df["simulated_net_profit"] = df["simulated_contribution_margin"] - df["simulated_loss_amount"]

        findings = [
            f"Simulated daily preparation quantity reduction of {prep_reduction_pct:.1f}%.",
            f"Applied across {int(mask.sum())} high-wastage dishes exhibiting structural overproduction.",
            "Wastage losses drop directly with zero disruption to consumer fulfilled demand."
        ]

        return self._build_simulation_result(
            scenario_name="Reduce preparation quantity",
            params={
                "item_id": item_id,
                "prep_reduction_pct": prep_reduction_pct,
                "dishes_affected": int(mask.sum())
            },
            simulated_df=df,
            findings=findings
        )

    def simulate_increase_predicted_demand(
        self,
        demand_increase_pct: float = 15.0,
        category_id: Optional[str] = None,
        item_id: Optional[str] = None,
        prep_scaling: bool = True
    ) -> SimulationResult:
        """
        Simulates macro demand expansion (holiday surges, local events, marketing push).
        Evaluates kitchen throughput scalability, revenue growth, and capacity constraints.
        """
        df = self.baseline_df.copy()
        mask = pd.Series(True, index=df.index)
        if item_id:
            mask = mask & (df["item_id"] == item_id)
        if category_id:
            mask = mask & (df["category_id"] == category_id)

        surge_factor = 1.0 + (demand_increase_pct / 100.0)

        df["simulated_price"] = df["base_price"]
        df["simulated_quantity"] = np.where(mask, df["quantity_sold"] * surge_factor, df["quantity_sold"])
        df["simulated_revenue"] = df["simulated_quantity"] * df["simulated_price"]
        df["simulated_cost"] = df["simulated_quantity"] * df["cost_price"]
        df["simulated_contribution_margin"] = df["simulated_revenue"] - df["simulated_cost"]

        turnover_efficiency = 0.90 if prep_scaling else 1.15
        df["simulated_wasted_quantity"] = np.where(mask, df["wasted_quantity"] * turnover_efficiency, df["wasted_quantity"])
        df["simulated_loss_amount"] = df["simulated_wasted_quantity"] * df["cost_price"]
        df["simulated_net_profit"] = df["simulated_contribution_margin"] - df["simulated_loss_amount"]

        findings = [
            f"Simulated network-wide demand expansion of +{demand_increase_pct:.1f}%.",
            f"Preparation scaling mode: {'Adaptive buffer scaling' if prep_scaling else 'Fixed prep bottlenecks'}.",
            "High turnover accelerates inventory velocity and amplifies gross operating margin."
        ]

        return self._build_simulation_result(
            scenario_name="Increase predicted demand",
            params={
                "demand_increase_pct": demand_increase_pct,
                "category_id": category_id,
                "prep_scaling": prep_scaling
            },
            simulated_df=df,
            findings=findings
        )

    def simulate_change_wastage_assumptions(
        self,
        wastage_rate_change_pct: Optional[float] = None,
        wastage_reduction_pct: Optional[float] = None,
        item_id: Optional[str] = None,
        category_id: Optional[str] = None
    ) -> SimulationResult:
        """
        Simulates changes in kitchen waste assumptions (cold chain improvements,
        supplier quality changes, or shelf-life degradation).
        Directly adjusts spoilage rate and recalculates net restaurant profitability.
        """
        if wastage_rate_change_pct is None:
            if wastage_reduction_pct is not None:
                wastage_rate_change_pct = -abs(float(wastage_reduction_pct))
            else:
                wastage_rate_change_pct = -20.0

        df = self.baseline_df.copy()
        mask = pd.Series(True, index=df.index)
        if item_id:
            mask = mask & (df["item_id"] == item_id)
        if category_id:
            mask = mask & (df["category_id"] == category_id)

        waste_factor = 1.0 + (wastage_rate_change_pct / 100.0)

        df["simulated_price"] = df["base_price"]
        df["simulated_quantity"] = df["quantity_sold"]
        df["simulated_revenue"] = df["revenue"]
        df["simulated_cost"] = df["cost"]
        df["simulated_contribution_margin"] = df["contribution_margin"]

        df["simulated_wasted_quantity"] = np.where(mask, df["wasted_quantity"] * waste_factor, df["wasted_quantity"])
        df["simulated_loss_amount"] = df["simulated_wasted_quantity"] * df["cost_price"]
        df["simulated_net_profit"] = df["simulated_contribution_margin"] - df["simulated_loss_amount"]

        findings = [
            f"Simulated wastage rate adjustment of {wastage_rate_change_pct:+.1f}%.",
            "Evaluates operational improvements (e.g. IoT cold chain, dynamic prep forecasting).",
            "Direct 1:1 flow-through from reduced spoilage into net restaurant profitability."
        ]

        return self._build_simulation_result(
            scenario_name="Change wastage assumptions",
            params={
                "wastage_rate_change_pct": wastage_rate_change_pct,
                "category_id": category_id
            },
            simulated_df=df,
            findings=findings
        )

    def run_standard_scenario_benchmark(self) -> pd.DataFrame:
        """
        Executes a standardized comparative benchmark across all 8 SRS scenarios.
        Returns a structured summary matrix with explicit estimate labels and disclaimers.
        """
        scenarios = [
            ("Increase menu price (+10% on Filet Mignon)", lambda: self.simulate_increase_menu_price(item_id="ITEM-046", price_increase_pct=10.0)),
            ("Reduce item price (-10% on Wild King Salmon)", lambda: self.simulate_reduce_item_price(item_id="ITEM-063", price_reduction_pct=10.0)),
            ("Change discount percentage (10% -> 20%)", lambda: self.simulate_change_discount_percentage(current_discount_pct=10.0, new_discount_pct=20.0)),
            ("Increase promotion frequency (1.5x frequency)", lambda: self.simulate_increase_promotion_frequency(promo_frequency_multiplier=1.5)),
            ("Remove a menu item (Phase out Wagyu ITEM-051)", lambda: self.simulate_remove_menu_item(item_id="ITEM-051", demand_substitution_rate=0.50)),
            ("Reduce preparation quantity (-20% high-waste prep)", lambda: self.simulate_reduce_preparation_quantity(prep_reduction_pct=20.0)),
            ("Increase predicted demand (+15% holiday surge)", lambda: self.simulate_increase_predicted_demand(demand_increase_pct=15.0)),
            ("Change wastage assumptions (-25% cold chain upgrade)", lambda: self.simulate_change_wastage_assumptions(wastage_rate_change_pct=-25.0)),
        ]

        benchmark_rows = []
        for label, fn in scenarios:
            res = fn()
            benchmark_rows.append({
                "scenario_name": res.scenario_name,
                "scenario_variant": label,
                "is_simulation_estimate": True,
                "baseline_revenue": round(res.revenue["baseline"], 2),
                "estimated_revenue": round(res.revenue["estimated"], 2),
                "estimated_revenue_delta": round(res.revenue["delta"], 2),
                "estimated_revenue_pct_change": round(res.revenue["pct_change"], 2),
                "baseline_contribution_margin": round(res.contribution_margin["baseline_dollars"], 2),
                "estimated_contribution_margin": round(res.contribution_margin["estimated_dollars"], 2),
                "estimated_margin_delta": round(res.contribution_margin["delta_dollars"], 2),
                "estimated_margin_pct_change": round(res.contribution_margin["delta_pct"], 2),
                "baseline_demand_units": round(res.demand["baseline_units"], 0),
                "estimated_demand_units": round(res.demand["estimated_units"], 0),
                "estimated_demand_delta": round(res.demand["delta_units"], 0),
                "estimated_demand_pct_change": round(res.demand["pct_change"], 2),
                "baseline_wastage_cost": round(res.wastage["baseline_cost"], 2),
                "estimated_wastage_cost": round(res.wastage["estimated_cost"], 2),
                "estimated_wastage_cost_delta": round(res.wastage["delta_cost"], 2),
                "estimated_wastage_pct_change": round(res.wastage["pct_change"], 2),
                "baseline_net_profitability": round(res.profitability["baseline_net_profit"], 2),
                "estimated_net_profitability": round(res.profitability["estimated_net_profit"], 2),
                "estimated_profitability_delta": round(res.profitability["delta_net_profit"], 2),
                "estimated_profitability_pct_change": round(res.profitability["pct_change"], 2),
                "disclaimer": res.disclaimer
            })

        return pd.DataFrame(benchmark_rows)


def run_what_if_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes end-to-end What-If simulation benchmark, exports datasets,
    and generates structured reports for Steps 40-41.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 75)
    print("DineIQ Analytics - What-If Scenario Analysis Engine (SRS Steps 40-41)")
    print("=" * 75)

    print("[Engine] Initializing What-If Scenario Engine...")
    engine = WhatIfScenarioEngine()

    print("[Simulation] Running standardized benchmark across all 8 SRS scenarios...")
    benchmark_df = engine.run_standard_scenario_benchmark()

    print(f"[Exporting] Persisting simulation outputs to {OUTPUT_DIR}...")
    parquet_path = os.path.join(OUTPUT_DIR, "what_if_scenario_benchmark.parquet")
    csv_path = os.path.join(OUTPUT_DIR, "what_if_scenario_benchmark.csv")
    benchmark_df.to_parquet(parquet_path, index=False)
    benchmark_df.to_csv(csv_path, index=False)

    stats = {
        "simulation_timestamp": datetime.now().isoformat(),
        "is_simulation_estimate": True,
        "disclaimer": SIMULATION_DISCLAIMER,
        "total_scenarios_simulated": len(benchmark_df),
        "srs_scenarios_covered": engine.SUPPORTED_SCENARIOS,
        "indicators_modeled": [
            "Revenue", "Contribution margin", "Demand", "Wastage", "Profitability"
        ],
        "scenario_benchmarks": benchmark_df.to_dict(orient="records")
    }

    json_path = os.path.join(REPORTS_DIR, "what_if_scenario_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    report_path = os.path.join(REPORTS_DIR, "what_if_scenario_report.md")
    report_content = _build_markdown_report(stats, benchmark_df)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[Done] Report generated at: {report_path}")
    print(f"[Done] Simulated all 8 SRS scenarios with complete 5-indicator impact projections.")

    return benchmark_df, stats


def _build_markdown_report(stats: Dict[str, Any], df: pd.DataFrame) -> str:
    """Generates comprehensive executive markdown report for Steps 40-41."""
    md = f"""# DineIQ Analytics - What-If Scenario Analysis Report
**SRS References:** Step 40 (What-If Scenario Analysis), Step 41 (Scenario Impact Analysis)  
**Simulation Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  

> [!WARNING]
> **MANDATORY SRS NOTICE: SIMULATED ESTIMATES ONLY**  
> *{SIMULATION_DISCLAIMER}*  
> All figures presented below represent hypothetical econometric projections and must not be interpreted as actual historical results.

---

The simulation engine evaluated the exact 8 scenarios mandated by SRS Step 40 across all 5 impact indicators required by SRS Step 41:
1. **Revenue**
2. **Contribution Margin**
3. **Demand**
4. **Wastage Cost**
5. **Net Profitability**

| # | Scenario Name | Estimated Revenue ($) | Estimated Margin ($) | Estimated Demand (Units) | Estimated Wastage ($) | Estimated Net Profit ($) | Profit Delta (%) |
|---|---|---|---|---|---|---|---|
"""
    for idx, row in df.iterrows():
        md += f"| {idx+1} | **{row['scenario_variant']}** | ${row['estimated_revenue']:,.2f} | ${row['estimated_contribution_margin']:,.2f} | {row['estimated_demand_units']:,.0f} | ${row['estimated_wastage_cost']:,.2f} | **${row['estimated_net_profitability']:,.2f}** | **{row['estimated_profitability_pct_change']:+.2f}%** |\n"

    md += """
---


"""
    for idx, row in df.iterrows():
        md += f"""### Scenario {idx+1}: {row['scenario_name']} (`{row['scenario_variant']}`)
*Classification: SIMULATED ESTIMATE (SRS Step 41)*

| Indicator | Baseline (Actual) | Estimated (Simulated) | Estimated Delta ($ / Units) | Estimated Change (%) |
|---|---|---|---|---|
| **Revenue** | ${row['baseline_revenue']:,.2f} | ${row['estimated_revenue']:,.2f} | ${row['estimated_revenue_delta']:+,.2f} | {row['estimated_revenue_pct_change']:+.2f}% |
| **Contribution Margin** | ${row['baseline_contribution_margin']:,.2f} | ${row['estimated_contribution_margin']:,.2f} | ${row['estimated_margin_delta']:+,.2f} | {row['estimated_margin_pct_change']:+.2f}% |
| **Demand** | {row['baseline_demand_units']:,.0f} units | {row['estimated_demand_units']:,.0f} units | {row['estimated_demand_delta']:+,.0f} units | {row['estimated_demand_pct_change']:+.2f}% |
| **Wastage Cost** | ${row['baseline_wastage_cost']:,.2f} | ${row['estimated_wastage_cost']:,.2f} | ${row['estimated_wastage_cost_delta']:+,.2f} | {row['estimated_wastage_pct_change']:+.2f}% |
| **Net Profitability** | ${row['baseline_net_profitability']:,.2f} | ${row['estimated_net_profitability']:,.2f} | ${row['estimated_profitability_delta']:+,.2f} | **{row['estimated_profitability_pct_change']:+.2f}%** |

**Strategic Takeaway:**  
- **Revenue & Margin Dynamic:** Simulated revenue changed by {row['estimated_revenue_pct_change']:+.2f}%, while contribution margin shifted by {row['estimated_margin_pct_change']:+.2f}%.
- **Wastage Impact:** Spoilage and waste costs experienced a {row['estimated_wastage_pct_change']:+.2f}% variance.
- **Net Bottom Line:** Bottom-line restaurant profitability experienced a net change of **${row['estimated_profitability_delta']:+,.2f}**.

---
"""

    md += """
1. **Price Elasticity Modeling:** Price increases and reductions dynamically incorporate empirical own-price elasticities (|ε|), penalizing price increases on elastic dishes through volume contraction.
2. **Promotion & Discount Dynamics:** Promotional simulations account for margin compression and inventory overproduction spoilage, preventing illusory promotional gains (SRS Traps 1 & 2).
3. **Wastage Flow-Through:** Reductions in daily preparation quantities directly reduce spoilage cost while maintaining fulfillment within empirical demand tolerances.
4. **Demand Substitution:** Item removals model category-level demand recapture, redirecting a portion of displaced patron orders into sibling menu items.

---
*Report generated automatically by DineIQ What-If Scenario Engine.*
"""
    return md


if __name__ == "__main__":
    run_what_if_pipeline()
