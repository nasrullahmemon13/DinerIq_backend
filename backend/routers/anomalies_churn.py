"""
DineIQ Analytics - Anomalies & Customer Churn Watch Router
Implements SRS Step 36 / Functional Requirement (xxxvi):
- Sales anomalies (sudden spikes, deep drops, channel irregularities)
- Rating anomalies (ratings below expectation, sudden sentiment collapse)
- High-value customer churn risk prediction & retention interventions
"""

import os
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Query
import pandas as pd

router = APIRouter(prefix="/api/v1/analytics/anomalies-churn", tags=["Anomalies & Churn"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def load_parquet(folder: str, filename: str) -> pd.DataFrame:
    p = os.path.join(PROJECT_ROOT, "processed_data", folder, filename)
    if os.path.exists(p):
        try:
            return pd.read_parquet(p)
        except Exception as e:
            print(f"Error loading {p}: {e}")
    return pd.DataFrame()

@router.get("")
def get_anomalies_churn_data(
    churn_tier: Optional[str] = Query(None),
    anomaly_type: Optional[str] = Query(None),
    search: Optional[str] = Query(None)
) -> Dict[str, Any]:
    sa_df = load_parquet("anomalies", "sales_anomalies.parquet")
    if sa_df.empty:
        sa_df = load_parquet("anomaly", "sales_anomalies.parquet")

    ra_df = load_parquet("ratings", "rating_anomalies.parquet")
    if ra_df.empty:
        ra_df = load_parquet("anomaly", "rating_anomalies.parquet")

    cr_df = load_parquet("churn", "customer_churn_risk.parquet")
    hva_df = load_parquet("churn", "high_value_at_risk.parquet")

    sales_anomalies = sa_df.head(100).to_dict(orient="records") if not sa_df.empty else []
    rating_anomalies = ra_df.head(100).to_dict(orient="records") if not ra_df.empty else []

    customers_at_risk = []
    if not cr_df.empty:
        filtered_cr = cr_df.copy()
        if churn_tier and churn_tier != "All":
            filtered_cr = filtered_cr[filtered_cr["churn_risk_tier"] == churn_tier]
        if search:
            match_first = filtered_cr["first_name"].str.contains(search, case=False, na=False, regex=False)
            match_last = filtered_cr["last_name"].str.contains(search, case=False, na=False, regex=False)
            match_id = filtered_cr["customer_id"].str.contains(search, case=False, na=False, regex=False)
            filtered_cr = filtered_cr[match_first | match_last | match_id]

        cols_to_keep = [
            "customer_id", "first_name", "last_name", "email", "customer_segment",
            "loyalty_tier", "total_orders", "total_spend", "avg_order_value", "recency_days",
            "churn_risk_score", "churn_risk_tier", "primary_risk_driver", "recommended_retention_action"
        ]
        available_cols = [c for c in cols_to_keep if c in filtered_cr.columns]
        customers_at_risk = filtered_cr[available_cols].head(150).to_dict(orient="records")

    summary = {
        "total_sales_anomalies": len(sa_df),
        "total_rating_anomalies": len(ra_df),
        "total_high_risk_customers": int((cr_df["churn_risk_tier"] == "High Risk").sum()) if not cr_df.empty and "churn_risk_tier" in cr_df.columns else 0,
        "high_value_revenue_at_risk": float(hva_df["total_spend"].sum()) if not hva_df.empty and "total_spend" in hva_df.columns else 0.0
    }

    return {
        "status": "success",
        "summary": summary,
        "sales_anomalies": sales_anomalies,
        "rating_anomalies": rating_anomalies,
        "customers_at_risk": customers_at_risk
    }
