"""
DineIQ Analytics - Data Governance, Quality, Pipeline & Big Data Router
Implements SRS Steps 2, 3, 4, 5, 6, 7 & 45:
- Data Ingestion & In-Memory Overview (Step 3 / FR-i)
- Data Quality Assessment & Profiling (Step 4 / FR-ii)
- Data Cleaning & Quarantine Engine (Step 5 / FR-vi)
- Data Integration & Multi-Source Joins (Step 6 / FR-vii)
- All 22 SRS-Defined Engineered Features (Step 7 / FR-viii, xix, xx)
- PySpark Cluster & Partition Management (Step 45 / FR-xli)
"""

import os
import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Query
import pandas as pd
from backend.services import governance_evidence as evidence

router = APIRouter(prefix="/api/v1/data-governance", tags=["Data Governance & Pipeline"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

@router.get("/ingestion-overview")
def get_ingestion_overview() -> Dict[str, Any]:
    return evidence.inventory()

@router.get("/quality-profile")
def get_quality_profile() -> Dict[str, Any]:
    return evidence.profile()

@router.get("/quarantine-manifest")
def get_quarantine_manifest() -> Dict[str, Any]:
    return evidence.quarantine()

@router.get("/integration-joins")
def get_integration_joins() -> Dict[str, Any]:
    return evidence.joins()

@router.get("/engineered-features")
def get_engineered_features() -> Dict[str, Any]:
    features = [
        {"feature_id": "F-01", "name": "Item revenue", "domain": "Menu / Financial", "type": "Continuous (Float)", "formula": "SUM(order_items.item_total)", "business_use": "Rank top grossing menu items"},
        {"feature_id": "F-02", "name": "Cost", "domain": "Menu / Financial", "type": "Continuous (Float)", "formula": "SUM(item_cost * quantity)", "business_use": "Cost of Goods Sold (COGS) tracking"},
        {"feature_id": "F-03", "name": "Contribution margin", "domain": "Menu / Financial", "type": "Continuous (Float)", "formula": "Item Revenue - Item Cost", "business_use": "Net cash generated per menu item"},
        {"feature_id": "F-04", "name": "Profit percentage", "domain": "Menu / Financial", "type": "Percentage", "formula": "(Contribution Margin / Revenue) * 100", "business_use": "Margin efficiency thresholding"},
        {"feature_id": "F-05", "name": "Order frequency", "domain": "Customer / Menu", "type": "Integer", "formula": "COUNT(DISTINCT order_id)", "business_use": "Velocity of dishes and customer ordering rate"},
        {"feature_id": "F-06", "name": "Item popularity", "domain": "Menu Intelligence", "type": "Percentile Rank", "formula": "SUM(quantity)", "business_use": "Volume Driver vs Hidden Opportunity quadrant"},
        {"feature_id": "F-07", "name": "Repeat-purchase rate", "domain": "Customer Loyalty", "type": "Percentage", "formula": "Repeat non-guest buyers / all non-guest buyers", "business_use": "Detect high-retention anchor dishes"},
        {"feature_id": "F-08", "name": "Average rating", "domain": "Guest Experience", "type": "Continuous (1.0-5.0)", "formula": "AVG(overall_rating)", "business_use": "Dish satisfaction tracking"},
        {"feature_id": "F-09", "name": "Rating trend", "domain": "Guest Experience", "type": "Categorical Slope", "formula": "AVG(newest 30% ratings) - AVG(oldest 30% ratings)", "business_use": "Early warning for declining food quality"},
        {"feature_id": "F-10", "name": "Wastage percentage", "domain": "Operations / Waste", "type": "Percentage", "formula": "Wasted units / (sold units + wasted units) * 100", "business_use": "Kitchen efficiency & overprep identification"},
        {"feature_id": "F-11", "name": "Promotion dependency", "domain": "Marketing", "type": "Percentage", "formula": "Promotional orders / total orders", "business_use": "Promotion trap & discount fatigue detection"},
        {"feature_id": "F-12", "name": "Discount percentage", "domain": "Marketing", "type": "Percentage", "formula": "(Discount Amount / Gross Total) * 100", "business_use": "Channel discount margin erosion monitoring"},
        {"feature_id": "F-13", "name": "Customer recency", "domain": "Customer RFM", "type": "Integer (Days)", "formula": "DATEDIFF(dataset maximum order date, last customer order date)", "business_use": "Churn propensity scoring"},
        {"feature_id": "F-14", "name": "Customer frequency", "domain": "Customer RFM", "type": "Integer", "formula": "COUNT(order_id) per customer", "business_use": "Customer lifecycle segmentation"},
        {"feature_id": "F-15", "name": "Customer monetary value", "domain": "Customer RFM", "type": "Continuous (Float)", "formula": "SUM(order_total) per customer", "business_use": "VIP and High-Value Customer identification"},
        {"feature_id": "F-16", "name": "Average order value (AOV)", "domain": "Sales / Channels", "type": "Continuous (Float)", "formula": "Total Revenue / Total Orders", "business_use": "Basket expansion benchmarking"},
        {"feature_id": "F-17", "name": "Peak-hour frequency", "domain": "Operations / Timing", "type": "Integer", "formula": "Orders in hours 12-14 or 18-21 / total orders", "business_use": "Kitchen and courier shift allocation"},
        {"feature_id": "F-18", "name": "Weekend-order ratio", "domain": "Sales Dynamics", "type": "Percentage", "formula": "Saturday/Sunday orders / total orders", "business_use": "Weekday vs Weekend prep scaling"},
        {"feature_id": "F-19", "name": "Location performance", "domain": "Multi-Store", "type": "Composite Index", "formula": "Mean order total at customer most-visited location", "business_use": "Store benchmarking & Tier assignment"},
        {"feature_id": "F-20", "name": "Channel preference", "domain": "Fulfillment", "type": "Categorical", "formula": "MODE(channel) per customer/store", "business_use": "Targeted marketing & delivery optimization"},
        {"feature_id": "F-21", "name": "Basket size", "domain": "Market Basket", "type": "Continuous (Units)", "formula": "AVG(items_count_per_order)", "business_use": "Cross-sell bundle propensity"},
        {"feature_id": "F-22", "name": "Price-change percentage", "domain": "Pricing Elasticity", "type": "Percentage", "formula": "((New Price - Old Price) / Old Price) * 100", "business_use": "Empirical elasticity coefficient computation"}
    ]

    feature_keys = ['item_revenue','cost','contribution_margin','profit_percentage','order_frequency',
                    'item_popularity','repeat_purchase_rate','average_rating','rating_trend','wastage_percentage',
                    'promotion_dependency','discount_percentage','customer_recency','customer_frequency',
                    'customer_monetary_value','average_order_value','peak_hour_frequency','weekend_order_ratio',
                    'location_performance','channel_preference','basket_size','price_change_percentage']
    marts = {}
    for name in ['menu_features','customer_features','rfm_features']:
        path = os.path.join(PROJECT_ROOT, 'parquet_data', 'features', name + '.parquet')
        if os.path.isfile(path):
            marts[name] = pd.read_parquet(path)
    for feature, key in zip(features, feature_keys):
        feature['column'] = key
        feature['status'] = 'unavailable'
        for mart, frame in marts.items():
            if key in frame:
                column = frame[key]
                feature.update(status='available', source_dataset=mart, type=str(column.dtype),
                               records=len(frame), null_count=int(column.isna().sum()),
                               sample=json.loads(column.dropna().head(3).to_json(orient='values')))
                feature['business_use'] += f" | {len(frame):,} records; {int(column.isna().sum()):,} nulls"
                break
    return {
        "status": "success",
        "available_features": sum(f['status'] == 'available' for f in features),
        "total_features": len(features),
        "source_step": "Derived analytics features",
        "features": features
    }

@router.get("/spark-cluster")
def get_spark_cluster() -> Dict[str, Any]:
    return evidence.spark_status()
