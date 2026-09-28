"""
DineIQ Analytics - Sales Anomaly Intelligence Package
Covers SRS Step 31 (Sales Anomaly Detection).
"""
from python_pipeline.anomalies.sales_anomaly import (
    SalesAnomalyDetector,
    run_sales_anomaly_pipeline
)

__all__ = [
    "SalesAnomalyDetector",
    "run_sales_anomaly_pipeline"
]
