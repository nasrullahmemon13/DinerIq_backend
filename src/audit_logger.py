"""
DineIQ Analytics - Enterprise Audit Trail Framework
Conforms to SRS Functional Requirement (lxiii):
  "lxiii. Audit Trail (log all data-processing jobs, predictions, exports, admin actions)"

Maintains an immutable record of:
  1. Data-Processing Jobs (ETL, partitioning, cleaning)
  2. Predictions (MLlib/Python model inferences)
  3. Data Exports (CSV, Excel generation)
  4. Admin Actions (Location changes, user edits, threshold updates)
  5. Authentication & System Errors
"""

import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from database.models import AuditLog


def log_audit_event(
    db: Session,
    event_type: str,                                                                                                                  
    action: str,
    actor: str,
    resource_id: Optional[str] = None,
    status: str = "SUCCESS",                                     
    details: Optional[Any] = None,
    ip_address: Optional[str] = "127.0.0.1"
) -> AuditLog:
    """Creates and persists an audit trail log entry."""
    audit_id = f"AUD-{uuid.uuid4().hex[:10].upper()}"
    det_str = json.dumps(details) if isinstance(details, (dict, list)) else (str(details) if details else None)

    entry = AuditLog(
        audit_id=audit_id,
        event_type=event_type,
        action=action,
        actor=actor,
        resource_id=resource_id,
        status=status,
        details=det_str,
        ip_address=ip_address,
        timestamp=datetime.now(timezone.utc)
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def seed_default_audit_trail(db: Session):
    """Populates baseline audit trail records across all 4 mandatory SRS categories."""
    if db.query(AuditLog).count() > 0:
        return

    now = datetime.now(timezone.utc)
    seed_records = [
        {
            "audit_id": "AUD-JOB-001",
            "event_type": "DATA_PROCESSING_JOB",
            "action": "RUN_SPARK_DATA_PARTITIONING",
            "actor": "admin_user",
            "resource_id": "orders_by_year_month",
            "status": "SUCCESS",
            "details": "Partitioned 100,000+ orders into Parquet storage across 24 monthly partitions.",
            "timestamp": now - timedelta(hours=8)
        },
        {
            "audit_id": "AUD-JOB-002",
            "event_type": "DATA_PROCESSING_JOB",
            "action": "RUN_CLEANING_AND_QUARANTINE",
            "actor": "system_scheduler",
            "resource_id": "raw_orders_stream",
            "status": "SUCCESS",
            "details": "Cleaned raw orders dataset. 4 records moved to quarantine due to negative amount.",
            "timestamp": now - timedelta(hours=6)
        },
        {
            "audit_id": "AUD-PRED-001",
            "event_type": "PREDICTION",
            "action": "BATCH_CHURN_PREDICTION",
            "actor": "data_analyst",
            "resource_id": "MV-CHURN-SPARK-V2",
            "status": "SUCCESS",
            "details": "Predicted churn risk for 4,000 loyalty customers using PySpark MLlib DecisionTree.",
            "timestamp": now - timedelta(hours=4)
        },
        {
            "audit_id": "AUD-PRED-002",
            "event_type": "PREDICTION",
            "action": "DEMAND_FORECAST_RUN",
            "actor": "system_worker",
            "resource_id": "MV-DEMAND-SPARK-V1",
            "status": "SUCCESS",
            "details": "Executed 30-day demand forecast for top 10 restaurant locations.",
            "timestamp": now - timedelta(hours=3)
        },
        {
            "audit_id": "AUD-EXP-001",
            "event_type": "DATA_EXPORT",
            "action": "EXPORT_EXECUTIVE_SUMMARY_CSV",
            "actor": "regional_mgr",
            "resource_id": "report_exec_summary.csv",
            "status": "SUCCESS",
            "details": "Exported executive metrics dataset containing 10 location summaries.",
            "timestamp": now - timedelta(hours=2)
        },
        {
            "audit_id": "AUD-EXP-002",
            "event_type": "DATA_EXPORT",
            "action": "EXPORT_WASTAGE_ANALYSIS_EXCEL",
            "actor": "store_mgr",
            "resource_id": "wastage_analysis_loc001.xlsx",
            "status": "SUCCESS",
            "details": "Exported item-level wastage report with cost indices.",
            "timestamp": now - timedelta(hours=1)
        },
        {
            "audit_id": "AUD-ADM-001",
            "event_type": "ADMIN_ACTION",
            "action": "CREATE_RESTAURANT_LOCATION",
            "actor": "admin_user",
            "resource_id": "LOC-004",
            "status": "SUCCESS",
            "details": "Provisioned new location Dallas Midtown Branch with 150 seating capacity.",
            "timestamp": now - timedelta(minutes=45)
        },
        {
            "audit_id": "AUD-ADM-002",
            "event_type": "ADMIN_ACTION",
            "action": "UPDATE_THRESHOLD_PARAMETERS",
            "actor": "admin_user",
            "resource_id": "threshold.wastage_high_risk",
            "status": "SUCCESS",
            "details": "Adjusted high-risk wastage threshold from 0.70 to 0.75.",
            "timestamp": now - timedelta(minutes=15)
        }
    ]

    for rec in seed_records:
        db.add(AuditLog(
            audit_id=rec["audit_id"],
            event_type=rec["event_type"],
            action=rec["action"],
            actor=rec["actor"],
            resource_id=rec["resource_id"],
            status=rec["status"],
            details=rec["details"],
            ip_address="127.0.0.1",
            timestamp=rec["timestamp"]
        ))
    db.commit()
