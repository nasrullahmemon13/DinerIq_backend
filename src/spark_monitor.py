"""
DineIQ Analytics - Spark Distributed Job Monitoring Service
Conforms to SRS Functional Requirement (lxv):
  "lxv. Spark Job Monitoring (display job status)"

Provides real-time visibility into Spark execution:
  - Job Status (SUBMITTED, RUNNING, COMPLETED, FAILED)
  - Execution Duration, Stages Completed / Total Stages
  - Records Processed & Throughput Metrics
  - Stage-level Diagnostic Errors and Driver Telemetry
"""

import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from database.models import SparkJob
from src.audit_logger import log_audit_event


def seed_default_spark_jobs(db: Session):
    """Initializes Spark job history with realistic distributed processing telemetry."""
    if db.query(SparkJob).count() > 0:
        return

    now = datetime.now(timezone.utc)
    sample_jobs = [
        {
            "job_id": "SPARK-JOB-101",
            "job_name": "Parquet Batch Partitioning (Year/Month)",
            "pipeline_type": "PySpark",
            "status": "COMPLETED",
            "stages_completed": 8,
            "total_stages": 8,
            "records_processed": 105420,
            "duration_seconds": 42.50,
            "metrics": json.dumps({"shuffle_bytes": "12.4 MB", "tasks": 32, "partitions_written": 24}),
            "error_message": None,
            "start_time": now - timedelta(hours=5),
            "end_time": now - timedelta(hours=5) + timedelta(seconds=42.5)
        },
        {
            "job_id": "SPARK-JOB-102",
            "job_name": "Customer RFM Distributed Aggregation",
            "pipeline_type": "PySpark SQL",
            "status": "COMPLETED",
            "stages_completed": 4,
            "total_stages": 4,
            "records_processed": 50000,
            "duration_seconds": 18.20,
            "metrics": json.dumps({"shuffle_bytes": "6.1 MB", "tasks": 16, "groupBy_keys": 4000}),
            "error_message": None,
            "start_time": now - timedelta(hours=4),
            "end_time": now - timedelta(hours=4) + timedelta(seconds=18.2)
        },
        {
            "job_id": "SPARK-JOB-103",
            "job_name": "MLlib Churn Decision Tree Classifier",
            "pipeline_type": "PySpark MLlib",
            "status": "COMPLETED",
            "stages_completed": 12,
            "total_stages": 12,
            "records_processed": 4000,
            "duration_seconds": 31.80,
            "metrics": json.dumps({"accuracy": 0.884, "trees_fitted": 20, "max_depth": 5}),
            "error_message": None,
            "start_time": now - timedelta(hours=3),
            "end_time": now - timedelta(hours=3) + timedelta(seconds=31.8)
        },
        {
            "job_id": "SPARK-JOB-104",
            "job_name": "Wastage Risk MLlib Evaluation",
            "pipeline_type": "PySpark MLlib",
            "status": "RUNNING",
            "stages_completed": 5,
            "total_stages": 8,
            "records_processed": 32000,
            "duration_seconds": 14.10,
            "metrics": json.dumps({"active_executors": 4, "pending_partitions": 3}),
            "error_message": None,
            "start_time": now - timedelta(minutes=10),
            "end_time": None
        },
        {
            "job_id": "SPARK-JOB-105",
            "job_name": "Hourly Real-Time Order Stream Validation",
            "pipeline_type": "Spark Structured Streaming",
            "status": "FAILED",
            "stages_completed": 2,
            "total_stages": 6,
            "records_processed": 1250,
            "duration_seconds": 9.40,
            "metrics": json.dumps({"processed_batches": 2}),
            "error_message": "PySpark Exception: SchemaMismatchError at field 'order_timestamp'. Expected ISO-8601 string, received empty token.",
            "start_time": now - timedelta(hours=1),
            "end_time": now - timedelta(hours=1) + timedelta(seconds=9.4)
        }
    ]

    for j in sample_jobs:
        db.add(SparkJob(
            job_id=j["job_id"],
            job_name=j["job_name"],
            pipeline_type=j["pipeline_type"],
            status=j["status"],
            stages_completed=j["stages_completed"],
            total_stages=j["total_stages"],
            records_processed=j["records_processed"],
            duration_seconds=j["duration_seconds"],
            metrics=j["metrics"],
            error_message=j["error_message"],
            start_time=j["start_time"],
            end_time=j["end_time"]
        ))
    db.commit()


def trigger_spark_job(
    db: Session,
    job_name: str,
    pipeline_type: str = "PySpark",
    total_stages: int = 5,
    records_processed: int = 25000,
    actor: str = "admin_user"
) -> SparkJob:
    """Simulates or launches a distributed Spark job with audit logging."""
    job_id = f"SPARK-JOB-{uuid.uuid4().hex[:6].upper()}"
    now = datetime.now(timezone.utc)

    new_job = SparkJob(
        job_id=job_id,
        job_name=job_name,
        pipeline_type=pipeline_type,
        status="RUNNING",
        stages_completed=1,
        total_stages=total_stages,
        records_processed=records_processed,
        duration_seconds=5.2,
        metrics=json.dumps({"executors": 4, "driver_memory": "4g", "partitions": 16}),
        start_time=now
    )
    db.add(new_job)
    db.commit()

    log_audit_event(
        db=db,
        event_type="SPARK_JOB",
        action="TRIGGER_SPARK_JOB",
        actor=actor,
        resource_id=job_id,
        status="RUNNING",
        details=f"Triggered Spark job '{job_name}' with {total_stages} stages."
    )

    db.refresh(new_job)
    return new_job
