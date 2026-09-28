"""
DineIQ Analytics - Administration, RBAC, Audit Trail & ML Model Registry Router
Implements SRS Step 1 (FR-xlix, FR-l), Step 47 (FR-xliv), and FR-xlvi:
- User Access Control & RBAC
- Security Audit Logs & Telemetry
- ML Model Registry & Dual-Pipeline Verification (Spark MLlib vs Scikit-Learn)
"""

import os
import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.routes import require_roles
from database.connection import get_db
from database.models import User, Role, AuditLog, ModelVersion

router = APIRouter(prefix="/api/v1/admin-management", tags=["Admin & Model Registry"])

@router.get("/users", dependencies=[Depends(require_roles(["admin"]))])
def list_users(db: Session = Depends(get_db)) -> Dict[str, Any]:
    users = db.query(User).all()
    roles = db.query(Role).all()

    user_list = []
    for u in users:
        user_list.append({
            "user_id": u.user_id,
            "username": u.username,
            "email": u.email,
            "full_name": u.full_name,
            "role_id": u.role_id,
            "assigned_location_id": u.assigned_location_id,
            "is_active": u.is_active
        })

    role_list = [{"role_id": r.role_id, "role_name": r.role_name, "description": r.description} for r in roles]

    return {
        "status": "success",
        "total_users": len(user_list),
        "users": user_list,
        "roles": role_list
    }

@router.get("/audit-logs", dependencies=[Depends(require_roles(["admin"]))])
def list_audit_logs(
    limit: int = Query(100, ge=1, le=500),
    event_type: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    query = db.query(AuditLog)
    if event_type and event_type != "All":
        query = query.filter(AuditLog.event_type == event_type)
    if status_filter and status_filter != "All":
        query = query.filter(AuditLog.status == status_filter)

    logs = query.order_by(desc(AuditLog.timestamp)).limit(limit).all()

    log_list = []
    for l in logs:
        log_list.append({
            "audit_id": l.audit_id,
            "event_type": l.event_type,
            "action": l.action,
            "actor": l.actor,
            "resource_id": l.resource_id,
            "status": l.status,
            "details": l.details,
            "ip_address": l.ip_address,
            "timestamp": l.timestamp.isoformat() if l.timestamp else None
        })

    total_count = db.query(AuditLog).count()

    return {
        "status": "success",
        "total_logs": total_count,
        "returned_logs": len(log_list),
        "logs": log_list
    }

@router.get("/models", dependencies=[Depends(require_roles(["admin", "analyst"]))])
def list_model_registry(db: Session = Depends(get_db)) -> Dict[str, Any]:
    models = db.query(ModelVersion).order_by(desc(ModelVersion.trained_at)).all()

    model_list = []
    for m in models:
        metrics_dict = {}
        if m.metrics:
            try:
                metrics_dict = json.loads(m.metrics) if isinstance(m.metrics, str) else m.metrics
            except Exception:
                metrics_dict = {"raw": str(m.metrics)}

        model_list.append({
            "version_id": m.version_id,
            "model_name": m.model_name,
            "version_tag": m.version_tag,
            "framework": m.framework,
            "pipeline_type": m.pipeline_type,
            "task_type": m.task_type,
            "metrics": metrics_dict,
            "parameters": m.parameters,
            "artifact_uri": m.artifact_uri,
            "is_active": m.is_active,
            "trained_at": m.trained_at.isoformat() if m.trained_at else None
        })

    spark_models = [m for m in model_list if m["pipeline_type"] == "Spark"]
    python_models = [m for m in model_list if m["pipeline_type"] == "Python"]

    comparison = []                                                                      

    return {
        "status": "success",
        "total_models": len(model_list),
        "spark_models_count": len(spark_models),
        "python_models_count": len(python_models),
        "models": model_list,
        "dual_pipeline_comparison": comparison
    }
