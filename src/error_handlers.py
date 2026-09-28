"""
DineIQ Analytics - Centralized Error Handling Framework
Conforms to SRS Functional Requirement (lxiv):
  "lxiv. Error Handling (understandable errors for processing, model, Spark, database failures)"

Provides clear, human-understandable error messages and structured responses for:
  1. Processing Failures (data ingestion, quarantine, ETL transforms)
  2. Model Failures (inference errors, missing weights, feature dimension mismatch)
  3. Spark Failures (cluster connection, stage failures, memory overflow, job abort)
  4. Database Failures (connection drops, lock timeouts, constraint violations)
"""

import uuid
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from database.connection import SessionLocal
from database.models import AuditLog


class DineIQException(Exception):
    """Base exception for all DineIQ platform domain errors."""
    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        category: str = "SYSTEM",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        suggested_action: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.status_code = status_code
        self.details = details or {}
        self.suggested_action = suggested_action or "Please contact your DineIQ system administrator."


class ProcessingError(DineIQException):
    """Raised when data pipeline, ETL ingestion, or data-cleaning encounters an error."""
    def __init__(
        self,
        message: str,
        stage: str = "ETL_PROCESSING",
        details: Optional[Dict[str, Any]] = None,
        suggested_action: Optional[str] = None
    ):
        full_details = {"stage": stage, **(details or {})}
        action = suggested_action or "Verify the raw data format, column headers, and schema integrity."
        super().__init__(
            message=f"Data Processing Error [{stage}]: {message}",
            error_code="DATA_PROCESSING_FAILURE",
            category="PROCESSING",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=full_details,
            suggested_action=action
        )


class ModelError(DineIQException):
    """Raised when an ML model prediction, inference pipeline, or training fails."""
    def __init__(
        self,
        message: str,
        model_name: str = "PredictiveModel",
        model_version: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggested_action: Optional[str] = None
    ):
        full_details = {"model_name": model_name, "model_version": model_version, **(details or {})}
        action = suggested_action or "Check model registry weights, active version tags, and input feature columns."
        super().__init__(
            message=f"Model Inference Failure [{model_name}]: {message}",
            error_code="MODEL_INFERENCE_FAILURE",
            category="MODEL",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=full_details,
            suggested_action=action
        )


class SparkExecutionError(DineIQException):
    """Raised when a Spark job, driver stage, or PySpark context fails."""
    def __init__(
        self,
        message: str,
        job_id: Optional[str] = None,
        stage_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        suggested_action: Optional[str] = None
    ):
        full_details = {"spark_job_id": job_id, "spark_stage": stage_id, **(details or {})}
        action = suggested_action or "Verify Spark master connectivity, driver memory allocation, or partition distribution."
        super().__init__(
            message=f"Spark Distributed Execution Error: {message}",
            error_code="SPARK_JOB_FAILED",
            category="SPARK",
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=full_details,
            suggested_action=action
        )


class DatabaseOperationError(DineIQException):
    """Raised when database query execution, connection pool, or relational constraint fails."""
    def __init__(
        self,
        message: str,
        operation: str = "QUERY_EXECUTION",
        table_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggested_action: Optional[str] = None
    ):
        full_details = {"operation": operation, "table": table_name, **(details or {})}
        action = suggested_action or "Check database connection parameters, transaction locks, and relational constraints."
        super().__init__(
            message=f"Database Operation Error [{operation}]: {message}",
            error_code="DATABASE_FAILURE",
            category="DATABASE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=full_details,
            suggested_action=action
        )


def _log_error_to_audit_trail(category: str, error_code: str, message: str, details: Dict[str, Any], actor: str = "System"):
    """Persists unhandled/critical errors to audit_log table without raising secondary exceptions."""
    try:
        with SessionLocal() as db:
            audit_entry = AuditLog(
                audit_id=f"ERR-{uuid.uuid4().hex[:10].upper()}",
                event_type="SYSTEM_ERROR",
                action=f"ERROR_{error_code}",
                actor=actor,
                resource_id=category,
                status="FAILED",
                details=f"Message: {message} | Details: {str(details)[:800]}",
                timestamp=datetime.now(timezone.utc)
            )
            db.add(audit_entry)
            db.commit()
    except Exception:
        pass


def register_error_handlers(app: FastAPI):
    """Registers comprehensive, understandable error handlers on FastAPI application."""

    @app.exception_handler(DineIQException)
    async def handle_dineiq_domain_exception(request: Request, exc: DineIQException):
        trace_id = f"TRC-{uuid.uuid4().hex[:8].upper()}"
        actor = request.headers.get("X-Username", "AnonymousUser")

        _log_error_to_audit_trail(
            category=exc.category,
            error_code=exc.error_code,
            message=exc.message,
            details={"trace_id": trace_id, "path": str(request.url), **exc.details},
            actor=actor
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "detail": exc.message,
                "category": exc.category,
                "error_code": exc.error_code,
                "message": exc.message,
                "suggested_action": exc.suggested_action,
                "details": exc.details,
                "trace_id": trace_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        trace_id = f"VAL-{uuid.uuid4().hex[:8].upper()}"
        readable_errors = []
        for err in exc.errors():
            field = " -> ".join(str(loc) for loc in err.get("loc", []))
            readable_errors.append(f"Field '{field}': {err.get('msg')}")

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "error",
                "detail": readable_errors,
                "category": "VALIDATION",
                "error_code": "REQUEST_VALIDATION_ERROR",
                "message": "Input validation failed. Please check the requested parameters and format.",
                "suggested_action": "Verify input data types, required fields, and values according to API documentation.",
                "details": {"validation_errors": readable_errors},
                "trace_id": trace_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException):
        category = "AUTH" if exc.status_code in (401, 403) else ("CLIENT" if exc.status_code < 500 else "SERVER")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "detail": str(exc.detail),
                "category": category,
                "error_code": f"HTTP_{exc.status_code}",
                "message": str(exc.detail),
                "suggested_action": "Ensure required authentication headers and valid permissions are supplied.",
                "details": {"status_code": exc.status_code, "path": str(request.url.path)},
                "trace_id": f"HTTP-{uuid.uuid4().hex[:8].upper()}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    @app.exception_handler(Exception)
    async def handle_generic_exception(request: Request, exc: Exception):
        trace_id = f"FATAL-{uuid.uuid4().hex[:8].upper()}"
        actor = request.headers.get("X-Username", "AnonymousUser")

        _log_error_to_audit_trail(
            category="INTERNAL",
            error_code="UNHANDLED_EXCEPTION",
            message=str(exc),
            details={"trace_id": trace_id, "path": str(request.url), "traceback": traceback.format_exc()[:500]},
            actor=actor
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "detail": str(exc),
                "category": "INTERNAL",
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": f"An unexpected system exception occurred: {str(exc)}",
                "suggested_action": "Please contact system engineering with the trace ID provided below.",
                "details": {"exception_type": type(exc).__name__},
                "trace_id": trace_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
