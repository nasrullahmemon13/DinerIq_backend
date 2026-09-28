"""
DineIQ Analytics - Model Version Tracking & Prediction Registry
Conforms to SRS Functional Requirements (lxi) and (lxii):
  "lxi. Database Storage (config, metadata, users, recommendations, results — securely stored)"
  "lxii. Model Version Tracking (every prediction tagged with model version)"

Ensures every analytical prediction is immutably tagged with:
  - Model Version ID (e.g. MV-CHURN-SPARK-V2)
  - Version Tag (e.g. v2.1.0-mllib)
  - Framework (e.g. PySpark MLlib, Scikit-Learn)
  - Pipeline Type (Spark vs Python)
  - Confidence Score, Predicted Value, and Timestamp
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from database.connection import SessionLocal
from database.models import ModelVersion, PredictionResult, SystemConfig


def seed_default_model_versions(db: Session):
    """Seeds baseline models for Spark MLlib and Python pipelines across all tasks."""
    if db.query(ModelVersion).count() > 0:
        return

    default_models = [
        {
            "version_id": "MV-CHURN-SPARK-V2",
            "model_name": "Customer Churn Classifier (Spark MLlib)",
            "version_tag": "v2.1.0-mllib",
            "framework": "PySpark MLlib",
            "pipeline_type": "Spark",
            "task_type": "Churn",
            "metrics": json.dumps({"accuracy": 0.884, "f1_score": 0.871, "auc_roc": 0.912}),
            "parameters": json.dumps({"maxDepth": 5, "numTrees": 20, "impurity": "gini"}),
            "artifact_uri": "models/spark/churn_random_forest_v2.model",
            "is_active": True
        },
        {
            "version_id": "MV-CHURN-PY-V1",
            "model_name": "Customer Churn Random Forest (Scikit-Learn)",
            "version_tag": "v1.2.0-sklearn",
            "framework": "Scikit-Learn",
            "pipeline_type": "Python",
            "task_type": "Churn",
            "metrics": json.dumps({"accuracy": 0.862, "f1_score": 0.850, "auc_roc": 0.895}),
            "parameters": json.dumps({"n_estimators": 100, "max_depth": 6, "criterion": "gini"}),
            "artifact_uri": "models/python/churn_rf_v1.pkl",
            "is_active": True
        },
        {
            "version_id": "MV-DEMAND-SPARK-V1",
            "model_name": "Location Demand Forecaster (Spark GBT)",
            "version_tag": "v1.5.0-mllib",
            "framework": "PySpark MLlib",
            "pipeline_type": "Spark",
            "task_type": "Demand",
            "metrics": json.dumps({"rmse": 14.32, "mae": 11.20, "r2": 0.879}),
            "parameters": json.dumps({"maxIter": 50, "stepSize": 0.1}),
            "artifact_uri": "models/spark/demand_gbt_v1.model",
            "is_active": True
        },
        {
            "version_id": "MV-DEMAND-PY-V1",
            "model_name": "Demand Forecasting Engine (Prophet/Statsmodels)",
            "version_tag": "v1.3.0-stats",
            "framework": "Statsmodels",
            "pipeline_type": "Python",
            "task_type": "Demand",
            "metrics": json.dumps({"rmse": 15.10, "mae": 12.05, "r2": 0.861}),
            "parameters": json.dumps({"seasonality_mode": "multiplicative", "order": [1, 1, 1]}),
            "artifact_uri": "models/python/demand_prophet_v1.pkl",
            "is_active": True
        },
        {
            "version_id": "MV-WASTAGE-SPARK-V1",
            "model_name": "Wastage Risk MLlib Predictor",
            "version_tag": "v1.1.0-mllib",
            "framework": "PySpark MLlib",
            "pipeline_type": "Spark",
            "task_type": "Wastage",
            "metrics": json.dumps({"accuracy": 0.895, "recall": 0.870}),
            "parameters": json.dumps({"threshold": 0.65}),
            "artifact_uri": "models/spark/wastage_risk_v1.model",
            "is_active": True
        },
        {
            "version_id": "MV-RECOM-PY-V1",
            "model_name": "Menu Optimization & Recommendation Engine",
            "version_tag": "v1.0.0-rules-ml",
            "framework": "Python Multi-Criteria",
            "pipeline_type": "Python",
            "task_type": "Recommendation",
            "metrics": json.dumps({"coverage": 0.94, "precision_at_k": 0.89}),
            "parameters": json.dumps({"min_confidence": 0.75, "min_lift": 1.2}),
            "artifact_uri": "models/python/recommendation_engine_v1.pkl",
            "is_active": True
        }
    ]

    for m in default_models:
        db.add(ModelVersion(
            version_id=m["version_id"],
            model_name=m["model_name"],
            version_tag=m["version_tag"],
            framework=m["framework"],
            pipeline_type=m["pipeline_type"],
            task_type=m["task_type"],
            metrics=m["metrics"],
            parameters=m["parameters"],
            artifact_uri=m["artifact_uri"],
            is_active=m["is_active"],
            trained_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc)
        ))

    configs = [
        ("pipeline.default_engine", "Spark", "PIPELINE", "Default distributed computation engine for batch pipelines", False),
        ("spark.master_url", "local[*]", "SPARK", "PySpark master cluster connection URI", False),
        ("spark.driver.memory", "4g", "SPARK", "Allocated Spark driver RAM", False),
        ("spark.executor.memory", "4g", "SPARK", "Allocated Spark worker executor RAM", False),
        ("storage.database_engine", "PostgreSQL/SQLite", "STORAGE", "Relational persistence tier for analytics and metadata", False),
        ("security.jwt_algorithm", "HS256", "SECURITY", "Hashing algorithm for auth session tokens", False),
        ("threshold.wastage_high_risk", "0.75", "THRESHOLDS", "Probability threshold for high-risk wastage alerts", False),
        ("threshold.churn_at_risk", "0.60", "THRESHOLDS", "Probability threshold for identifying at-risk diners", False),
    ]

    for key, val, cat, desc, secret in configs:
        if not db.query(SystemConfig).filter(SystemConfig.config_key == key).first():
            db.add(SystemConfig(
                config_key=key,
                config_value=val,
                category=cat,
                description=desc,
                is_secret=secret,
                updated_by="SystemInit",
                updated_at=datetime.now(timezone.utc)
            ))

    db.commit()


def get_active_model_version(db: Session, task_type: str, pipeline_type: str = "Spark") -> Optional[ModelVersion]:
    """Retrieves current active model version for a given task and pipeline."""
    return db.query(ModelVersion).filter(
        ModelVersion.task_type == task_type,
        ModelVersion.pipeline_type == pipeline_type,
        ModelVersion.is_active == True
    ).first()


def record_prediction(
    db: Session,
    model_version_id: str,
    task_type: str,
    entity_type: str,
    entity_id: str,
    predicted_value: Any,
    actual_value: Optional[Any] = None,
    confidence_score: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> PredictionResult:
    """
    Saves a prediction tagged immutably with the generating model version (SRS lxii).
    """
    pred_id = f"PRED-{uuid.uuid4().hex[:10].upper()}"
    pred_str = json.dumps(predicted_value) if isinstance(predicted_value, (dict, list)) else str(predicted_value)
    act_str = json.dumps(actual_value) if isinstance(actual_value, (dict, list)) else (str(actual_value) if actual_value is not None else None)
    meta_str = json.dumps(metadata) if metadata else None

    pred_record = PredictionResult(
        prediction_id=pred_id,
        model_version_id=model_version_id,
        task_type=task_type,
        entity_type=entity_type,
        entity_id=entity_id,
        predicted_value=pred_str,
        actual_value=act_str,
        confidence_score=confidence_score,
        metadata_json=meta_str,
        prediction_timestamp=datetime.now(timezone.utc)
    )
    db.add(pred_record)
    db.commit()
    db.refresh(pred_record)
    return pred_record
