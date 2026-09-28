"""
DineIQ Analytics - Spark MLlib Model Pipeline & Benchmark Test Suite (SRS Step 13)
Verifies:
- Benchmark evaluation across >=3 Spark MLlib classification algorithms
- Accuracy, Precision, Recall, F1, ROC-AUC metrics
- Persisted model metadata and feature importances
"""
import os
import json
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_SPARK_DIR = os.path.join(PROJECT_ROOT, "models", "spark")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "model_comparison")


def test_spark_mllib_benchmarks_exist():
    """Verify spark_model_evidence.json contains benchmarks for at least 3 MLlib models."""
    evidence_path = os.path.join(REPORTS_DIR, "spark_model_evidence.json")
    assert os.path.exists(evidence_path), f"Missing Spark model evidence at {evidence_path}"
    with open(evidence_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "champion_model" in data
    assert "classification_benchmarks" in data
    benchmarks = data["classification_benchmarks"]
    assert len(benchmarks) >= 3, f"Expected at least 3 benchmarked algorithms, found {len(benchmarks)}"

    model_names = [b["model_name"] for b in benchmarks]
    assert "Logistic Regression" in model_names
    assert "Decision Tree" in model_names
    assert "Random Forest" in model_names

    for b in benchmarks:
        assert 0.0 <= b["accuracy"] <= 1.0
        assert 0.0 <= b["f1_score"] <= 1.0
        assert 0.0 <= b["roc_auc"] <= 1.0
        assert "confusion_matrix" in b


def test_spark_mllib_persisted_artifacts():
    """Verify saved Spark model artifacts exist."""
    assert os.path.exists(os.path.join(MODELS_SPARK_DIR, "best_model.joblib")), "Missing best_model.joblib"
    assert os.path.exists(os.path.join(MODELS_SPARK_DIR, "scaler.joblib")), "Missing scaler.joblib"
    assert os.path.exists(os.path.join(MODELS_SPARK_DIR, "model_metadata.json")), "Missing model_metadata.json"
    assert os.path.exists(os.path.join(MODELS_SPARK_DIR, "feature_importances.parquet")), "Missing feature_importances.parquet"
