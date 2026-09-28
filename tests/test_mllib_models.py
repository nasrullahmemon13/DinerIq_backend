"""
DineIQ Analytics - Machine Learning Model Comparison Tests (SRS Step 12)
Verifies:
- At least 3 algorithms from SRS list trained and benchmarked
- Model comparison evidence reports exist (markdown & json)
- Champion model artifact saved to models/spark/
- Serialized model can be loaded and generates valid inference predictions
- Evaluation metrics exceed baseline thresholds (AUC > 0.75, F1 > 0.70)
"""
import os
import json
import joblib
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "spark")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "model_comparison")

@pytest.fixture(scope="module")
def evidence_data():
    json_path = os.path.join(REPORTS_DIR, "spark_model_evidence.json")
    assert os.path.exists(json_path), f"Missing evidence json at {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def loaded_model():
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    assert os.path.exists(model_path), f"Missing champion model at {model_path}"
    return joblib.load(model_path)

def test_evidence_report_existence():
    """Verify spark_model_evidence.md exists and contains benchmarking table."""
    md_path = os.path.join(REPORTS_DIR, "spark_model_evidence.md")
    assert os.path.exists(md_path), f"Missing markdown evidence report at {md_path}"
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Algorithm Benchmarking Matrix" in content
    assert "Champion Model Selection Rationale" in content

def test_at_least_three_algorithms_compared(evidence_data):
    """SRS Requirement: Train and compare AT LEAST 3 algorithms from SRS list."""
    benchmarks = evidence_data.get("classification_benchmarks", [])
    model_names = [m["model_name"] for m in benchmarks]
    
    assert len(benchmarks) >= 3, f"Expected at least 3 models, got {len(benchmarks)}"
    
                                             
    srs_algo_pool = {
        "Logistic Regression", "Decision Tree", "Random Forest",
        "Gradient-Boosted Trees", "Linear Regression", "Generalized Linear Regression", "K-Means"
    }
    for m in model_names:
        assert any(algo in m for algo in srs_algo_pool), f"Model '{m}' not in SRS algorithm list"

def test_champion_model_metadata(evidence_data):
    """Verify champion model metadata has required performance metrics."""
    champion_name = evidence_data.get("champion_model")
    assert champion_name is not None
    
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    assert os.path.exists(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    metrics = meta["evaluation_metrics"]
    assert metrics["accuracy"] >= 0.70, "Champion accuracy must exceed 70%"
    assert metrics["roc_auc"] >= 0.75, "Champion ROC-AUC must exceed 0.75"
    assert metrics["f1_score"] >= 0.70, "Champion F1-score must exceed 0.70"

def test_model_inference_execution(loaded_model):
    """Verify the saved champion model produces valid binary predictions on synthetic input."""
                                                  
    dummy_input = np.array([
        [15.0, 5.0, 450.0, 90.0, 8.0, 10.0, 0.25, 0.50, 0.30, 180.0],
        [300.0, 1.0, 45.0, 45.0, 2.0, 0.0, 0.0, 0.10, 0.0, 120.0]
    ])
    preds = loaded_model.predict(dummy_input)
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})

    if hasattr(loaded_model, "predict_proba"):
        probs = loaded_model.predict_proba(dummy_input)
        assert probs.shape == (2, 2)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

def test_feature_importances_artifact():
    """Verify feature importances are saved in Parquet and CSV formats."""
    assert os.path.exists(os.path.join(MODELS_DIR, "feature_importances.parquet"))
    assert os.path.exists(os.path.join(MODELS_DIR, "feature_importances.csv"))
    df = pd.read_parquet(os.path.join(MODELS_DIR, "feature_importances.parquet"))
    assert len(df) == 10
    assert "importance_weight" in df.columns
