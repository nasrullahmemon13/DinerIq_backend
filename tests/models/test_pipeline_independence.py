"""
DineIQ Analytics - Pipeline Independence Verification Test Suite (SRS Step 13)
Verifies:
- Complete architectural and execution separation between Big Data (Spark) and Python ML pipelines
- Python pipeline scripts do not import or depend on Spark prediction files or Spark models
- Independent data loading directly from operational data marts
"""
import os
import sys
import re
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
PYTHON_PIPELINE_DIR = os.path.join(PROJECT_ROOT, "python_pipeline")
if PYTHON_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PYTHON_PIPELINE_DIR)


def test_python_scripts_do_not_import_spark_predictions():
    """Verify python_pipeline scripts do not load spark prediction or evidence files."""
    forbidden_patterns = [
        r"spark_model_evidence",
        r"pyspark\.ml",
        r"spark\.read\.parquet\(.*spark",
        r"models/spark"
    ]
    
    python_files = []
    for root, _, files in os.walk(PYTHON_PIPELINE_DIR):
        for f in files:
            if f.endswith(".py") and not f.startswith("dual_pipeline"):
                python_files.append(os.path.join(root, f))

    for py_file in python_files:
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
            for pat in forbidden_patterns:
                match = re.search(pat, content)
                assert not match, f"File {os.path.basename(py_file)} illegally references Spark prediction: {pat}"


def test_data_loader_independence():
    """Verify data_loader reads directly from cleaned operational stores, not Spark outputs."""
    from data_loader import load_operational_data
    data = load_operational_data()
    assert "orders" in data
    assert "order_items" in data
    assert "customers" in data
    assert "menu_items" in data
    assert len(data["menu_items"]) == 150
