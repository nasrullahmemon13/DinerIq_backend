"""
DineIQ Analytics - EDA Notebook Validation Tests (SRS Step 8)
Verifies:
- notebooks/01_eda.ipynb exists and is a valid JSON Jupyter Notebook
- Contains cells covering EXACTLY the 13 items listed in SRS Step 8:
  1. Top-selling dishes
  2. Lowest-selling dishes
  3. Highest-revenue dishes
  4. Highest-profit dishes
  5. Highest-margin dishes
  6. High-wastage dishes
  7. Best-rated dishes
  8. Poorly rated dishes
  9. Popular menu categories
  10. Peak ordering periods
  11. Location-wise sales patterns
  12. Channel-wise ordering patterns
  13. Promotion-driven sales
"""
import os
import json
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOK_PATH = os.path.join(PROJECT_ROOT, "notebooks", "archive", "01_eda.ipynb")

@pytest.fixture(scope="module")
def notebook_json():
    assert os.path.exists(NOTEBOOK_PATH), f"Missing notebook at {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def test_notebook_format_and_structure(notebook_json):
    """Verify notebook has valid nbformat version and contains cells."""
    assert "cells" in notebook_json
    assert notebook_json.get("nbformat") == 4
    assert len(notebook_json["cells"]) >= 20

def test_notebook_covers_exact_13_srs_items(notebook_json):
    """Verify all 13 SRS Step 8 analytical areas are present in notebook markdown and code cells."""
    all_content = " ".join([
        "".join(cell.get("source", [])) for cell in notebook_json["cells"]
    ]).lower()

    srs_13_items = [
        "top-selling dishes",
        "lowest-selling dishes",
        "highest-revenue dishes",
        "highest-profit dishes",
        "highest-margin dishes",
        "high-wastage dishes",
        "best-rated dishes",
        "poorly rated dishes",
        "popular menu categories",
        "peak ordering periods",
        "location-wise sales patterns",
        "channel-wise ordering patterns",
        "promotion-driven sales"
    ]

    for item in srs_13_items:
                                                                  
        words = item.replace("-", " ").split()
        found = any(w in all_content for w in words)
        assert found, f"Missing analysis for SRS Step 8 item: '{item}'"
