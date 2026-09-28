"""
DineIQ Analytics - Dual Pipeline Comparison Tests (SRS Step 14)
Verifies:
- Dual pipeline comparison report exists at reports/model_comparison/dual_pipeline_comparison_report.md
- Covers AT LEAST 100 records (Deliverable #7)
- Contains EXACTLY the 8 required fields:
  1. Record ID
  2. Actual class or value
  3. Spark result
  4. Python result
  5. Match or mismatch
  6. Numerical difference (wherever applicable)
  7. Explanation of disagreement
  8. Overall agreement percentage
- Agreement percentage is calculated and falls within valid range
"""
import os
import json
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "model_comparison")
MD_PATH = os.path.join(REPORTS_DIR, "dual_pipeline_comparison_report.md")
JSON_PATH = os.path.join(REPORTS_DIR, "dual_pipeline_comparison_report.json")

@pytest.fixture(scope="module")
def comparison_data():
    assert os.path.exists(JSON_PATH), f"Missing JSON report at {JSON_PATH}"
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def markdown_content():
    assert os.path.exists(MD_PATH), f"Missing Markdown report at {MD_PATH}"
    with open(MD_PATH, "r", encoding="utf-8") as f:
        return f.read()

def test_reports_exist():
    """Verify both markdown and json comparison reports are present."""
    assert os.path.exists(MD_PATH)
    assert os.path.exists(JSON_PATH)

def test_at_least_100_records_covered(comparison_data):
    """Deliverable #7: Covering at least 100 records."""
    total_records = comparison_data.get("total_records_compared", 0)
    assert total_records >= 100, f"Expected >= 100 records, got {total_records}"
    assert len(comparison_data.get("records", [])) == total_records

def test_all_eight_srs_required_fields_present(markdown_content, comparison_data):
    """Verify all 8 exact SRS required fields are present in report and records."""
                                                 
    required_fields = [
        "Record ID",
        "Actual class or value",
        "Spark result",
        "Python result",
        "Match or mismatch",
        "Numerical difference",
        "Explanation of disagreement",
        "Overall Agreement Percentage"
    ]
    for field in required_fields:
        assert field.lower() in markdown_content.lower(), f"Missing required SRS field: '{field}'"

                                       
    first_rec = comparison_data["records"][0]
    expected_record_keys = {
        "Record ID",
        "Actual class or value",
        "Spark result",
        "Python result",
        "Match or mismatch",
        "Numerical difference (wherever applicable)",
        "Explanation of disagreement"
    }
    assert expected_record_keys.issubset(set(first_rec.keys()))

def test_overall_agreement_percentage_validity(comparison_data):
    """Verify overall agreement percentage is mathematically sound and realistic."""
    agreement_pct = comparison_data.get("overall_agreement_percentage")
    assert agreement_pct is not None
    assert 70.0 <= agreement_pct <= 100.0, f"Agreement percentage {agreement_pct}% out of expected bounds"

    total = comparison_data["total_records_compared"]
    matches = comparison_data["match_count"]
    mismatches = comparison_data["mismatch_count"]
    assert matches + mismatches == total
    assert round((matches / total) * 100, 2) == agreement_pct

def test_disagreement_explanations_populated(comparison_data):
    """Verify all mismatched items have detailed technical explanations."""
    mismatched = [r for r in comparison_data["records"] if r["Match or mismatch"] == "MISMATCH"]
    for rec in mismatched:
        exp = rec.get("Explanation of disagreement", "")
        assert len(exp) > 15, f"Disagreement explanation too short for {rec['Record ID']}"
        assert exp != "Consensus across both independent engines"
