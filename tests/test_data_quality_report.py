"""
Unit & Regression Tests for Data Quality Report (SRS Step 4)
Verifies:
- All 15 required issues are identified and cataloged
- dq_report.json schema and validity
- dq_report.md markdown structure and completeness
"""
import os
import json
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "data_quality")

REQUIRED_15_ISSUES = [
    "missing values",
    "duplicate orders",
    "duplicate order-line records",
    "invalid menu prices",
    "negative quantities",
    "invalid dates",
    "invalid ratings",
    "missing customer IDs",
    "missing menu IDs",
    "invalid restaurant IDs",
    "impossible wastage quantities",
    "incorrect discounts",
    "cancelled transactions",
    "inconsistent units",
    "invalid location references"
]

def test_dq_report_files_exist():
    """Verify both Markdown and JSON reports exist."""
    json_path = os.path.join(REPORTS_DIR, "dq_report.json")
    md_path = os.path.join(REPORTS_DIR, "dq_report.md")
    assert os.path.exists(json_path), f"Missing {json_path}"
    assert os.path.exists(md_path), f"Missing {md_path}"

def test_dq_report_json_content():
    """Verify JSON report structure and that all 15 issues are captured."""
    json_path = os.path.join(REPORTS_DIR, "dq_report.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "summary" in data
    assert "issues" in data
    assert data["metadata"]["total_issues_tracked"] == 15
    assert 0.0 <= data["metadata"]["overall_data_quality_score"] <= 100.0

    reported_issue_names = [issue["issue_name"] for issue in data["issues"]]
    for req_issue in REQUIRED_15_ISSUES:
        assert req_issue in reported_issue_names, f"Missing required issue: {req_issue}"

                                                     
    for issue in data["issues"]:
        assert issue["violation_count"] > 0, f"Issue {issue['issue_name']} has 0 violations"
        assert issue["total_records"] > 0
        assert 0.0 <= issue["error_rate_pct"] <= 100.0
        assert issue["severity"] in ["Critical", "High", "Medium", "Low"]

def test_dq_report_markdown_content():
    """Verify Markdown report contains summary and all 15 issues."""
    md_path = os.path.join(REPORTS_DIR, "dq_report.md")
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "DineIQ Analytics" in content
    assert "Overall Platform Data Quality Score" in content
    for req_issue in REQUIRED_15_ISSUES:
        assert req_issue in content, f"Issue '{req_issue}' not found in Markdown report"
