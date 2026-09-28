"""
DineIQ Analytics - Prescriptive Recommendation Evidence Test Suite (SRS Step 27)
Verifies:
- All recommendations contain mandatory fields: RECOMMENDATION, EVIDENCE, METRICS, BUSINESS REASON, PRIORITY
- Coverage of all required operational recommendation domains:
  Menu optimization, Inventory, Customer targeting, Promotions, Pricing, Wastage, Bundles
"""
import os
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RECS_DIR = os.path.join(PROJECT_ROOT, "processed_data", "recommendations")


def test_recommendation_evidence_completeness():
    """Verify that every generated recommendation contains supporting evidence, metrics, rationale, and priority."""
    pq_path = os.path.join(RECS_DIR, "recommendations.parquet")
    assert os.path.exists(pq_path), f"Missing {pq_path}"
    df = pd.read_parquet(pq_path)
    assert len(df) > 0, "Recommendations table is empty"

    required_fields = [
        "recommended_action",                          
        "formatted_evidence",                    
        "potential_business_impact",            
        "business_impact_rationale",                    
        "priority"                               
    ]
    for field in required_fields:
        assert field in df.columns, f"Missing required recommendation field: {field}"
        assert df[field].notna().all(), f"Found NaN values in recommendation field: {field}"

    valid_priorities = {"Critical", "High", "Medium", "Low"}
    assert set(df["priority"].unique()).issubset(valid_priorities), "Invalid priority levels found"


def test_recommendation_domain_coverage():
    """Verify recommendations cover menu optimization, inventory, customer targeting, promotions, pricing, wastage, bundles."""
    pq_path = os.path.join(RECS_DIR, "recommendations.parquet")
    df = pd.read_parquet(pq_path)
    
    categories = " ".join(df["category"].astype(str).tolist()).lower()
    actions = " ".join(df["recommended_action"].astype(str).tolist()).lower()
    combined_text = f"{categories} {actions}"

                   
    assert "low performer" in combined_text or "opportunity" in combined_text or "menu" in combined_text, "Missing menu optimization recommendations"
    assert "promotion" in combined_text or "promo" in combined_text, "Missing promotion recommendations"
    assert "customer" in combined_text or "segment" in combined_text or "vip" in combined_text, "Missing customer targeting recommendations"
    assert "wastage" in combined_text or "waste" in combined_text or "spoilage" in combined_text, "Missing wastage recommendations"
    assert "stock" in combined_text or "peak" in combined_text or "prep" in combined_text, "Missing inventory recommendations"
    assert "pricing" in combined_text or "price" in combined_text, "Missing pricing recommendations"
    assert "bundle" in combined_text or "frequently purchased" in combined_text, "Missing bundle recommendations"
