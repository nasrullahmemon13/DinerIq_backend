"""
Unit and Integration Tests for DineIQ Market Basket Analysis & Association Rules (SRS Steps 17 & 18)
Validates:
- Step 17: Support, Confidence, Lift calculation and frequent itemset discovery
- Step 18: Association-rule backed commercial recommendations across all 4 categories:
  * Combo Meals
  * Cross-Sell Opportunities
  * Upsell Combinations
  * Frequently Paired Dishes
"""
import os
import json
import pytest
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASKET_DIR = os.path.join(PROJECT_ROOT, "processed_data", "basket_analysis")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "basket_analysis")

def test_basket_analysis_output_files_exist():
    """Verify all required parquet, csv, md, and json artifacts exist."""
    required_files = [
        os.path.join(BASKET_DIR, "association_rules.parquet"),
        os.path.join(BASKET_DIR, "association_rules.csv"),
        os.path.join(BASKET_DIR, "menu_recommendations.parquet"),
        os.path.join(BASKET_DIR, "menu_recommendations.csv"),
        os.path.join(REPORTS_DIR, "basket_analysis_report.md"),
        os.path.join(REPORTS_DIR, "basket_analysis_report.json")
    ]
    for file_path in required_files:
        assert os.path.exists(file_path), f"Missing required file: {file_path}"
        assert os.path.getsize(file_path) > 0, f"File is empty: {file_path}"

def test_association_rules_metrics():
    """Verify Step 17 metrics: Support, Confidence, Lift are properly bounded."""
    rules_path = os.path.join(BASKET_DIR, "association_rules.parquet")
    df = pd.read_parquet(rules_path)
    
                              
    required_cols = ["antecedent_name", "consequent_name", "support", "confidence", "lift"]
    for col in required_cols:
        assert col in df.columns, f"Missing required metric: {col}"
    
    assert len(df) > 0, "No association rules were mined."
    
                                
    assert (df["support"] > 0).all() and (df["support"] <= 1).all(), "Support out of (0, 1] range"
    assert (df["confidence"] > 0).all() and (df["confidence"] <= 1).all(), "Confidence out of (0, 1] range"
    assert (df["lift"] >= 1.0).all(), "Mined rules should exhibit positive affinity (Lift >= 1.0)"

def test_commercial_recommendations_categories():
    """Verify Step 18 produces all 4 SRS-mandated recommendation types."""
    recs_path = os.path.join(BASKET_DIR, "menu_recommendations.parquet")
    df = pd.read_parquet(recs_path)
    
    assert len(df) > 0, "No recommendations generated."
    assert "recommendation_type" in df.columns, "Missing recommendation_type column."
    
    rec_types = set(df["recommendation_type"].unique())
    expected_types = {
        "Combo Meal",
        "Cross-Sell Opportunity",
        "Upsell Combination",
        "Frequently Paired Dishes"
    }
    
    missing_types = expected_types - rec_types
    assert not missing_types, f"Missing required recommendation categories: {missing_types}"

def test_recommendations_have_rule_evidence():
    """Verify all recommendations carry empirical association rule metrics and commercial rationale."""
    recs_path = os.path.join(BASKET_DIR, "menu_recommendations.parquet")
    df = pd.read_parquet(recs_path)
    
    for _, row in df.iterrows():
        assert pd.notna(row["primary_item"]) and len(row["primary_item"]) > 0
        assert pd.notna(row["recommended_item"]) and len(row["recommended_item"]) > 0
        assert row["support"] > 0, "Recommendation must have valid support"
        assert row["confidence"] > 0, "Recommendation must have valid confidence"
        assert row["lift"] >= 1.0, "Recommendation must be backed by positive lift"
        assert len(str(row["commercial_rationale"]).strip()) > 10, "Rationale must be descriptive"

def test_basket_analysis_report_content():
    """Verify markdown and JSON reports contain summary counts and structured data."""
    md_path = os.path.join(REPORTS_DIR, "basket_analysis_report.md")
    json_path = os.path.join(REPORTS_DIR, "basket_analysis_report.json")
    
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    
    assert "Market Basket Analysis & Association Rules" in md_text
    assert "Top Association Rules by Lift" in md_text
    assert "Actionable Commercial Recommendations" in md_text
    assert "Combo Meal" in md_text
    assert "Cross-Sell Opportunity" in md_text
    assert "Upsell Combination" in md_text
    assert "Frequently Paired Dishes" in md_text
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "total_transactions_analyzed" in data
    assert "rules_mined_count" in data
    assert "recommendations_count" in data
    assert data["rules_mined_count"] > 0
    assert data["recommendations_count"] >= 4
