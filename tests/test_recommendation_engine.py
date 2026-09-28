"""
Unit and Integration Tests for SRS Steps 37, 38, and 39:
- Step 37: Evidence-Based Recommendation Engine (All 9 SRS categories)
- Step 38: Recommendation Evidence (Action + Reason bullets format)
- Step 39: Recommendation Priority (Critical, High, Medium, Low based on business impact)
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from python_pipeline.recommendation.recommendation_engine import (
    DineIQRecommendationEngine,
    run_recommendation_pipeline
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "recommendations")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "recommendations")


@pytest.fixture(scope="module")
def engine():
    return DineIQRecommendationEngine()


@pytest.fixture(scope="module")
def recommendations(engine):
    recs_df = engine.generate_all_recommendations()
    prio_df = engine.summarize_by_priority()
    cat_df = engine.summarize_by_category()
    return recs_df, prio_df, cat_df


def test_engine_initialization(engine):
    """Assert DineIQRecommendationEngine initializes and loads prerequisite domain datasets."""
    assert engine is not None
    assert not engine.menu_class.empty
    assert not engine.wastage_item.empty
    assert not engine.pricing.empty
    assert not engine.basket.empty
    assert not engine.slow_moving.empty
    assert not engine.churn.empty
    assert not engine.promo_trap.empty
    assert not engine.loc_matrix.empty


def test_all_9_srs_step_37_categories_present(recommendations):
    """
    SRS Step 37 requires generating recommendations from EXACTLY this list:
    1. Promote high-margin Hidden Opportunities
    2. Reduce preparation quantity of high-wastage dishes
    3. Review pricing of price-sensitive dishes
    4. Bundle frequently purchased items
    5. Remove or redesign persistent Low Performers
    6. Increase stock before predicted peak periods
    7. Target selected customer segments
    8. Review ineffective promotions
    9. Investigate anomalous locations
    """
    recs_df, _, cat_df = recommendations

    expected_categories = {
        "Promote high-margin Hidden Opportunities",
        "Reduce preparation quantity of high-wastage dishes",
        "Review pricing of price-sensitive dishes",
        "Bundle frequently purchased items",
        "Remove or redesign persistent Low Performers",
        "Increase stock before predicted peak periods",
        "Target selected customer segments",
        "Review ineffective promotions",
        "Investigate anomalous locations"
    }

    actual_categories = set(recs_df["category"].unique())
    assert actual_categories == expected_categories
    assert len(cat_df) == 9


    for cat in expected_categories:
        count = (recs_df["category"] == cat).sum()
        assert count > 0, f"Category '{cat}' has 0 recommendations"


def test_recommendation_evidence_format_step_38(recommendations):
    """
    SRS Step 38 requirement:
    Every recommendation must display analytical evidence supporting it in format:
    Recommended Action:
      <Action>

    Reason:
      * <Evidence 1>
      * <Evidence 2>

    Explicit rule: 'The application must not provide unexplained recommendations.'
    """
    recs_df, _, _ = recommendations

    for _, row in recs_df.iterrows():

        action = row["recommended_action"]
        assert isinstance(action, str)
        assert len(action.strip()) > 10


        bullets = row["reason_bullets"]
        assert isinstance(bullets, (list, np.ndarray))
        assert len(bullets) >= 3, f"Rec {row['recommendation_id']} has fewer than 3 evidence bullets"

        for b in bullets:
            assert isinstance(b, str)
            assert len(b.strip()) > 5


        evidence_str = row["formatted_evidence"]
        assert "Recommended Action:" in evidence_str
        assert "Reason:" in evidence_str
        assert "*" in evidence_str


def test_recommendation_priority_levels_step_39(recommendations):
    """
    SRS Step 39 requirement:
    Recommendations should be assigned a priority such as:
      Low, Medium, High, Critical
    Priority should be based on potential business impact.
    """
    recs_df, prio_df, _ = recommendations

    expected_priorities = {"Critical", "High", "Medium", "Low"}
    actual_priorities = set(recs_df["priority"].unique())
    assert actual_priorities == expected_priorities


    assert (recs_df["potential_business_impact"] > 0).all()


    crit_impact = prio_df[prio_df["priority"] == "Critical"]["avg_potential_business_impact"].iloc[0]
    med_impact = prio_df[prio_df["priority"] == "Medium"]["avg_potential_business_impact"].iloc[0]
    assert crit_impact > med_impact


def test_high_impact_critical_recommendations(recommendations):
    """Verify that extreme hazards (severe wastage, VIP churn, negative promo) receive Critical priority."""
    recs_df, _, _ = recommendations

    critical_recs = recs_df[recs_df["priority"] == "Critical"]
    assert len(critical_recs) >= 5


    vip_recs = critical_recs[critical_recs["target_entity_type"] == "Customer Segment"]
    assert len(vip_recs) > 0


def test_pipeline_execution_and_file_artifacts():
    """Assert pipeline runs and persists all parquet, CSV, JSON, and Markdown artifacts."""
    recs_df, priority_summary, category_summary, stats = run_recommendation_pipeline()

    assert stats["total_recommendations"] == len(recs_df)
    assert stats["srs_categories_covered"] == 9
    assert stats["total_potential_business_impact"] > 1000000.0


    assert os.path.exists(os.path.join(OUTPUT_DIR, "recommendations.parquet"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "recommendations.csv"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "recommendation_priority_summary.parquet"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "recommendation_priority_summary.csv"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "recommendation_category_summary.parquet"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "recommendation_category_summary.csv"))


    assert os.path.exists(os.path.join(REPORTS_DIR, "recommendation_engine_report.md"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "recommendation_engine_summary.json"))


    report_size = os.path.getsize(os.path.join(REPORTS_DIR, "recommendation_engine_report.md"))
    assert report_size > 2000
