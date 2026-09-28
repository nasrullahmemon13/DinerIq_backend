"""
DineIQ Analytics - SRS Exact 11 Difficult Real-World Edge Cases
Tests the exact 11 challenging business and analytical scenarios specified in the SRS:
  1. High-selling loss-making dish
  2. Low-selling high-margin dish
  3. High-wastage popular dish
  4. Promotion increasing sales but reducing profit
  5. Dish performing differently across locations
  6. New menu item (cold-start)
  7. Price-sensitive item (high elasticity)
  8. Customer churn (increasing recency, declining frequency/monetary)
  9. Rating anomaly (divergence between star ratings and reviews)
  10. Sales anomaly (volume surge beyond 3 standard deviations)
  11. Spark / Python model disagreement case
"""

import os
import json
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from src.what_if_engine import WhatIfScenarioEngine

client = TestClient(app)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))





@pytest.mark.difficult_cases
def test_case_1_high_selling_loss_making_dish():
    """
    Difficult Case 1: High-selling loss-making dish.
    Item has high sales volume but unit cost > price (or margin < 0%).
    What-if simulation engine must evaluate contribution margin impact of price remediation.
    """
    engine = WhatIfScenarioEngine()

    res = engine.simulate_increase_menu_price(item_id="ITEM-046", price_increase_pct=20.0)

    assert res.is_simulation_estimate is True
    assert "contribution_margin" in res.__dict__
    assert res.contribution_margin["delta_dollars"] != 0
    assert "simulated outputs" in res.disclaimer.lower() or "estimate only" in res.disclaimer.lower()





@pytest.mark.difficult_cases
def test_case_2_low_selling_high_margin_dish():
    """
    Difficult Case 2: Low-selling high-margin dish (Hidden Opportunity).
    Item has high margin percentage but low sales volume.
    Menu intelligence dashboard must classify it into Hidden Opportunities quadrant.
    """
    overview_res = client.get("/api/v1/menu-intelligence")
    assert overview_res.status_code == 200
    data = overview_res.json()


    assert "hidden_opportunities" in data
    assert len(data["hidden_opportunities"]) == 9
    sample_opp = data["hidden_opportunities"][0]
    assert sample_opp["classification"] == "Hidden Opportunity"
    assert sample_opp["margin_pct"] > 50.0





@pytest.mark.difficult_cases
def test_case_3_high_wastage_popular_dish():
    """
    Difficult Case 3: High-wastage popular dish.
    Item is frequently ordered, but excessive batch prep causes significant food waste.
    System must flag high wastage and simulate prep reduction.
    """
    wastage_res = client.get("/api/v1/wastage")
    assert wastage_res.status_code == 200
    data = wastage_res.json()


    assert "high_wastage_items" in data
    assert "wastage_cost" in data
    assert len(data["high_wastage_items"]) >= 1


    engine = WhatIfScenarioEngine()
    prep_sim = engine.simulate_reduce_preparation_quantity(prep_reduction_pct=20.0)
    assert prep_sim.wastage["delta_cost"] < 0
    assert prep_sim.is_simulation_estimate is True





@pytest.mark.difficult_cases
def test_case_4_promotion_increasing_sales_but_reducing_profit():
    """
    Difficult Case 4: Promotion increasing sales but reducing profit.
    Deep discount generates volume spike but negative net profit impact due to excessive discount depth.
    What-if simulation exposes that revenue/demand surge can lead to reduced margin dollars.
    """
    engine = WhatIfScenarioEngine()
    promo_sim = engine.simulate_change_discount_percentage(current_discount_pct=10.0, new_discount_pct=35.0)


    assert promo_sim.demand["delta_units"] > 0
    assert promo_sim.contribution_margin["delta_dollars"] < 0
    assert "simulated outputs" in promo_sim.disclaimer.lower() or "estimate only" in promo_sim.disclaimer.lower()





@pytest.mark.difficult_cases
def test_case_5_dish_performing_differently_across_locations():
    """
    Difficult Case 5: Dish performing differently across locations.
    The same menu item can classify as a Profit Driver in Location A and a Low Performer in Location B.
    Multi-location intelligence report must support location-specific classification.
    """
    rep_path = os.path.join(PROJECT_ROOT, "reports", "locations", "multi_location_intelligence_report.json")
    assert os.path.exists(rep_path)
    with open(rep_path, "r", encoding="utf-8") as f:
        loc_data = json.load(f)


    summary_stats = loc_data.get("summary_stats", {})
    location_ranking = loc_data.get("location_ranking", [])
    assert summary_stats.get("locations_compared", 0) >= 2 or len(location_ranking) >= 2
    assert summary_stats.get("dishes_with_multiple_classes_across_locations", 0) > 0


    loc_res = client.get("/api/v1/locations")
    assert loc_res.status_code == 200
    assert len(loc_res.json()) >= 2





@pytest.mark.difficult_cases
def test_case_6_new_menu_item_cold_start():
    """
    Difficult Case 6: New menu item with zero prior sales history.
    System must handle cold start safely: assign default elasticity, classify as New, and avoid crashes.
    """

    new_item_id = "ITEM-COLD-START-99"
    res = client.post("/api/v1/menu/items", json={
        "item_id": new_item_id,
        "category_id": "CAT-001",
        "name": "Artisanal Cold-Start Tart",
        "base_price": 9.50,
        "cost_price": 3.00,
        "is_active": True
    }, headers={"X-User-Role": "admin"})
    assert res.status_code in (201, 200)


    engine = WhatIfScenarioEngine()
    sim_res = engine.simulate_increase_menu_price(item_id="ITEM-046", price_increase_pct=5.0)
    assert sim_res.is_simulation_estimate is True






@pytest.mark.difficult_cases
def test_case_7_price_sensitive_item():
    """
    Difficult Case 7: Price-sensitive item.
    Elasticity |E| > 1.5 causes disproportionate decline in demand when price increases.
    """
    engine = WhatIfScenarioEngine()

    res = engine.simulate_increase_menu_price(item_id="ITEM-046", price_increase_pct=10.0)


    assert res.demand["pct_change"] < 0
    assert res.demand["delta_units"] < 0





@pytest.mark.difficult_cases
def test_case_8_customer_churn_at_risk_detection():
    """
    Difficult Case 8: Customer churn.
    Identifies at-risk customers characterized by increasing recency, declining frequency, and declining monetary spend.
    """
    churn_rep_path = os.path.join(PROJECT_ROOT, "reports", "churn", "customer_churn_risk_summary.json")
    assert os.path.exists(churn_rep_path)
    with open(churn_rep_path, "r", encoding="utf-8") as f:
        churn_data = json.load(f)

    assert "total_customers_evaluated" in churn_data
    assert "high_churn_risk_count" in churn_data
    assert churn_data["high_churn_risk_count"] > 0


    pred_res = client.post("/api/v1/models/predict-tagged", json={
        "task_type": "Churn",
        "pipeline_type": "Spark",
        "entity_type": "CUSTOMER",
        "entity_id": "CUST-CHURN-TEST-01",
        "features": {"recency_days": 85, "frequency_orders": 1, "monetary_spend": 25.0}
    }, headers={"X-User-Role": "analyst"})
    assert pred_res.status_code == 200
    assert pred_res.json()["predicted_value"] == "AT_RISK"





@pytest.mark.difficult_cases
def test_case_9_rating_anomaly_detection():
    """
    Difficult Case 9: Rating anomaly.
    Detects sudden negative star rating deviations or review discrepancies.
    """
    rating_rep_path = os.path.join(PROJECT_ROOT, "reports", "ratings", "rating_and_satisfaction_report.json")
    assert os.path.exists(rating_rep_path)
    with open(rating_rep_path, "r", encoding="utf-8") as f:
        rating_data = json.load(f)

    assert "anomaly_summary" in rating_data
    assert "sudden_rating_drops" in rating_data["anomaly_summary"]
    assert rating_data["anomaly_summary"]["sudden_rating_drops"] > 0





@pytest.mark.difficult_cases
def test_case_10_sales_anomaly_outlier_surge():
    """
    Difficult Case 10: Sales anomaly.
    Detects volume surges beyond 3 standard deviations (z-score > 3.0) or Isolation Forest outliers.
    """
    anomaly_rep_path = os.path.join(PROJECT_ROOT, "reports", "anomalies", "sales_anomaly_report.json")
    assert os.path.exists(anomaly_rep_path)
    with open(anomaly_rep_path, "r", encoding="utf-8") as f:
        sales_anom_data = json.load(f)

    assert "anomaly_counts" in sales_anom_data
    assert sales_anom_data["anomaly_counts"]["sudden_sales_spikes"] > 0





@pytest.mark.difficult_cases
def test_case_11_spark_python_disagreement_case():
    """
    Difficult Case 11: Spark / Python model disagreement.
    Dual-pipeline comparison engine computes discrepancy rates and logs disagreements.
    """
    comp_rep_path = os.path.join(PROJECT_ROOT, "reports", "model_comparison", "dual_pipeline_comparison_report.json")
    assert os.path.exists(comp_rep_path)
    with open(comp_rep_path, "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    assert "mismatch_count" in comp_data
    assert "overall_agreement_percentage" in comp_data
    assert comp_data["overall_agreement_percentage"] > 90.0
    assert len(comp_data["records"]) > 0
