"""
Unit and Integration Tests for Customer Intelligence Dashboard (SRS Step 44)
Verifies:
- customer segments
- RFM distribution
- high-value customers
- at-risk customers
- promotion-sensitive customers
- customer trends
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_customer_intelligence_dashboard_all_srs_fields():
    """Verify that /api/v1/customer-intelligence returns all exact Step 44 fields."""
    resp = client.get("/api/v1/customer-intelligence")
    assert resp.status_code == 200
    data = resp.json()

                   
    assert data["srs_step"] == 44
    assert "summary_metrics" in data
    assert data["summary_metrics"]["total_customers"] == 50000

                          
    assert "customer_segments" in data
    assert len(data["customer_segments"]) == 6
    segment_names = [s["segment_name"] for s in data["customer_segments"]]
    expected_segments = [
        "High-Value Loyal Customers",
        "Occasional Customers",
        "Promotion-Driven Customers",
        "At-Risk Customers",
        "New Customers",
        "Frequent Customers"
    ]
    for exp in expected_segments:
        assert exp in segment_names, f"Missing segment: {exp}"

                           
    sample_seg = data["customer_segments"][0]
    for k in ["segment_name", "customer_count", "total_spend", "avg_monetary_value", "avg_frequency", "avg_recency"]:
        assert k in sample_seg, f"Missing key in segment: {k}"

                         
    assert "rfm_distribution" in data
    rfm = data["rfm_distribution"]
    assert "recency_distribution" in rfm and len(rfm["recency_distribution"]) == 5
    assert "frequency_distribution" in rfm and len(rfm["frequency_distribution"]) == 5
    assert "monetary_distribution" in rfm and len(rfm["monetary_distribution"]) == 5
    assert "average_r_score" in rfm
    assert "average_f_score" in rfm
    assert "average_m_score" in rfm

                             
    assert "high_value_customers" in data
    hvc = data["high_value_customers"]
    assert hvc["total_count"] == 5707
    assert hvc["total_spend"] > 0
    assert hvc["spend_share_pct"] > 0
    assert len(hvc["customers"]) > 0
    sample_hvc = hvc["customers"][0]
    for k in ["customer_id", "name", "monetary_value", "frequency", "recency", "loyalty_tier"]:
        assert k in sample_hvc, f"Missing high value customer key: {k}"

                                                
    assert "at_risk_customers" in data
    arc = data["at_risk_customers"]
    assert arc["total_count"] == 5749
    assert arc["total_spend"] > 0
    assert len(arc["customers"]) > 0
    sample_arc = arc["customers"][0]
    for k in ["customer_id", "churn_risk_score", "churn_risk_tier", "primary_risk_driver", "recommended_retention_action"]:
        assert k in sample_arc, f"Missing at-risk customer key: {k}"

                                      
    assert "promotion_sensitive_customers" in data
    psc = data["promotion_sensitive_customers"]
    assert psc["total_count"] == 6102
    assert psc["total_spend"] > 0
    assert len(psc["customers"]) > 0
    sample_psc = psc["customers"][0]
    assert "promotion_sensitivity" in sample_psc
    assert sample_psc["promotion_sensitivity"] >= 0

                        
    assert "customer_trends" in data
    assert len(data["customer_trends"]) == 12
    sample_trend = data["customer_trends"][0]
    for k in ["month", "new_signups", "active_customers", "monthly_spend", "repeat_orders"]:
        assert k in sample_trend, f"Missing trend key: {k}"


def test_customer_segments_endpoint():
    """Verify /api/v1/customer-intelligence/segments returns segment summaries."""
    resp = client.get("/api/v1/customer-intelligence/segments")
    assert resp.status_code == 200
    data = resp.json()
    assert "segments" in data
    assert len(data["segments"]) == 6


def test_high_value_endpoint():
    """Verify /api/v1/customer-intelligence/high-value returns VIP cohort."""
    resp = client.get("/api/v1/customer-intelligence/high-value")
    assert resp.status_code == 200
    data = resp.json()
    assert "high_value_customers" in data
    assert data["high_value_customers"]["total_count"] == 5707


def test_at_risk_endpoint():
    """Verify /api/v1/customer-intelligence/at-risk returns at-risk cohort."""
    resp = client.get("/api/v1/customer-intelligence/at-risk")
    assert resp.status_code == 200
    data = resp.json()
    assert "at_risk_customers" in data
    assert data["at_risk_customers"]["total_count"] == 5749


def test_promotion_sensitive_endpoint():
    """Verify /api/v1/customer-intelligence/promotion-sensitive returns promo cohort."""
    resp = client.get("/api/v1/customer-intelligence/promotion-sensitive")
    assert resp.status_code == 200
    data = resp.json()
    assert "promotion_sensitive_customers" in data
    assert data["promotion_sensitive_customers"]["total_count"] == 6102
