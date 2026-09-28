"""
Tests for SRS Functional Requirements (i) through (xi):
  (i)   User Registration and Authentication (Managers, Analysts, Regional Managers, Admins)
  (ii)  Role-Based Access Control (RBAC)
  (iii) Restaurant Location Management (Admin CRUD)
  (iv)  Menu Management (Categories, Items, Prices, Costs, Descriptions, Availability)
  (v)   Pricing History Management
  (vi)  Customer Data Management (Anonymized Profiles)
  (vii) Order Management (Headers + Order-Lines)
  (viii) Promotion Management (Discounts, Coupons, Campaign Periods, Applicable Items)
  (ix)  Rating Management
  (x)   Inventory Management
  (xi)  Wastage Management
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


ADMIN_HEADERS = {"X-User-Role": "admin", "X-Username": "admin_tester"}
MANAGER_HEADERS = {"X-User-Role": "manager", "X-Username": "manager_tester"}
ANALYST_HEADERS = {"X-User-Role": "analyst", "X-Username": "analyst_tester"}
REGIONAL_HEADERS = {"X-User-Role": "regional_manager", "X-Username": "reg_tester"}






def test_user_registration_and_login_flow():
    """Verify user registration across roles, login token issuance, and authenticated /me."""
    unique_suffix = "test_srs_flow"


    reg_payload = {
        "username": f"mgr_{unique_suffix}",
        "email": f"mgr_{unique_suffix}@dineiq.com",
        "password": "SecurePassword123!",
        "full_name": "Test Store Manager",
        "role_id": "manager",
        "assigned_location_id": "LOC-001"
    }
    reg_res = client.post("/api/v1/auth/register", json=reg_payload)
    assert reg_res.status_code in (201, 400)


    login_res = client.post("/api/v1/auth/login", json={
        "username": f"mgr_{unique_suffix}",
        "password": "SecurePassword123!"
    })
    assert login_res.status_code == 200
    login_data = login_res.json()
    assert "access_token" in login_data
    assert login_data["user"]["role_id"] == "manager"
    token = login_data["access_token"]


    me_res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["user"]["username"] == f"mgr_{unique_suffix}"


    bad_login = client.post("/api/v1/auth/login", json={
        "username": f"mgr_{unique_suffix}",
        "password": "WrongPassword!"
    })
    assert bad_login.status_code == 401


def test_rbac_access_restrictions():
    """Verify RBAC properly protects admin endpoints from unauthorized roles."""
    location_payload = {
        "location_id": "LOC-RBAC-TEST",
        "name": "RBAC Test Location",
        "city": "Dallas",
        "state": "TX"
    }


    res_analyst = client.post("/api/v1/locations", json=location_payload, headers=ANALYST_HEADERS)
    assert res_analyst.status_code == 403


    res_admin = client.post("/api/v1/locations", json=location_payload, headers=ADMIN_HEADERS)
    assert res_admin.status_code in (201, 400)






def test_restaurant_location_management_crud():
    """Verify Admin CRUD for restaurant locations."""
    loc_id = "LOC-CRUD-99"

    create_res = client.post("/api/v1/locations", json={
        "location_id": loc_id,
        "name": "CRUD Test Branch",
        "city": "Austin",
        "state": "TX",
        "seating_capacity": 150,
        "cost_index": 1.05
    }, headers=ADMIN_HEADERS)
    assert create_res.status_code in (201, 400)


    list_res = client.get("/api/v1/locations")
    assert list_res.status_code == 200
    assert isinstance(list_res.json(), list)


    get_res = client.get(f"/api/v1/locations/{loc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["city"] == "Austin"


    update_res = client.put(f"/api/v1/locations/{loc_id}", json={
        "seating_capacity": 180,
        "operating_status": "ACTIVE"
    }, headers=ADMIN_HEADERS)
    assert update_res.status_code == 200


    del_res = client.delete(f"/api/v1/locations/{loc_id}", headers=ADMIN_HEADERS)
    assert del_res.status_code == 200






def test_menu_management():
    """Verify menu categories, menu items, price/cost setting, and availability toggles."""
    cat_id = "CAT-TEST-1"

    c_res = client.post("/api/v1/menu/categories", json={
        "category_id": cat_id,
        "name": "Test Signature Mains",
        "description": "Premium test burgers and mains",
        "target_margin_pct": 65.0
    }, headers=MANAGER_HEADERS)
    assert c_res.status_code in (201, 200)


    cats = client.get("/api/v1/menu/categories")
    assert cats.status_code == 200


    item_id = "ITEM-TEST-BURGER"
    item_res = client.post("/api/v1/menu/items", json={
        "item_id": item_id,
        "category_id": cat_id,
        "name": "Signature Wagyu Burger",
        "description": "Double patty wagyu beef",
        "base_price": 18.50,
        "cost_price": 5.50,
        "is_active": True
    }, headers=MANAGER_HEADERS)
    assert item_res.status_code in (201, 200)


    items = client.get("/api/v1/menu/items")
    assert items.status_code == 200


    patch_res = client.patch(f"/api/v1/menu/items/{item_id}/availability?is_active=false", headers=MANAGER_HEADERS)
    assert patch_res.status_code == 200
    assert patch_res.json()["is_active"] is False






def test_pricing_history_management():
    """Verify pricing history audit log entries."""
    hist_payload = {
        "item_id": "ITEM-TEST-BURGER",
        "location_id": "LOC-001",
        "base_price": 19.50,
        "cost_price": 5.80,
        "change_reason": "Quarterly inflation adjustment"
    }
    create_hist = client.post("/api/v1/pricing-history", json=hist_payload, headers=MANAGER_HEADERS)
    assert create_hist.status_code == 201

    list_hist = client.get("/api/v1/pricing-history?item_id=ITEM-TEST-BURGER")
    assert list_hist.status_code == 200
    assert len(list_hist.json()) >= 1






def test_customer_data_management():
    """Verify anonymized customer data profiles, creation, and updates."""
    cust_id = "CUST-TEST-01"
    create_c = client.post("/api/v1/customers", json={
        "customer_id": cust_id,
        "first_name": "Jane",
        "last_name": "Doe",
        "customer_segment": "High-Value",
        "loyalty_tier": "Gold",
        "loyalty_points": 500,
        "preferred_location_id": "LOC-001"
    }, headers=MANAGER_HEADERS)
    assert create_c.status_code in (201, 200)


    list_c = client.get("/api/v1/customers?segment=High-Value")
    assert list_c.status_code == 200
    data = list_c.json()
    assert len(data) >= 1
    assert "***" in data[0]["anonymized_name"]


    up_c = client.put(f"/api/v1/customers/{cust_id}", json={
        "loyalty_points": 650,
        "churn_risk_score": 0.05
    }, headers=MANAGER_HEADERS)
    assert up_c.status_code == 200






def test_order_management():
    """Verify order headers + order lines creation, reading, and status updates."""
    order_id = "ORD-TEST-9001"
    order_payload = {
        "order_id": order_id,
        "customer_id": "CUST-TEST-01",
        "location_id": "LOC-001",
        "order_type": "Dine-in",
        "order_status": "PENDING",
        "payment_method": "Credit Card",
        "table_number": 12,
        "tip_amount": 5.00,
        "items": [
            {
                "item_id": "ITEM-TEST-BURGER",
                "quantity": 2,
                "unit_price": 18.50,
                "item_discount": 0.0
            }
        ]
    }
    create_ord = client.post("/api/v1/orders", json=order_payload, headers=MANAGER_HEADERS)
    assert create_ord.status_code in (201, 200)


    get_ord = client.get(f"/api/v1/orders/{order_id}")
    assert get_ord.status_code == 200
    ord_data = get_ord.json()
    assert ord_data["order_id"] == order_id
    assert len(ord_data["items"]) == 1


    patch_ord = client.patch(f"/api/v1/orders/{order_id}/status?status_val=COMPLETED", headers=MANAGER_HEADERS)
    assert patch_ord.status_code == 200






def test_promotion_management():
    """Verify promotion campaign creation, listing, and updating."""
    promo_id = "PROMO-TEST-WINTER"
    create_p = client.post("/api/v1/promotions", json={
        "promotion_id": promo_id,
        "promotion_name": "Winter Special 20% Off",
        "discount_type": "Percentage",
        "discount_value": 20.0,
        "min_order_amount": 30.0,
        "target_segment": "All"
    }, headers=MANAGER_HEADERS)
    assert create_p.status_code in (201, 200)

    list_p = client.get("/api/v1/promotions")
    assert list_p.status_code == 200

    up_p = client.put(f"/api/v1/promotions/{promo_id}", json={
        "discount_value": 25.0
    }, headers=ADMIN_HEADERS)
    assert up_p.status_code == 200






def test_rating_management():
    """Verify submitting customer reviews and listing ratings."""
    rat_res = client.post("/api/v1/ratings", json={
        "location_id": "LOC-001",
        "overall_rating": 5,
        "food_rating": 5,
        "service_rating": 4,
        "review_text": "Exceptional dining experience and fast service!"
    })
    assert rat_res.status_code == 201

    list_r = client.get("/api/v1/ratings?location_id=LOC-001&min_rating=4")
    assert list_r.status_code == 200
    assert len(list_r.json()) >= 1






def test_inventory_management():
    """Verify inventory snapshots, stock status, and updates."""
    inv_id = "INV-TEST-001"
    create_inv = client.post("/api/v1/inventory", json={
        "inventory_id": inv_id,
        "location_id": "LOC-001",
        "item_id": "ITEM-TEST-BURGER",
        "starting_stock": 50,
        "quantity_received": 20,
        "quantity_sold": 15,
        "quantity_wasted": 2,
        "ending_stock": 53,
        "reorder_point": 15
    }, headers=MANAGER_HEADERS)
    assert create_inv.status_code in (201, 200)

    list_inv = client.get("/api/v1/inventory?location_id=LOC-001")
    assert list_inv.status_code == 200

    up_inv = client.put(f"/api/v1/inventory/{inv_id}", json={
        "ending_stock": 45,
        "stock_status": "Adequate"
    }, headers=MANAGER_HEADERS)
    assert up_inv.status_code == 200






def test_wastage_management():
    """Verify logging wastage, listing records, and admin cleanup."""
    wast_id = "WST-TEST-99"
    log_w = client.post("/api/v1/wastage", json={
        "wastage_id": wast_id,
        "item_id": "ITEM-TEST-BURGER",
        "location_id": "LOC-001",
        "quantity_wasted": 4,
        "unit_cost": 5.50,
        "wastage_reason": "Expired"
    }, headers=MANAGER_HEADERS)
    assert log_w.status_code in (201, 200)
    assert log_w.json()["total_loss"] == 22.0

    list_w = client.get("/api/v1/wastage?location_id=LOC-001")
    assert list_w.status_code == 200

    del_w = client.delete(f"/api/v1/wastage/{wast_id}", headers=ADMIN_HEADERS)
    assert del_w.status_code == 200
