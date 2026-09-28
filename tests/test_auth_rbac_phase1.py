"""
Phase FIX-1 Test Suite: Authentication & Role-Based Access Control (RBAC)
Covers SRS Functional Requirements (i) and (ii):
  - FR-i: User Registration and Authentication for 4 roles:
          Administrator, Regional Manager, Store Manager, Data Analyst.
  - FR-ii: Role-Based Access Control (RBAC) enforcement across system endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

ROLES_CREDENTIALS = [
    ("admin_user", "admin123", "admin"),
    ("regional_mgr", "regional123", "regional_manager"),
    ("store_mgr", "manager123", "manager"),
    ("data_analyst", "analyst123", "analyst"),
]


class TestAuthenticationFRi:
    """Tests for Functional Requirement (i) - User Authentication."""

    @pytest.mark.parametrize("username,password,expected_role", ROLES_CREDENTIALS)
    def test_login_all_four_roles_success(self, username, password, expected_role):
        """Verify each of the 4 SRS roles can authenticate and receive a valid token."""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": password}
        )
        assert response.status_code == 200, f"Failed login for {username}: {response.text}"
        data = response.json()
        assert "access_token" in data
        assert data["token_type"].lower() == "bearer"
        assert data["user"]["role_id"] == expected_role
        assert data["user"]["username"] == username

    def test_login_via_email(self):
        """Verify users can authenticate using their registered email address."""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin@dineiq.com", "password": "admin123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user"]["role_id"] == "admin"
        assert data["user"]["email"] == "admin@dineiq.com"

    def test_login_invalid_password_returns_401(self):
        """Verify incorrect password returns 401 Unauthorized."""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin_user", "password": "WrongPassword!999"}
        )
        assert response.status_code == 401
        assert "detail" in response.json()

    def test_login_nonexistent_user_returns_401(self):
        """Verify non-existent username returns 401 Unauthorized."""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "non_existent_user_xyz", "password": "password123"}
        )
        assert response.status_code == 401

    def test_get_current_user_profile_me(self):
        """Verify /auth/me returns the active session user's full context."""
                  
        login_res = client.post(
            "/api/v1/auth/login",
            json={"username": "data_analyst", "password": "analyst123"}
        )
        token = login_res.json()["access_token"]

                        
        me_res = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert me_res.status_code == 200
        user = me_res.json()["user"]
        assert user["username"] == "data_analyst"
        assert user["role_id"] == "analyst"

    def test_unauthenticated_request_rejected(self):
        """Verify protected endpoints reject requests without auth headers."""
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401

    def test_logout_invalidates_session_token(self):
        """Verify logging out pops the token from active sessions, causing subsequent requests to fail with 401."""
                  
        login_res = client.post(
            "/api/v1/auth/login",
            json={"username": "store_mgr", "password": "manager123"}
        )
        token = login_res.json()["access_token"]

                                   
        res_before = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert res_before.status_code == 200

                   
        logout_res = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert logout_res.status_code == 200
        assert "logged out" in logout_res.json()["message"].lower()

                                                  
        res_after = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert res_after.status_code == 401

    def test_list_all_system_roles(self):
        """Verify all 4 required SRS roles exist in the database."""
        response = client.get("/api/v1/auth/roles")
        assert response.status_code == 200
        roles = response.json()
        role_ids = {r["role_id"] for r in roles}
        assert {"admin", "regional_manager", "manager", "analyst"}.issubset(role_ids)


class TestRoleBasedAccessControlFRii:
    """Tests for Functional Requirement (ii) - Role-Based Access Control (RBAC)."""

    def test_admin_can_perform_admin_operations(self):
        """Administrator token has access to admin endpoints."""
        login_res = client.post(
            "/api/v1/auth/login",
            json={"username": "admin_user", "password": "admin123"}
        )
        admin_token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {admin_token}"}

                                          
        res = client.get("/api/v1/system/config", headers=headers)
        assert res.status_code in [200, 404]                               

    def test_manager_forbidden_from_admin_operations(self):
        """Store Manager attempting admin-restricted location deletion must receive 403 Forbidden."""
        login_res = client.post(
            "/api/v1/auth/login",
            json={"username": "store_mgr", "password": "manager123"}
        )
        manager_token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {manager_token}"}

                                                   
        res = client.delete("/api/v1/locations/LOC-001", headers=headers)
        assert res.status_code == 403
        assert "Access forbidden" in res.json()["detail"]

    def test_analyst_forbidden_from_inventory_updates(self):
        """Analyst attempting to write inventory updates must receive 403 Forbidden."""
        login_res = client.post(
            "/api/v1/auth/login",
            json={"username": "data_analyst", "password": "analyst123"}
        )
        analyst_token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {analyst_token}"}

        res = client.put(
            "/api/v1/inventory/INV-NONEXISTENT",
            json={"ending_stock": 50},
            headers=headers
        )
        assert res.status_code == 403
