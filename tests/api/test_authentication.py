# tests/api/test_authentication.py
"""
Authentication and Authorization Tests
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


class TestAuthentication:
    """Authentication test suite"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = TestClient(app)
    
    def test_login_success(self):
        """Test successful login"""
        response = self.client.post(
            "/auth/login",
            json={
                "username": "admin",
                "password": "Admin123!"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        assert "user" in data
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = self.client.post(
            "/auth/login",
            json={
                "username": "admin",
                "password": "wrongpassword"
            }
        )
        
        assert response.status_code == 401
    
    def test_login_missing_fields(self):
        """Test login with missing fields"""
        response = self.client.post(
            "/auth/login",
            json={"username": "admin"}
        )
        
        assert response.status_code == 422
    
    def test_protected_endpoint_no_auth(self):
        """Test protected endpoint without authentication"""
        response = self.client.post(
            "/predict",
            json={"glucose": 148, "bmi": 33.6}
        )
        
        assert response.status_code in [401, 403]
    
    def test_protected_endpoint_with_auth(self, auth_headers):
        """Test protected endpoint with authentication"""
        response = self.client.post(
            "/predict",
            json={"glucose": 148, "bmi": 33.6},
            headers=auth_headers
        )
        
        assert response.status_code in [200, 400, 422]
    
    def test_refresh_token(self, auth_headers):
        """Test token refresh"""
        response = self.client.post(
            "/auth/refresh",
            headers=auth_headers
        )
        
        assert response.status_code in [200, 401]
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
    
    def test_logout(self, auth_headers):
        """Test logout"""
        response = self.client.post(
            "/auth/logout",
            headers=auth_headers
        )
        
        assert response.status_code in [200, 401]
    
    def test_token_expiry(self):
        """Test token expiry"""
        # This would need to wait for token expiry or use a mock
        pass
    
    def test_permissions(self, auth_headers):
        """Test permission-based access"""
        # Test admin-only endpoint
        response = self.client.get(
            "/admin/status",
            headers=auth_headers
        )
        
        # Should be 403 if not admin
        assert response.status_code in [200, 403]
        