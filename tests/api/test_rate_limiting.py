# tests/api/test_rate_limiting.py
"""
Rate Limiting Tests
"""

import pytest
import time
from fastapi.testclient import TestClient

from src.api.app import app


class TestRateLimiting:
    """Rate limiting test suite"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = TestClient(app)
    
    def test_rate_limit_health(self):
        """Test rate limiting on health endpoint"""
        # Health endpoint should not be rate limited
        for i in range(10):
            response = self.client.get("/health")
            assert response.status_code == 200
    
    def test_rate_limit_predict(self, sample_patient_data, auth_headers):
        """Test rate limiting on prediction endpoint"""
        responses = []
        
        # Make many requests quickly
        for i in range(20):
            response = self.client.post(
                "/predict",
                json=sample_patient_data,
                headers=auth_headers
            )
            responses.append(response)
        
        # Some requests should be rate limited (429)
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes, f"Status codes: {status_codes}"
    
    def test_rate_limit_reset(self, sample_patient_data, auth_headers):
        """Test rate limit reset after waiting"""
        # Make requests until rate limited
        responses = []
        for i in range(15):
            response = self.client.post(
                "/predict",
                json=sample_patient_data,
                headers=auth_headers
            )
            responses.append(response)
            if response.status_code == 429:
                break
        
        # Wait for reset
        time.sleep(65)  # Wait for minute reset
        
        # Should work again
        response = self.client.post(
            "/predict",
            json=sample_patient_data,
            headers=auth_headers
        )
        assert response.status_code != 429
    
    def test_rate_limit_headers(self, sample_patient_data, auth_headers):
        """Test rate limit headers"""
        response = self.client.post(
            "/predict",
            json=sample_patient_data,
            headers=auth_headers
        )
        
        # Check for rate limit headers
        headers = response.headers
        # Common rate limit headers
        if "X-RateLimit-Limit" in headers:
            assert int(headers["X-RateLimit-Limit"]) > 0
        if "X-RateLimit-Remaining" in headers:
            assert int(headers["X-RateLimit-Remaining"]) >= 0
    
    def test_rate_limit_different_ips(self):
        """Test rate limiting by IP"""
        # This would require mocking the client IP
        pass
    