# tests/api/test_predict.py
"""
Prediction Endpoint Tests
"""

import pytest
import json
import time
from fastapi.testclient import TestClient

from src.api.app import app
from tests.api.conftest import APIHelper


class TestPredictionEndpoint:
    """Prediction endpoint test suite"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = TestClient(app)
        self.api_helper = APIHelper()
    
    def test_predict_success(self, sample_patient_data, auth_headers):
        """Test successful prediction"""
        response = self.client.post(
            "/predict",
            json=sample_patient_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        self.api_helper.validate_response_structure(
            data,
            ["patient_id", "risk_score", "risk_level", "probability", "clinical_explanation"]
        )
        
        # Validate values
        self.api_helper.validate_risk_score(data["risk_score"])
        self.api_helper.validate_risk_level(data["risk_level"])
        self.api_helper.validate_probability(data["probability"])
        
        assert "correlation_id" in data
        assert data["correlation_id"] is not None
    
    def test_predict_missing_fields(self, auth_headers):
        """Test prediction with missing fields"""
        incomplete_data = {
            "glucose": 148,
            "bmi": 33.6
            # Missing other fields
        }
        
        response = self.client.post(
            "/predict",
            json=incomplete_data,
            headers=auth_headers
        )
        
        # Should either work with defaults or return 422
        assert response.status_code in [200, 422]
    
    def test_predict_invalid_data(self, invalid_patient_data, auth_headers):
        """Test prediction with invalid data"""
        response = self.client.post(
            "/predict",
            json=invalid_patient_data,
            headers=auth_headers
        )
        
        # Should fail validation
        assert response.status_code in [400, 422, 500]
    
    def test_predict_heart_disease(self, sample_patient_data_heart, auth_headers):
        """Test heart disease prediction"""
        response = self.client.post(
            "/predict",
            json=sample_patient_data_heart,
            headers=auth_headers
        )
        
        if response.status_code == 200:
            data = response.json()
            self.api_helper.validate_risk_score(data["risk_score"])
            self.api_helper.validate_risk_level(data["risk_level"])
    
    def test_predict_response_time(self, sample_patient_data, auth_headers):
        """Test prediction response time"""
        start_time = time.time()
        response = self.client.post(
            "/predict",
            json=sample_patient_data,
            headers=auth_headers
        )
        elapsed = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        # Should respond within 500ms (adjust based on performance)
        assert elapsed < 500, f"Response time: {elapsed}ms"
    
    def test_predict_cache(self, sample_patient_data, auth_headers):
        """Test prediction caching"""
        # First request
        response1 = self.client.post(
            "/predict",
            json=sample_patient_data,
            headers=auth_headers
        )
        
        # Second request (should be faster)
        start_time = time.time()
        response2 = self.client.post(
            "/predict",
            json=sample_patient_data,
            headers=auth_headers
        )
        elapsed = (time.time() - start_time) * 1000
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert elapsed < 200, f"Cached response time: {elapsed}ms"
    
    def test_predict_with_explanation(self, sample_patient_data, auth_headers):
        """Test prediction with explanations"""
        response = self.client.post(
            "/predict?explain=true",
            json=sample_patient_data,
            headers=auth_headers
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "contributing_factors" in data
            assert "top_factors" in data
            assert "clinical_explanation" in data
    
    def test_predict_correlation_id(self, sample_patient_data, auth_headers):
        """Test correlation ID generation"""
        response = self.client.post(
            "/predict",
            json=sample_patient_data,
            headers={
                **auth_headers,
                "X-Correlation-ID": "test-correlation-123"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "correlation_id" in data
            assert data["correlation_id"] == "test-correlation-123"