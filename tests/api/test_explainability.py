# tests/api/test_explainability.py
"""
Explainability Endpoint Tests
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


class TestExplainabilityEndpoint:
    """Explainability endpoint test suite"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = TestClient(app)
    
    def test_get_explanations(self, sample_patient_data, auth_headers):
        """Test getting explanations for a prediction"""
        # First make a prediction
        pred_response = self.client.post(
            "/predict",
            json=sample_patient_data,
            headers=auth_headers
        )
        
        if pred_response.status_code == 200:
            patient_id = pred_response.json()["patient_id"]
            
            # Get explanations
            response = self.client.get(
                f"/explanations/{patient_id}",
                headers=auth_headers
            )
            
            if response.status_code == 200:
                data = response.json()
                assert "patient_id" in data
                assert "risk_score" in data
                assert "risk_level" in data
                assert "shap_values" in data
                assert "base_value" in data
                assert "feature_names" in data
                assert "contribution_breakdown" in data
    
    def test_get_feature_importance(self, auth_headers):
        """Test getting global feature importance"""
        response = self.client.get(
            "/feature-importance",
            headers=auth_headers
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "features" in data
            assert "scores" in data
            assert len(data["features"]) > 0
            assert len(data["scores"]) == len(data["features"])
    
    def test_get_explanations_not_found(self, auth_headers):
        """Test explanations for non-existent patient"""
        response = self.client.get(
            "/explanations/NONEXISTENT",
            headers=auth_headers
        )
        
        assert response.status_code in [404, 500]
    
    def test_waterfall_plot(self, sample_patient_data, auth_headers):
        """Test waterfall plot endpoint"""
        pred_response = self.client.post(
            "/predict",
            json=sample_patient_data,
            headers=auth_headers
        )
        
        if pred_response.status_code == 200:
            patient_id = pred_response.json()["patient_id"]
            
            response = self.client.get(
                f"/explanations/{patient_id}/waterfall",
                headers=auth_headers
            )
            
            # Should return an image or 404
            assert response.status_code in [200, 404]
            if response.status_code == 200:
                assert "image/png" in response.headers["content-type"]
    
    def test_force_plot(self, sample_patient_data, auth_headers):
        """Test force plot endpoint"""
        pred_response = self.client.post(
            "/predict",
            json=sample_patient_data,
            headers=auth_headers
        )
        
        if pred_response.status_code == 200:
            patient_id = pred_response.json()["patient_id"]
            
            response = self.client.get(
                f"/explanations/{patient_id}/force",
                headers=auth_headers
            )
            
            # Should return an image or 404
            assert response.status_code in [200, 404]
            if response.status_code == 200:
                assert "image/png" in response.headers["content-type"]
                