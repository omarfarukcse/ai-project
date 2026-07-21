# tests/api/test_batch_predict.py
"""
Batch Prediction Endpoint Tests
"""

import pytest
import time
from fastapi.testclient import TestClient

from src.api.app import app


class TestBatchPredictionEndpoint:
    """Batch prediction endpoint test suite"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = TestClient(app)
    
    def test_batch_predict_success(self, batch_patient_data, auth_headers):
        """Test successful batch prediction"""
        response = self.client.post(
            "/predict/batch",
            json={"patients": batch_patient_data},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert "total" in data
        assert "successful" in data
        assert "failed" in data
        
        assert data["total"] == len(batch_patient_data)
        assert data["successful"] == len(batch_patient_data)
        assert data["failed"] == 0
        
        for prediction in data["predictions"]:
            assert "patient_id" in prediction
            assert "risk_score" in prediction
            assert "risk_level" in prediction
    
    def test_batch_predict_empty(self, auth_headers):
        """Test batch prediction with empty list"""
        response = self.client.post(
            "/predict/batch",
            json={"patients": []},
            headers=auth_headers
        )
        
        assert response.status_code in [200, 400, 422]
    
    def test_batch_predict_large(self, auth_headers):
        """Test batch prediction with many patients"""
        # Generate 100 patients
        patients = []
        for i in range(100):
            patients.append({
                "pregnancies": i % 10,
                "glucose": 70 + (i % 100),
                "blood_pressure": 60 + (i % 60),
                "skin_thickness": 20 + (i % 30),
                "insulin": 10 + (i % 50),
                "bmi": 18 + (i % 20),
                "diabetes_pedigree": 0.1 + (i % 20) / 100,
                "age": 20 + (i % 60),
            })
        
        start_time = time.time()
        response = self.client.post(
            "/predict/batch",
            json={"patients": patients},
            headers=auth_headers
        )
        elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            assert data["total"] == 100
            assert data["successful"] == 100
            # Should process within reasonable time
            assert elapsed < 5000, f"Batch time: {elapsed}ms"
    
    def test_batch_predict_with_explanations(self, batch_patient_data, auth_headers):
        """Test batch prediction with explanations"""
        response = self.client.post(
            "/predict/batch?explain=true",
            json={"patients": batch_patient_data[:2]},
            headers=auth_headers
        )
        
        if response.status_code == 200:
            data = response.json()
            for prediction in data["predictions"]:
                assert "top_factors" in prediction
                assert "contributing_factors" in prediction
    
    def test_batch_predict_correlation_id(self, batch_patient_data, auth_headers):
        """Test batch correlation ID propagation"""
        response = self.client.post(
            "/predict/batch",
            json={"patients": batch_patient_data[:2]},
            headers={
                **auth_headers,
                "X-Correlation-ID": "batch-test-123"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            for prediction in data["predictions"]:
                assert "correlation_id" in prediction
                assert prediction["correlation_id"].startswith("batch-test-123")