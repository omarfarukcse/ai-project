# tests/smoke/test_smoke.py
import pytest
import requests
import json
import time
from typing import Dict, Any

class TestSmoke:
    """Smoke tests for production deployment validation"""
    
    @pytest.fixture
    def api_url(self, request):
        """Get API URL from command line or environment"""
        return request.config.getoption("--api-url") or os.getenv("API_URL", "http://localhost:8000")
    
    def test_health_endpoint(self, api_url):
        """Verify health check endpoint is operational"""
        response = requests.get(f"{api_url}/api/v1/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert data['model_loaded'] is True
        print(f"✅ Health check passed: {data['model_version']}")
    
    def test_prediction_endpoint(self, api_url):
        """Verify prediction endpoint generates valid response"""
        # First, get auth token
        auth_response = requests.post(f"{api_url}/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        }, timeout=10)
        assert auth_response.status_code == 200
        token = auth_response.json()['access_token']
        
        # Test prediction
        headers = {"Authorization": f"Bearer {token}"}
        patient_data = {
            "patient": {
                "Glucose": 145,
                "BMI": 33.6,
                "Age": 45,
                "BloodPressure": 72,
                "Pregnancies": 2
            }
        }
        
        response = requests.post(
            f"{api_url}/api/v1/predict",
            json=patient_data,
            headers=headers,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'prediction' in data
        assert 'risk_status' in data
        assert 'probability' in data
        print(f"✅ Prediction test passed: {data['risk_status']} ({data['probability']:.2f})")
    
    def test_batch_prediction(self, api_url):
        """Verify batch prediction endpoint"""
        # Get auth token
        auth_response = requests.post(f"{api_url}/api/v1/auth/login", json={
            "username": "admin",
            "password": "admin123"
        }, timeout=10)
        assert auth_response.status_code == 200
        token = auth_response.json()['access_token']
        
        headers = {"Authorization": f"Bearer {token}"}
        patients = [
            {"Glucose": 145, "BMI": 33.6, "Age": 45, "BloodPressure": 72, "Pregnancies": 2},
            {"Glucose": 85, "BMI": 22.5, "Age": 28, "BloodPressure": 68, "Pregnancies": 0}
        ]
        
        response = requests.post(
            f"{api_url}/api/v1/batch_predict",
            json={"patients": patients},
            headers=headers,
            timeout=60
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['total_patients'] == 2
        assert 'predictions' in data
        print(f"✅ Batch prediction test passed: {data['total_patients']} patients")
    
    def test_redis_connection(self, api_url):
        """Verify Redis is reachable"""
        response = requests.get(f"{api_url}/api/v1/health", timeout=10)
        data = response.json()
        assert data['system_metrics']['redis_status'] is not None
        print("✅ Redis connection verified")
    
    def test_mlflow_connection(self, api_url):
        """Verify MLflow is reachable"""
        response = requests.get(f"{api_url}/api/v1/health", timeout=10)
        data = response.json()
        assert 'model_version' in data
        print(f"✅ MLflow connection verified: {data['model_version']}")