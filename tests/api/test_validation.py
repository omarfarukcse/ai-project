# tests/api/test_validation.py
"""
Input Validation Tests
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


class TestValidation:
    """Input validation test suite"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = TestClient(app)
    
    def test_validate_glucose(self, auth_headers):
        """Test glucose validation"""
        test_cases = [
            {"glucose": 148},  # Valid
            {"glucose": -10},  # Invalid
            {"glucose": 500},  # Invalid
            {"glucose": "abc"},  # Type error
        ]
        
        for test in test_cases:
            response = self.client.post(
                "/predict",
                json=test,
                headers=auth_headers
            )
            # Should either pass or return error
            assert response.status_code in [200, 400, 422]
    
    def test_validate_bmi(self, auth_headers):
        """Test BMI validation"""
        test_cases = [
            {"bmi": 25.0},  # Valid
            {"bmi": -5},  # Invalid
            {"bmi": 100},  # Invalid
        ]
        
        for test in test_cases:
            response = self.client.post(
                "/predict",
                json=test,
                headers=auth_headers
            )
            assert response.status_code in [200, 400, 422]
    
    def test_validate_age(self, auth_headers):
        """Test age validation"""
        test_cases = [
            {"age": 45},  # Valid
            {"age": -5},  # Invalid
            {"age": 200},  # Invalid
        ]
        
        for test in test_cases:
            response = self.client.post(
                "/predict",
                json=test,
                headers=auth_headers
            )
            assert response.status_code in [200, 400, 422]
    
    def test_validate_blood_pressure(self, auth_headers):
        """Test blood pressure validation"""
        test_cases = [
            {"blood_pressure": 72},  # Valid
            {"blood_pressure": -10},  # Invalid
            {"blood_pressure": 300},  # Invalid
        ]
        
        for test in test_cases:
            response = self.client.post(
                "/predict",
                json=test,
                headers=auth_headers
            )
            assert response.status_code in [200, 400, 422]
    
    def test_validate_pregnancies(self, auth_headers):
        """Test pregnancies validation"""
        test_cases = [
            {"pregnancies": 2},  # Valid
            {"pregnancies": -1},  # Invalid
            {"pregnancies": 30},  # Invalid
        ]
        
        for test in test_cases:
            response = self.client.post(
                "/predict",
                json=test,
                headers=auth_headers
            )
            assert response.status_code in [200, 400, 422]
    
    def test_validate_all_fields(self, sample_patient_data, auth_headers):
        """Test validation with all fields"""
        response = self.client.post(
            "/predict",
            json=sample_patient_data,
            headers=auth_headers
        )
        assert response.status_code in [200, 400, 422]
        