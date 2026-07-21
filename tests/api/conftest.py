# tests/api/conftest.py
"""
Test Configuration and Fixtures for API Tests
"""

import pytest
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Generator, AsyncGenerator
from datetime import datetime
import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np

from src.api.app import app
from src.config_manager import get_config_manager
from src.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# 📊 Test Configuration
# ============================================================================
@pytest.fixture(scope="session")
def test_config():
    """Load test configuration"""
    config_manager = get_config_manager()
    return {
        "base_url": "http://testserver",
        "api_prefix": "",
        "timeout": 30,
        "test_data_path": Path("tests/api/fixtures"),
    }


# ============================================================================
# 🌐 Test Client
# ============================================================================
@pytest.fixture(scope="function")
def test_client():
    """Create test client for FastAPI app"""
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="function")
async def async_client():
    """Create async test client"""
    async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


# ============================================================================
# 📊 Test Data Fixtures
# ============================================================================
@pytest.fixture(scope="session")
def sample_patient_data() -> Dict[str, Any]:
    """Sample patient data for testing"""
    return {
        "pregnancies": 6,
        "glucose": 148,
        "blood_pressure": 72,
        "skin_thickness": 35,
        "insulin": 0,
        "bmi": 33.6,
        "diabetes_pedigree": 0.627,
        "age": 50,
        "patient_id": "TEST-001"
    }


@pytest.fixture(scope="session")
def sample_patient_data_heart() -> Dict[str, Any]:
    """Sample heart disease patient data"""
    return {
        "age": 65,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 280,
        "fbs": 1,
        "restecg": 2,
        "thalach": 120,
        "exang": 1,
        "oldpeak": 2.5,
        "slope": 2,
        "ca": 2,
        "thal": 2,
        "patient_id": "TEST-002"
    }


@pytest.fixture(scope="session")
def batch_patient_data() -> List[Dict[str, Any]]:
    """Batch patient data for testing"""
    return [
        {
            "pregnancies": 6,
            "glucose": 148,
            "blood_pressure": 72,
            "skin_thickness": 35,
            "insulin": 0,
            "bmi": 33.6,
            "diabetes_pedigree": 0.627,
            "age": 50,
        },
        {
            "pregnancies": 0,
            "glucose": 85,
            "blood_pressure": 65,
            "skin_thickness": 20,
            "insulin": 10,
            "bmi": 22.0,
            "diabetes_pedigree": 0.2,
            "age": 25,
        },
        {
            "pregnancies": 2,
            "glucose": 120,
            "blood_pressure": 70,
            "skin_thickness": 25,
            "insulin": 50,
            "bmi": 28.5,
            "diabetes_pedigree": 0.4,
            "age": 40,
        }
    ]


@pytest.fixture(scope="session")
def invalid_patient_data() -> Dict[str, Any]:
    """Invalid patient data for validation tests"""
    return {
        "pregnancies": -1,  # Invalid
        "glucose": 500,     # Out of range
        "blood_pressure": 300,  # Out of range
        "skin_thickness": -10,  # Invalid
        "insulin": -100,    # Invalid
        "bmi": 100,         # Out of range
        "diabetes_pedigree": 5, # Out of range
        "age": 200,         # Invalid
        "patient_id": "INVALID"
    }


# ============================================================================
# 🔐 Authentication Fixtures
# ============================================================================
@pytest.fixture(scope="session")
def test_user_credentials():
    """Test user credentials"""
    return {
        "username": "test_user",
        "password": "TestPass123!",
    }


@pytest.fixture(scope="function")
def auth_headers(test_client, test_user_credentials):
    """Get authentication headers"""
    response = test_client.post(
        "/auth/login",
        json=test_user_credentials
    )
    if response.status_code == 200:
        token = response.json().get("access_token")
        return {"Authorization": f"Bearer {token}"}
    return {}


@pytest.fixture(scope="function")
def admin_auth_headers(test_client):
    """Get admin authentication headers"""
    response = test_client.post(
        "/auth/login",
        json={"username": "admin", "password": "Admin123!"}
    )
    if response.status_code == 200:
        token = response.json().get("access_token")
        return {"Authorization": f"Bearer {token}"}
    return {}


# ============================================================================
# 📊 Expected Responses Fixtures
# ============================================================================
@pytest.fixture(scope="session")
def expected_prediction_response():
    """Expected prediction response structure"""
    return {
        "patient_id": str,
        "risk_score": float,
        "risk_level": str,
        "probability": float,
        "top_factors": list,
        "contributing_factors": list,
        "clinical_explanation": str,
        "confidence": float,
        "drift_detected": bool,
        "model_version": str,
        "correlation_id": str,
        "processing_time_ms": float,
    }


# ============================================================================
# 🧪 Test Helpers
# ============================================================================
class APIHelper:
    """Helper class for API testing"""
    
    @staticmethod
    def validate_response_structure(response: Dict, expected_fields: List[str]):
        """Validate response contains expected fields"""
        for field in expected_fields:
            assert field in response, f"Missing field: {field}"
        return True
    
    @staticmethod
    def validate_risk_score(score: float):
        """Validate risk score is between 0 and 100"""
        assert 0 <= score <= 100, f"Invalid risk score: {score}"
        return True
    
    @staticmethod
    def validate_risk_level(level: str):
        """Validate risk level"""
        valid_levels = ["Low Risk", "Moderate Risk", "High Risk"]
        assert level in valid_levels, f"Invalid risk level: {level}"
        return True
    
    @staticmethod
    def validate_probability(prob: float):
        """Validate probability is between 0 and 1"""
        assert 0 <= prob <= 1, f"Invalid probability: {prob}"
        return True


@pytest.fixture(scope="function")
def api_helper():
    """Provide API helper"""
    return APIHelper()


# ============================================================================
# 🔧 Database Fixtures (if needed)
# ============================================================================
@pytest.fixture(scope="session")
def test_db_engine():
    """Create test database engine"""
    # Use SQLite for testing
    engine = create_engine("sqlite:///:memory:")
    return engine


@pytest.fixture(scope="function")
def test_db_session(test_db_engine):
    """Create test database session"""
    Session = sessionmaker(bind=test_db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


# ============================================================================
# 🧹 Cleanup
# ============================================================================
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Clean up test data after tests"""
    yield
    # Clean up any test artifacts
    pass