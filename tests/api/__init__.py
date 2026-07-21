# tests/api/__init__.py
"""
API Tests Package - Comprehensive Testing for CDSS API

This package provides complete test coverage for all API endpoints:
- Health checks
- Prediction endpoints (single, batch)
- Explainability endpoints
- Metrics endpoints
- Authentication and authorization
- Rate limiting
- Security
- Performance
- Input validation

Test Categories:
    Unit Tests: Individual component tests
    Integration Tests: End-to-end API tests
    Performance Tests: Load and stress tests
    Security Tests: Auth, injection, rate limiting

Usage:
    pytest tests/api/ -v
    pytest tests/api/test_predict.py -v
    pytest tests/api/ -k "test_predict" -v
"""

import pytest

pytest_plugins = [
    "tests.api.fixtures",
    "tests.api.conftest",
]