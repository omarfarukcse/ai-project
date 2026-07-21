# src/validation/__init__.py
"""
Validation Package - Enterprise Data Validation

This package provides production-grade data validation with:
- Pydantic Models: Runtime type validation and parsing
- Pandera Schemas: Statistical data validation for DataFrames
- Golden Tests: Known test cases for model validation
- Schema Versioning: Schema evolution and migration
- Custom Validators: Domain-specific validation rules
- Error Aggregation: Comprehensive error reporting

Architecture:
    schema_validation.py  → Pydantic + Pandera schemas

Features:
    - Automatic data validation on ingestion
    - Schema version tracking
    - Custom clinical validators
    - Golden test management
    - Detailed error reporting
    - Performance optimized validation

Version: 3.0.0
"""

from src.validation.schema_validation import (
    # Pydantic Models
    ClinicalDataModel,
    PatientModel,
    PredictionModel,
    BatchPredictionModel,
    
    # Pandera Schemas
    ClinicalDataSchema,
    PatientSchema,
    TrainingDataSchema,
    
    # Golden Tests
    GoldenTest,
    GoldenTestResult,
    GoldenTestSuite,
    
    # Validators
    ClinicalValidator,
    DataValidator,
    
    # Utilities
    ValidationResult,
    ValidationError,
    get_validator,
)

__version__ = "3.0.0"
__all__ = [
    # Pydantic Models
    "ClinicalDataModel",
    "PatientModel",
    "PredictionModel",
    "BatchPredictionModel",
    
    # Pandera Schemas
    "ClinicalDataSchema",
    "PatientSchema",
    "TrainingDataSchema",
    
    # Golden Tests
    "GoldenTest",
    "GoldenTestResult",
    "GoldenTestSuite",
    
    # Validators
    "ClinicalValidator",
    "DataValidator",
    
    # Utilities
    "ValidationResult",
    "ValidationError",
    "get_validator",
]

import logging
logger = logging.getLogger(__name__)
logger.info(f"✅ Validation Package v{__version__} initialized")