# src/validation/schema_validation.py
"""
Enterprise Data Validation with Pydantic and Pandera
"""

import json
import re
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime, date
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator, root_validator, ValidationError as PydanticValidationError
import pandera as pa
from pandera.typing import DataFrame, Series
from pandera.errors import SchemaError

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


# ============================================================================
# 📋 Enums and Constants
# ============================================================================

class Gender(str, Enum):
    """Gender enum"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Risk level enum"""
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"


class ValidationSeverity(str, Enum):
    """Validation severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


# ============================================================================
# 🔧 Pydantic Models for Data Validation
# ============================================================================

class ClinicalDataModel(BaseModel):
    """
    Pydantic model for clinical data validation
    
    Features:
    - Type validation
    - Range validation
    - Custom business rules
    - Cross-field validation
    """
    
    # Demographics
    patient_id: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    gender: Optional[Gender] = None
    
    # Diabetes features
    pregnancies: Optional[int] = Field(None, ge=0, le=20, description="Number of pregnancies")
    glucose: Optional[float] = Field(None, ge=0, le=300, description="Glucose level (mg/dL)")
    blood_pressure: Optional[float] = Field(None, ge=0, le=200, description="Diastolic blood pressure (mm Hg)")
    skin_thickness: Optional[float] = Field(None, ge=0, le=100, description="Skin thickness (mm)")
    insulin: Optional[float] = Field(None, ge=0, le=1000, description="2-Hour serum insulin (mu U/ml)")
    bmi: Optional[float] = Field(None, ge=10, le=60, description="Body Mass Index")
    diabetes_pedigree: Optional[float] = Field(None, ge=0, le=3, description="Diabetes pedigree function")
    
    # Heart disease features
    cp: Optional[int] = Field(None, ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: Optional[float] = Field(None, ge=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: Optional[float] = Field(None, ge=0, le=600, description="Serum cholesterol (mg/dL)")
    fbs: Optional[int] = Field(None, ge=0, le=1, description="Fasting blood sugar > 120 mg/dL")
    restecg: Optional[int] = Field(None, ge=0, le=2, description="Resting ECG results")
    thalach: Optional[float] = Field(None, ge=50, le=250, description="Maximum heart rate achieved")
    exang: Optional[int] = Field(None, ge=0, le=1, description="Exercise induced angina")
    oldpeak: Optional[float] = Field(None, ge=0, le=10, description="ST depression induced by exercise")
    slope: Optional[int] = Field(None, ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: Optional[int] = Field(None, ge=0, le=4, description="Number of major vessels colored by fluoroscopy")
    thal: Optional[int] = Field(None, ge=0, le=3, description="Thalassemia")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    source: Optional[str] = None
    version: str = "1.0.0"
    
    # ============================================================================
    # 🔧 Custom Validators
    # ============================================================================
    
    @validator('glucose')
    def validate_glucose(cls, v):
        """Validate glucose levels with clinical context"""
        if v is not None:
            if v > 200:
                # Log warning but don't reject - extreme values are possible
                logger.warning(f"Extreme glucose level detected: {v} mg/dL")
            elif v < 70:
                logger.warning(f"Low glucose level detected: {v} mg/dL")
        return v
    
    @validator('bmi')
    def validate_bmi(cls, v):
        """Validate BMI with clinical context"""
        if v is not None:
            if v > 40:
                logger.warning(f"Extreme BMI detected: {v}")
            elif v < 15:
                logger.warning(f"Very low BMI detected: {v}")
        return v
    
    @validator('age')
    def validate_age(cls, v):
        """Validate age with clinical context"""
        if v is not None:
            if v > 100:
                logger.warning(f"Unusual age detected: {v}")
            elif v < 0:
                raise ValueError("Age cannot be negative")
        return v
    
    @validator('blood_pressure')
    def validate_blood_pressure(cls, v):
        """Validate blood pressure"""
        if v is not None:
            if v > 180:
                logger.warning(f"Severe hypertension detected: {v} mm Hg")
            elif v < 40:
                logger.warning(f"Low blood pressure detected: {v} mm Hg")
        return v
    
    @root_validator
    def validate_complete(cls, values):
        """Validate that at least some clinical data is provided"""
        # Skip validation fields
        skip_fields = {'patient_id', 'timestamp', 'source', 'version', 'gender', 'age'}
        clinical_fields = {k: v for k, v in values.items() if k not in skip_fields}
        
        if all(v is None for v in clinical_fields.values()):
            raise ValueError("At least one clinical value must be provided")
        
        return values
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        return pd.DataFrame([self.dict(exclude_none=True)])


class PatientModel(BaseModel):
    """Patient model with PII handling"""
    
    patient_id: str = Field(..., min_length=1, max_length=50)
    name: Optional[str] = None
    date_of_birth: Optional[date] = None
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[Gender] = None
    contact: Optional[Dict[str, str]] = None
    address: Optional[Dict[str, str]] = None
    emergency_contact: Optional[Dict[str, str]] = None
    
    @validator('patient_id')
    def validate_patient_id(cls, v):
        """Validate patient ID format"""
        if not re.match(r'^[A-Z0-9]{4,20}$', v):
            raise ValueError("Patient ID must be alphanumeric, 4-20 characters")
        return v
    
    @validator('name')
    def validate_name(cls, v):
        """Validate name (no special characters)"""
        if v and not re.match(r'^[a-zA-Z\s\-\.\']+$', v):
            raise ValueError("Name contains invalid characters")
        return v
    
    def get_age(self) -> Optional[int]:
        """Calculate age from date of birth"""
        if self.date_of_birth:
            today = date.today()
            return today.year - self.date_of_birth.year - (
                (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
            )
        return self.age


class PredictionModel(BaseModel):
    """Prediction request/response model"""
    
    patient_id: str
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    probability: float = Field(..., ge=0, le=1)
    top_factors: List[str] = Field(default_factory=list)
    contributing_factors: List[Dict[str, Any]] = Field(default_factory=list)
    clinical_explanation: str
    confidence: float = Field(..., ge=0, le=1)
    drift_detected: bool = False
    processing_time_ms: float = Field(..., ge=0)
    model_version: str
    correlation_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('risk_score')
    def validate_risk_score(cls, v, values):
        """Validate risk score consistency"""
        if v >= 60 and values.get('risk_level') != RiskLevel.HIGH:
            raise ValueError("Risk score >= 60 requires HIGH risk level")
        elif v >= 30 and v < 60 and values.get('risk_level') != RiskLevel.MODERATE:
            raise ValueError("Risk score 30-59 requires MODERATE risk level")
        elif v < 30 and values.get('risk_level') != RiskLevel.LOW:
            raise ValueError("Risk score < 30 requires LOW risk level")
        return v


class BatchPredictionModel(BaseModel):
    """Batch prediction request model"""
    
    patients: List[ClinicalDataModel] = Field(..., min_items=1, max_items=1000)
    correlation_id: Optional[str] = None
    explain: bool = False
    
    @validator('patients')
    def validate_batch_size(cls, v):
        """Validate batch size"""
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 patients")
        return v
    
    @validator('patients')
    def validate_unique_patients(cls, v):
        """Ensure unique patient IDs"""
        patient_ids = [p.patient_id for p in v if p.patient_id]
        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError("Duplicate patient IDs found in batch")
        return v


# ============================================================================
# 📊 Pandera Schemas for DataFrame Validation
# ============================================================================

class ClinicalDataSchema(pa.SchemaModel):
    """
    Pandera schema for clinical DataFrames
    
    Features:
    - Statistical validation
    - Null value handling
    - Data type enforcement
    - Custom checks
    """
    
    # Demographics
    age: Series[int] = pa.Field(ge=0, le=120, nullable=True, description="Age in years")
    gender: Series[str] = pa.Field(isin=["male", "female", "other"], nullable=True)
    
    # Diabetes features
    pregnancies: Series[int] = pa.Field(ge=0, le=20, nullable=True)
    glucose: Series[float] = pa.Field(ge=0, le=300, nullable=True)
    blood_pressure: Series[float] = pa.Field(ge=0, le=200, nullable=True)
    skin_thickness: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    insulin: Series[float] = pa.Field(ge=0, le=1000, nullable=True)
    bmi: Series[float] = pa.Field(ge=10, le=60, nullable=True)
    diabetes_pedigree: Series[float] = pa.Field(ge=0, le=3, nullable=True)
    
    # Heart disease features
    cp: Series[int] = pa.Field(ge=0, le=3, nullable=True)
    trestbps: Series[float] = pa.Field(ge=0, le=300, nullable=True)
    chol: Series[float] = pa.Field(ge=0, le=600, nullable=True)
    fbs: Series[int] = pa.Field(ge=0, le=1, nullable=True)
    restecg: Series[int] = pa.Field(ge=0, le=2, nullable=True)
    thalach: Series[float] = pa.Field(ge=50, le=250, nullable=True)
    exang: Series[int] = pa.Field(ge=0, le=1, nullable=True)
    oldpeak: Series[float] = pa.Field(ge=0, le=10, nullable=True)
    slope: Series[int] = pa.Field(ge=0, le=2, nullable=True)
    ca: Series[int] = pa.Field(ge=0, le=4, nullable=True)
    thal: Series[int] = pa.Field(ge=0, le=3, nullable=True)
    
    # Target
    target: Series[int] = pa.Field(ge=0, le=1, nullable=True)
    
    # Metadata
    patient_id: Series[str] = pa.Field(str, nullable=True)
    timestamp: Series[datetime] = pa.Field(datetime, nullable=True)
    
    @pa.check("age")
    def check_age_consistency(cls, age: Series[int]) -> Series[bool]:
        """Check age consistency"""
        return (age >= 0) & (age <= 120)
    
    @pa.check("glucose")
    def check_glucose_consistency(cls, glucose: Series[float]) -> Series[bool]:
        """Check glucose consistency"""
        return (glucose >= 0) & (glucose <= 300)
    
    @pa.dataframe_check
    def check_clinical_plausibility(cls, df: pd.DataFrame) -> bool:
        """Check clinical plausibility of the data"""
        # BMI and glucose interaction
        if "bmi" in df.columns and "glucose" in df.columns:
            high_bmi_high_glucose = ((df["bmi"] > 30) & (df["glucose"] > 200)).sum()
            if high_bmi_high_glucose > len(df) * 0.5:
                logger.warning("High percentage of patients with both high BMI and high glucose")
        
        # Age and pregnancies
        if "age" in df.columns and "pregnancies" in df.columns:
            high_age_high_pregnancies = ((df["age"] < 20) & (df["pregnancies"] > 3)).sum()
            if high_age_high_pregnancies > 0:
                logger.warning("Young patients with multiple pregnancies detected")
        
        return True


class PatientSchema(pa.SchemaModel):
    """Patient data schema with PII"""
    
    patient_id: Series[str] = pa.Field(str, unique=True)
    name: Series[str] = pa.Field(str, nullable=True)
    date_of_birth: Series[datetime] = pa.Field(datetime, nullable=True)
    age: Series[int] = pa.Field(ge=0, le=120, nullable=True)
    gender: Series[str] = pa.Field(isin=["male", "female", "other"], nullable=True)
    email: Series[str] = pa.Field(str, nullable=True)
    phone: Series[str] = pa.Field(str, nullable=True)
    
    @pa.check("email")
    def check_email_format(cls, email: Series[str]) -> Series[bool]:
        """Check email format"""
        if email.isnull().all():
            return email.isnull()
        return email.str.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', na=True)


class TrainingDataSchema(pa.SchemaModel):
    """Training data schema with additional checks"""
    
    class Config:
        """Schema configuration"""
        strict = True
    
    # All features from ClinicalDataSchema
    age: Series[int] = pa.Field(ge=0, le=120)
    pregnancies: Series[int] = pa.Field(ge=0, le=20)
    glucose: Series[float] = pa.Field(ge=0, le=300)
    blood_pressure: Series[float] = pa.Field(ge=0, le=200)
    skin_thickness: Series[float] = pa.Field(ge=0, le=100)
    insulin: Series[float] = pa.Field(ge=0, le=1000)
    bmi: Series[float] = pa.Field(ge=10, le=60)
    diabetes_pedigree: Series[float] = pa.Field(ge=0, le=3)
    target: Series[int] = pa.Field(ge=0, le=1)
    
    # Training specific checks
    @pa.dataframe_check
    def check_class_balance(cls, df: pd.DataFrame) -> bool:
        """Check if both classes are represented"""
        if "target" in df.columns:
            class_counts = df["target"].value_counts()
            if len(class_counts) < 2:
                logger.warning("Only one class present in training data")
                return False
            if min(class_counts) / max(class_counts) < 0.01:
                logger.warning("Severe class imbalance detected")
        return True
    
    @pa.dataframe_check
    def check_correlation(cls, df: pd.DataFrame) -> bool:
        """Check for highly correlated features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_corr = (corr_matrix.abs() > 0.95)
            high_corr.values[np.tril_indices_from(high_corr)] = False
            if high_corr.any().any():
                logger.warning("Highly correlated features detected in training data")
        return True


# ============================================================================
# 🔧 Validation Result Models
# ============================================================================

@dataclass
class ValidationError:
    """Single validation error"""
    field: str
    message: str
    value: Any
    severity: ValidationSeverity = ValidationSeverity.CRITICAL
    
    def to_dict(self) -> Dict:
        return {
            "field": self.field,
            "message": self.message,
            "value": str(self.value) if self.value is not None else None,
            "severity": self.severity.value,
        }


@dataclass
class ValidationResult:
    """Validation result container"""
    
    is_valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info: List[ValidationError] = field(default_factory=list)
    
    def add_error(self, field: str, message: str, value: Any = None):
        """Add a critical error"""
        self.is_valid = False
        self.errors.append(ValidationError(field, message, value, ValidationSeverity.CRITICAL))
    
    def add_warning(self, field: str, message: str, value: Any = None):
        """Add a warning"""
        self.warnings.append(ValidationError(field, message, value, ValidationSeverity.WARNING))
    
    def add_info(self, field: str, message: str, value: Any = None):
        """Add an info message"""
        self.info.append(ValidationError(field, message, value, ValidationSeverity.INFO))
    
    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [e.to_dict() for e in self.warnings],
            "info": [e.to_dict() for e in self.info],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.info),
        }
    
    def __bool__(self) -> bool:
        return self.is_valid


# ============================================================================
# 🔧 Golden Tests
# ============================================================================

@dataclass
class GoldenTest:
    """Golden test case definition"""
    name: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    tolerance: float = 0.01
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "tolerance": self.tolerance,
            "enabled": self.enabled,
            "tags": self.tags,
        }


@dataclass
class GoldenTestResult:
    """Golden test result"""
    test_name: str
    passed: bool
    actual_output: Dict[str, Any]
    expected_output: Dict[str, Any]
    differences: List[str] = field(default_factory=list)
    duration_ms: float = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "differences": self.differences,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class GoldenTestSuite:
    """Suite of golden tests"""
    
    def __init__(self, name: str = "Default Golden Tests"):
        self.name = name
        self.tests: List[GoldenTest] = []
        self.results: List[GoldenTestResult] = []
        
        # Load default tests
        self._load_default_tests()
    
    def _load_default_tests(self):
        """Load default golden test cases"""
        
        # Diabetes tests
        self.tests.extend([
            GoldenTest(
                name="diabetes_high_risk",
                description="Patient with high diabetes risk",
                input_data={
                    "pregnancies": 6,
                    "glucose": 148,
                    "blood_pressure": 72,
                    "skin_thickness": 35,
                    "insulin": 0,
                    "bmi": 33.6,
                    "diabetes_pedigree": 0.627,
                    "age": 50,
                },
                expected_output={"risk_level": "High Risk", "risk_score_range": [70, 100]},
                tags=["diabetes", "high_risk"],
            ),
            GoldenTest(
                name="diabetes_low_risk",
                description="Patient with low diabetes risk",
                input_data={
                    "pregnancies": 0,
                    "glucose": 80,
                    "blood_pressure": 65,
                    "skin_thickness": 20,
                    "insulin": 10,
                    "bmi": 22.0,
                    "diabetes_pedigree": 0.2,
                    "age": 25,
                },
                expected_output={"risk_level": "Low Risk", "risk_score_range": [0, 30]},
                tags=["diabetes", "low_risk"],
            ),
        ])
        
        # Heart disease tests
        self.tests.extend([
            GoldenTest(
                name="heart_disease_high_risk",
                description="Patient with high heart disease risk",
                input_data={
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
                },
                expected_output={"risk_level": "High Risk", "risk_score_range": [70, 100]},
                tags=["heart_disease", "high_risk"],
            ),
            GoldenTest(
                name="heart_disease_low_risk",
                description="Patient with low heart disease risk",
                input_data={
                    "age": 35,
                    "sex": 0,
                    "cp": 0,
                    "trestbps": 110,
                    "chol": 180,
                    "fbs": 0,
                    "restecg": 0,
                    "thalach": 150,
                    "exang": 0,
                    "oldpeak": 0.5,
                    "slope": 1,
                    "ca": 0,
                    "thal": 0,
                },
                expected_output={"risk_level": "Low Risk", "risk_score_range": [0, 30]},
                tags=["heart_disease", "low_risk"],
            ),
        ])
    
    def add_test(self, test: GoldenTest):
        """Add a golden test"""
        self.tests.append(test)
    
    def run_tests(self, predict_fn: Callable) -> List[GoldenTestResult]:
        """
        Run all golden tests
        
        Args:
            predict_fn: Function to run prediction
            
        Returns:
            List of test results
        """
        
        self.results = []
        
        for test in self.tests:
            if not test.enabled:
                continue
            
            start_time = datetime.now()
            
            try:
                # Run prediction
                result = predict_fn(test.input_data)
                
                # Compare with expected
                passed, differences = self._compare_results(
                    result, test.expected_output, test.tolerance
                )
                
                result_obj = GoldenTestResult(
                    test_name=test.name,
                    passed=passed,
                    actual_output=result,
                    expected_output=test.expected_output,
                    differences=differences,
                    duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
                )
                
            except Exception as e:
                result_obj = GoldenTestResult(
                    test_name=test.name,
                    passed=False,
                    actual_output={},
                    expected_output=test.expected_output,
                    differences=[str(e)],
                    error=str(e),
                )
            
            self.results.append(result_obj)
        
        return self.results
    
    def _compare_results(
        self,
        actual: Dict[str, Any],
        expected: Dict[str, Any],
        tolerance: float,
    ) -> Tuple[bool, List[str]]:
        """Compare actual and expected results"""
        
        differences = []
        
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            
            if actual_value is None:
                differences.append(f"Missing key: {key}")
                continue
            
            if isinstance(expected_value, list) and len(expected_value) == 2:
                # Range check
                min_val, max_val = expected_value
                if not (min_val <= actual_value <= max_val):
                    differences.append(
                        f"{key}: expected between {min_val} and {max_val}, got {actual_value}"
                    )
            elif isinstance(expected_value, (int, float)):
                # Numeric check with tolerance
                if abs(actual_value - expected_value) > tolerance * max(1, abs(expected_value)):
                    differences.append(
                        f"{key}: expected {expected_value} ± {tolerance}, got {actual_value}"
                    )
            elif actual_value != expected_value:
                differences.append(f"{key}: expected {expected_value}, got {actual_value}")
        
        return len(differences) == 0, differences
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test suite summary"""
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        return {
            "suite_name": self.name,
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "results": [r.to_dict() for r in self.results],
        }


# ============================================================================
# 🔧 Main Validator Class
# ============================================================================

class DataValidator:
    """
    Main data validator combining Pydantic and Pandera
    
    Features:
    - Schema validation
    - Data quality checks
    - Golden tests
    - Error aggregation
    - Performance optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.pandera_schema = ClinicalDataSchema
        self.golden_suite = GoldenTestSuite()
        self._validation_cache = {}
        
        logger.info("✅ DataValidator initialized")
    
    def validate_clinical_data(
        self,
        data: Union[Dict, pd.DataFrame],
        strict: bool = False,
    ) -> ValidationResult:
        """
        Validate clinical data
        
        Args:
            data: Input data (dict or DataFrame)
            strict: Strict validation mode
            
        Returns:
            ValidationResult
        """
        
        result = ValidationResult()
        
        # Convert to DataFrame if dict
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Generate cache key
        cache_key = self._generate_cache_key(data)
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]
        
        # Step 1: Pydantic validation (row-level)
        errors = []
        for idx, row in data.iterrows():
            try:
                row_dict = row.to_dict()
                ClinicalDataModel(**row_dict)
            except PydanticValidationError as e:
                for error in e.errors():
                    field = ".".join(str(loc) for loc in error["loc"])
                    result.add_error(field, error["msg"], row.get(field))
        
        # Step 2: Pandera validation (DataFrame-level)
        try:
            self.pandera_schema.validate(data)
        except SchemaError as e:
            for error in e.errors:
                if hasattr(error, 'column') and hasattr(error, 'failure_cases'):
                    for case in error.failure_cases:
                        result.add_error(
                            error.column,
                            str(error),
                            case
                        )
        
        # Step 3: Custom clinical checks
        self._run_clinical_checks(data, result)
        
        # Cache result
        self._validation_cache[cache_key] = result
        
        return result
    
    def _run_clinical_checks(self, df: pd.DataFrame, result: ValidationResult):
        """Run custom clinical checks"""
        
        # Check for missing critical values
        critical_cols = ['glucose', 'bmi', 'age']
        for col in critical_cols:
            if col in df.columns:
                missing = df[col].isnull().sum()
                if missing > 0:
                    result.add_warning(
                        col,
                        f"{missing} missing values in critical column {col}"
                    )
        
        # Check for outliers
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr
                outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                if outliers > 0:
                    result.add_warning(
                        col,
                        f"{outliers} outliers detected in {col}",
                        f"Range: {lower:.2f} - {upper:.2f}"
                    )
        
        # Check class balance (if target exists)
        if 'target' in df.columns:
            class_counts = df['target'].value_counts()
            if len(class_counts) > 1:
                ratio = min(class_counts) / max(class_counts)
                if ratio < 0.1:
                    result.add_warning(
                        "target",
                        f"Severe class imbalance detected: {class_counts.to_dict()}"
                    )
    
    def _generate_cache_key(self, df: pd.DataFrame) -> str:
        """Generate cache key for validation"""
        df_hash = hashlib.md5(df.to_json().encode()).hexdigest()
        return f"validation:{df_hash}"
    
    def load_golden_tests(self) -> List[Dict]:
        """Load golden test cases"""
        return [t.to_dict() for t in self.golden_suite.tests]
    
    def run_golden_tests(self, predict_fn: Callable) -> Dict[str, Any]:
        """Run golden tests"""
        results = self.golden_suite.run_tests(predict_fn)
        return self.golden_suite.get_summary()
    
    def clear_cache(self):
        """Clear validation cache"""
        self._validation_cache.clear()
        logger.info("🧹 Validation cache cleared")


# ============================================================================
# 🔧 Validator Factory
# ============================================================================

class ClinicalValidator:
    """
    Clinical-specific validator with domain rules
    """
    
    @staticmethod
    def validate_diabetes_risk(data: Dict) -> ValidationResult:
        """Validate diabetes risk factors"""
        
        result = ValidationResult()
        
        # Glucose check
        glucose = data.get('glucose', 0)
        if glucose > 200:
            result.add_warning('glucose', 'Extreme glucose level > 200 mg/dL', glucose)
        
        # BMI check
        bmi = data.get('bmi', 0)
        if bmi > 35:
            result.add_warning('bmi', 'BMI > 35 - High risk', bmi)
        
        # Age and pregnancy interaction
        age = data.get('age', 0)
        pregnancies = data.get('pregnancies', 0)
        if age < 20 and pregnancies > 3:
            result.add_warning('pregnancies', 'Young patient with multiple pregnancies', pregnancies)
        
        return result
    
    @staticmethod
    def validate_heart_disease_risk(data: Dict) -> ValidationResult:
        """Validate heart disease risk factors"""
        
        result = ValidationResult()
        
        # Cholesterol check
        chol = data.get('chol', 0)
        if chol > 300:
            result.add_warning('chol', 'Very high cholesterol > 300 mg/dL', chol)
        
        # Blood pressure check
        bp = data.get('trestbps', 0)
        if bp > 160:
            result.add_warning('trestbps', 'Severe hypertension > 160 mm Hg', bp)
        
        # Age check
        age = data.get('age', 0)
        if age > 65 and data.get('exang', 0) == 1:
            result.add_warning('exang', 'Elderly patient with exercise angina', data.get('exang'))
        
        return result


# ============================================================================
# 🔧 Validator Factory Functions
# ============================================================================

_validator: Optional[DataValidator] = None


def get_validator() -> DataValidator:
    """Get validator singleton"""
    global _validator
    if _validator is None:
        _validator = DataValidator()
    return _validator


def validate_data(data: Union[Dict, pd.DataFrame]) -> ValidationResult:
    """Convenience function for data validation"""
    validator = get_validator()
    return validator.validate_clinical_data(data)


def run_golden_tests(predict_fn: Callable) -> Dict[str, Any]:
    """Convenience function for running golden tests"""
    validator = get_validator()
    return validator.run_golden_tests(predict_fn)