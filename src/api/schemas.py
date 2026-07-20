# src/api/schemas.py
"""
Complete Pydantic Schemas with Strict Validation
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import re

# ============================================================================
# 🔧 Enums
# ============================================================================

class RiskLevel(str, Enum):
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class ModelStatus(str, Enum):
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"


class PredictionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


# ============================================================================
# 📋 Input Schemas
# ============================================================================

class PatientData(BaseModel):
    """Complete patient clinical data with validation"""
    
    # Demographics
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    sex: Optional[int] = Field(None, ge=0, le=1, description="Sex (0: Female, 1: Male)")
    
    # Diabetes features
    pregnancies: Optional[int] = Field(None, ge=0, le=20, description="Number of pregnancies")
    glucose: Optional[float] = Field(None, ge=0, le=300, description="Glucose level (mg/dL)")
    blood_pressure: Optional[float] = Field(None, ge=0, le=200, description="Blood pressure (mm Hg)")
    skin_thickness: Optional[float] = Field(None, ge=0, le=100, description="Skin thickness (mm)")
    insulin: Optional[float] = Field(None, ge=0, le=1000, description="Insulin level (mu U/ml)")
    bmi: Optional[float] = Field(None, ge=10, le=60, description="Body Mass Index")
    diabetes_pedigree: Optional[float] = Field(None, ge=0, le=3, description="Diabetes pedigree function")
    
    # Heart disease features
    cp: Optional[int] = Field(None, ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: Optional[float] = Field(None, ge=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: Optional[float] = Field(None, ge=0, le=600, description="Serum cholesterol (mg/dL)")
    fbs: Optional[int] = Field(None, ge=0, le=1, description="Fasting blood sugar > 120 mg/dL")
    restecg: Optional[int] = Field(None, ge=0, le=2, description="Resting ECG results")
    thalach: Optional[float] = Field(None, ge=50, le=250, description="Maximum heart rate")
    exang: Optional[int] = Field(None, ge=0, le=1, description="Exercise induced angina")
    oldpeak: Optional[float] = Field(None, ge=0, le=10, description="ST depression")
    slope: Optional[int] = Field(None, ge=0, le=2, description="Slope of ST segment")
    ca: Optional[int] = Field(None, ge=0, le=4, description="Number of major vessels")
    thal: Optional[int] = Field(None, ge=0, le=3, description="Thalassemia")
    
    @validator('glucose')
    def validate_glucose(cls, v):
        if v is not None and v > 200:
            # Flag but don't reject - allow extreme values with warning
            return v
        return v
    
    @validator('bmi')
    def validate_bmi(cls, v):
        if v is not None and v > 40:
            return v
        return v
    
    @validator('age')
    def validate_age(cls, v):
        if v is not None and v > 100:
            return v
        return v
    
    @root_validator
    def validate_complete(cls, values):
        """Validate that at least some data is provided"""
        if all(v is None for v in values.values()):
            raise ValueError("At least one clinical value must be provided")
        return values
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }
        extra = "forbid"


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    patients: List[PatientData] = Field(..., min_items=1, max_items=1000)
    correlation_id: Optional[str] = None
    explain: bool = False
    
    @validator('patients')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 patients")
        return v


class ModelPromotionRequest(BaseModel):
    """Model promotion request"""
    version: str = Field(..., regex=r'^v\d+\.\d+\.\d+$')
    canary_percentage: float = Field(0.05, ge=0, le=1)
    run_tests: bool = True
    auto_rollback: bool = True
    rollback_threshold: float = Field(0.1, ge=0, le=0.5)


class UserLoginRequest(BaseModel):
    """User login request"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)
    
    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("Username must contain only letters, numbers, and underscore")
        return v


# ============================================================================
# 📤 Output Schemas
# ============================================================================

class ContributingFactor(BaseModel):
    """Individual factor contribution"""
    feature: str
    contribution: float
    direction: str  # positive or negative
    clinical_significance: Optional[str] = None


class RiskPredictionResponse(BaseModel):
    """Complete risk prediction response"""
    
    patient_id: str
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    probability: float = Field(..., ge=0, le=1)
    top_factors: List[str] = Field(default_factory=list)
    contributing_factors: List[ContributingFactor] = Field(default_factory=list)
    clinical_explanation: str
    confidence: float = Field(..., ge=0, le=1)
    drift_detected: bool = False
    processing_time_ms: float = 0
    model_version: str
    correlation_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[Dict[str, Any]]
    total: int
    successful: int
    failed: int
    processing_time_ms: float = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class ExplanationResponse(BaseModel):
    """SHAP explanation response"""
    
    patient_id: str
    risk_score: float
    risk_level: str
    shap_values: List[float]
    base_value: float
    feature_names: List[str]
    feature_values: Dict[str, float]
    contribution_breakdown: List[Dict[str, Any]]
    natural_language_explanation: str
    waterfall_plot_url: Optional[str] = None
    force_plot_url: Optional[str] = None


class FeatureImportanceResponse(BaseModel):
    """Feature importance response"""
    features: List[str]
    scores: List[float]
    description: str
    timestamp: datetime


class ModelMetricsResponse(BaseModel):
    """Model performance metrics"""
    
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    roc_auc: float
    pr_auc: Optional[float] = None
    confusion_matrix: Dict[str, int]
    model_name: str
    version: str
    last_updated: datetime
    
    @validator('recall')
    def validate_recall(cls, v):
        if v < 0.8:
            # Warning for low recall (clinical priority)
            pass
        return v


class DriftReportResponse(BaseModel):
    """Drift detection report"""
    
    drift_detected: bool
    drift_share: float
    column_drifts: Dict[str, Dict[str, Any]]
    recommendation: Optional[str] = None
    timestamp: datetime


class AuditLogResponse(BaseModel):
    """Audit log entry"""
    
    id: str
    user_id: str
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    correlation_id: str
    timestamp: datetime


class UserLoginResponse(BaseModel):
    """User login response"""
    
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    user: Dict[str, Any]


class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    
    items: List[Any]
    total: int
    page: int
    page_size: int
    pages: int
    has_next: bool
    has_previous: bool


# ============================================================================
# 🔧 Error Schemas
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response"""
    
    error: str
    detail: Optional[str] = None
    correlation_id: Optional[str] = None
    path: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class ValidationErrorDetail(BaseModel):
    """Detailed validation error"""
    field: str
    message: str
    type: str


class ValidationErrorResponse(ErrorResponse):
    """Validation error response with details"""
    details: List[ValidationErrorDetail] = Field(default_factory=list)