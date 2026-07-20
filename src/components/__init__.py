# src/components/__init__.py
"""
Components Package - Complete ML Pipeline Components

This package provides all core ML components with enterprise features:
- Data Ingestion: Multi-source data loading with validation
- Preprocessing: Advanced feature engineering and transformation
- Model Training: Multi-model training with optimization
- Model Evaluation: Comprehensive metrics with clinical focus
- Model Calibration: Platt/Isotonic scaling for reliable probabilities
- Model Registry: MLflow-based model versioning and staging
- Explainability: SHAP-based model interpretation
- Reporting: Automated report generation
- Clinical Decision: Risk scoring and clinical recommendations
- Fallback System: Rule-based fallback for high availability
- Human Review: Human-in-the-loop for high-risk predictions

Version: 3.0.0
"""

from src.components.data_ingestion import DataIngestion
from src.components.preprocessing import ClinicalPreprocessor
from src.components.model_training import ClinicalModelTrainer
from src.components.model_evaluation import ModelEvaluator
from src.components.model_calibration import ModelCalibrator
from src.components.model_registry import ModelRegistry
from src.components.explainability import ClinicalSHAPExplainer
from src.components.reporting import ClinicalReporter
from src.components.clinical_decision import ClinicalDecisionEngine
from src.components.fallback_system import FallbackSystem
from src.components.human_review import HumanReviewSystem

__version__ = "3.0.0"
__all__ = [
    "DataIngestion",
    "ClinicalPreprocessor",
    "ClinicalModelTrainer",
    "ModelEvaluator",
    "ModelCalibrator",
    "ModelRegistry",
    "ClinicalSHAPExplainer",
    "ClinicalReporter",
    "ClinicalDecisionEngine",
    "FallbackSystem",
    "HumanReviewSystem",
]