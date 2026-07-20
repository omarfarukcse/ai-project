# src/pipelines/__init__.py
"""
Pipeline Package - Training and Inference Pipelines

This package provides:
- Training Pipeline: End-to-end model training with validation
- Inference Pipeline: Production-grade prediction with optimizations
- Feature Engineering: Automated feature creation
- Model Validation: Golden tests and performance checks
- Pipeline Orchestration: Step-by-step execution with monitoring

Architecture:
    training_pipeline.py  → Complete training workflow
    inference_pipeline.py → Production inference with fallback

Version: 3.0.0
"""

from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.inference_pipeline import InferencePipeline

__version__ = "3.0.0"
__all__ = [
    "TrainingPipeline",
    "InferencePipeline",
]