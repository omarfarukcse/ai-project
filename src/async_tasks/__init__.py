# src/async_tasks/__init__.py
"""
Async Tasks Package - Distributed Task Processing

This package provides production-grade async task processing with:
- Celery distributed task queue
- Redis/RabbitMQ broker support
- Task scheduling and orchestration
- Retry logic with exponential backoff
- Task monitoring and observability
- Result caching and management
- Error handling and alerting

Architecture:
    celery_worker.py  → Celery configuration and worker setup
    tasks.py         → Task definitions (SHAP, reports, batch)

Features:
    - Async SHAP explanation generation
    - Batch prediction processing
    - Report generation
    - Model retraining
    - Data drift detection
    - Automated monitoring
    - Task scheduling with beat

Version: 3.0.0
"""

from src.async_tasks.celery_worker import (
    celery_app,
    CeleryConfig,
    WorkerConfig,
    get_celery_app,
    init_celery,
)
from src.async_tasks.tasks import (
    # Prediction tasks
    generate_explanation_report,
    batch_predict_async,
    
    # Model tasks
    retrain_model,
    evaluate_model,
    validate_model,
    promote_model_async,
    
    # Monitoring tasks
    check_data_drift,
    check_model_drift,
    check_bias,
    
    # Reporting tasks
    generate_weekly_report,
    generate_monthly_report,
    generate_patient_report,
    
    # Data tasks
    ingest_data,
    preprocess_data,
    feature_engineering,
    
    # Maintenance tasks
    cleanup_logs,
    cleanup_cache,
    backup_models,
    
    # Utility tasks
    send_notification,
    health_check,
)

__version__ = "3.0.0"
__all__ = [
    # Celery
    "celery_app",
    "CeleryConfig",
    "WorkerConfig",
    "get_celery_app",
    "init_celery",
    
    # Prediction tasks
    "generate_explanation_report",
    "batch_predict_async",
    
    # Model tasks
    "retrain_model",
    "evaluate_model",
    "validate_model",
    "promote_model_async",
    
    # Monitoring tasks
    "check_data_drift",
    "check_model_drift",
    "check_bias",
    
    # Reporting tasks
    "generate_weekly_report",
    "generate_monthly_report",
    "generate_patient_report",
    
    # Data tasks
    "ingest_data",
    "preprocess_data",
    "feature_engineering",
    
    # Maintenance tasks
    "cleanup_logs",
    "cleanup_cache",
    "backup_models",
    
    # Utility tasks
    "send_notification",
    "health_check",
]

import logging
logger = logging.getLogger(__name__)
logger.info(f"🚀 Async Tasks Package v{__version__} initialized")