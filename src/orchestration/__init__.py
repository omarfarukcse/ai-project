# src/orchestration/__init__.py
"""
Orchestration Package - Enterprise Workflow Management

This package provides production-grade workflow orchestration with:
- Apache Airflow DAGs for ML pipelines
- Task scheduling with dependency management
- Retry logic and error handling
- Monitoring and alerting
- Resource optimization
- SLA tracking

Architecture:
    airflow_dag.py  → Airflow DAG definitions for ML workflows
    scheduler.py    → Task scheduling and coordination

Features:
    - Automated daily retraining
    - Data pipeline orchestration
    - Model validation and promotion
    - Drift detection scheduling
    - Report generation automation
    - Slack/email notifications

Version: 3.0.0
"""

from src.orchestration.airflow_dag import (
    create_training_dag,
    create_validation_dag,
    create_monitoring_dag,
    DAGConfig,
    TaskConfig,
)
from src.orchestration.scheduler import (
    TaskScheduler,
    ScheduledTask,
    TaskStatus,
    TaskPriority,
    get_scheduler,
)

__version__ = "3.0.0"
__all__ = [
    # Airflow DAGs
    "create_training_dag",
    "create_validation_dag",
    "create_monitoring_dag",
    "DAGConfig",
    "TaskConfig",
    
    # Scheduler
    "TaskScheduler",
    "ScheduledTask",
    "TaskStatus",
    "TaskPriority",
    "get_scheduler",
]

import logging
logger = logging.getLogger(__name__)
logger.info(f"🚀 Orchestration Package v{__version__} initialized")