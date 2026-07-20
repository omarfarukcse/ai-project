# src/async_tasks/celery_worker.py
"""
Celery Worker Configuration with Advanced Features
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import timedelta
from kombu import Exchange, Queue
from celery import Celery, Task
from celery.signals import (
    worker_init,
    worker_ready,
    worker_shutdown,
    task_prerun,
    task_postrun,
    task_failure,
    task_success,
    task_retry,
)
from celery.schedules import crontab
from celery.utils.log import get_task_logger
import socket
import time
import json

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


# ============================================================================
# 📊 Celery Configuration
# ============================================================================

@dataclass
class CeleryConfig:
    """Celery configuration"""
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/1"
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: List[str] = field(default_factory=lambda: ["json"])
    timezone: str = "UTC"
    enable_utc: bool = True
    
    # Task settings
    task_track_started: bool = True
    task_time_limit: int = 1800  # 30 minutes
    task_soft_time_limit: int = 1500  # 25 minutes
    task_acks_late: bool = True
    task_reject_on_worker_lost: bool = True
    
    # Worker settings
    worker_prefetch_multiplier: int = 1
    worker_max_tasks_per_child: int = 100
    worker_concurrency: int = 4
    
    # Result settings
    result_expires: int = 86400  # 1 day
    result_compression: str = "gzip"
    
    # Broker settings
    broker_connection_retry_on_startup: bool = True
    broker_connection_retry: bool = True
    broker_connection_max_retries: int = 10
    
    # Queue settings
    task_default_queue: str = "default"
    task_default_exchange: str = "default"
    task_default_routing_key: str = "default"
    
    # Beat settings
    beat_schedule: Dict[str, Dict] = field(default_factory=dict)
    
    # Logging
    worker_redirect_stdouts: bool = False
    worker_redirect_stdouts_level: str = "INFO"
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class WorkerConfig:
    """Worker-specific configuration"""
    queues: List[str] = field(default_factory=lambda: ["default"])
    concurrency: int = 4
    max_tasks_per_child: int = 100
    pool: str = "prefork"  # prefork, eventlet, gevent, solo
    hostname: str = field(default_factory=lambda: socket.gethostname())
    log_level: str = "INFO"
    
    # Autoscale settings
    autoscale: Optional[Tuple[int, int]] = None  # (max, min)
    
    # Quality of service
    prefetch_multiplier: int = 1
    task_acks_late: bool = True


# ============================================================================
# 🚀 Celery App Creation
# ============================================================================

def create_celery_app(config: Optional[CeleryConfig] = None) -> Celery:
    """
    Create and configure Celery application
    
    Args:
        config: Celery configuration (uses defaults if None)
        
    Returns:
        Configured Celery app
    """
    
    if config is None:
        config = CeleryConfig()
    
    # Get broker URL from environment
    broker_url = os.getenv("CELERY_BROKER_URL", config.broker_url)
    result_backend = os.getenv("CELERY_RESULT_BACKEND", config.result_backend)
    
    # Create app
    app = Celery(
        "cdss_tasks",
        broker=broker_url,
        backend=result_backend,
        include=["src.async_tasks.tasks"],
    )
    
    # Update configuration
    app.config_from_object(config)
    
    # Set default queue
    app.conf.task_default_queue = config.task_default_queue
    app.conf.task_default_exchange = config.task_default_exchange
    app.conf.task_default_routing_key = config.task_default_routing_key
    
    # Configure queues
    app.conf.task_queues = (
        Queue("default", Exchange("default"), routing_key="default"),
        Queue("high_priority", Exchange("high_priority"), routing_key="high_priority"),
        Queue("low_priority", Exchange("low_priority"), routing_key="low_priority"),
        Queue("ml_tasks", Exchange("ml_tasks"), routing_key="ml_tasks"),
        Queue("reporting", Exchange("reporting"), routing_key="reporting"),
        Queue("monitoring", Exchange("monitoring"), routing_key="monitoring"),
    )
    
    # Configure routes
    app.conf.task_routes = {
        "src.async_tasks.tasks.generate_explanation_report": {"queue": "high_priority"},
        "src.async_tasks.tasks.batch_predict_async": {"queue": "ml_tasks"},
        "src.async_tasks.tasks.retrain_model": {"queue": "ml_tasks"},
        "src.async_tasks.tasks.generate_weekly_report": {"queue": "reporting"},
        "src.async_tasks.tasks.generate_monthly_report": {"queue": "reporting"},
        "src.async_tasks.tasks.check_data_drift": {"queue": "monitoring"},
        "src.async_tasks.tasks.check_model_drift": {"queue": "monitoring"},
        "src.async_tasks.tasks.cleanup_logs": {"queue": "low_priority"},
        "src.async_tasks.tasks.cleanup_cache": {"queue": "low_priority"},
    }
    
    # Beat schedule
    app.conf.beat_schedule = {
        # Daily training
        "retrain-model-daily": {
            "task": "src.async_tasks.tasks.retrain_model",
            "schedule": crontab(hour=2, minute=0),
            "options": {"queue": "ml_tasks"},
        },
        # Hourly drift check
        "check-drift-hourly": {
            "task": "src.async_tasks.tasks.check_data_drift",
            "schedule": crontab(minute=0),
            "options": {"queue": "monitoring"},
        },
        # Weekly report
        "weekly-report": {
            "task": "src.async_tasks.tasks.generate_weekly_report",
            "schedule": crontab(day_of_week=1, hour=3, minute=0),
            "options": {"queue": "reporting"},
        },
        # Monthly report
        "monthly-report": {
            "task": "src.async_tasks.tasks.generate_monthly_report",
            "schedule": crontab(day_of_month=1, hour=4, minute=0),
            "options": {"queue": "reporting"},
        },
        # Maintenance
        "cleanup-logs": {
            "task": "src.async_tasks.tasks.cleanup_logs",
            "schedule": crontab(hour=5, minute=0),
            "options": {"queue": "low_priority"},
        },
        "cleanup-cache": {
            "task": "src.async_tasks.tasks.cleanup_cache",
            "schedule": crontab(hour=5, minute=30),
            "options": {"queue": "low_priority"},
        },
        # Model drift check
        "check-model-drift": {
            "task": "src.async_tasks.tasks.check_model_drift",
            "schedule": crontab(hour="*/6", minute=0),  # Every 6 hours
            "options": {"queue": "monitoring"},
        },
        # Health check
        "health-check": {
            "task": "src.async_tasks.tasks.health_check",
            "schedule": crontab(minute="*/5"),
            "options": {"queue": "monitoring"},
        },
    }
    
    # Register signals
    setup_signals(app)
    
    logger.info("✅ Celery app configured")
    logger.info(f"   Broker: {broker_url}")
    logger.info(f"   Backend: {result_backend}")
    logger.info(f"   Queues: {len(app.conf.task_queues)}")
    logger.info(f"   Beat Tasks: {len(app.conf.beat_schedule)}")
    
    return app


# ============================================================================
# 📊 Signal Handlers
# ============================================================================

def setup_signals(app: Celery):
    """Setup Celery signal handlers"""
    
    @worker_init.connect
    def on_worker_init(sender, **kwargs):
        """Called when worker initializes"""
        logger.info(f"🚀 Worker initializing: {sender.hostname}")
        logger.info(f"   Queues: {sender.consumer.task_consumer.queues}")
    
    @worker_ready.connect
    def on_worker_ready(sender, **kwargs):
        """Called when worker is ready"""
        logger.info(f"✅ Worker ready: {sender.hostname}")
        logger.info(f"   Concurrency: {sender.concurrency}")
        logger.info(f"   Prefetch: {sender.prefetch_multiplier}")
    
    @worker_shutdown.connect
    def on_worker_shutdown(sender, **kwargs):
        """Called when worker shuts down"""
        logger.info(f"🛑 Worker shutting down: {sender.hostname}")
    
    @task_prerun.connect
    def on_task_prerun(sender, task_id, task, args, kwargs, **extra):
        """Called before a task starts"""
        logger.info(f"▶️ Task starting: {task.name} ({task_id})")
        logger.info(f"   Args: {len(args)}, Kwargs: {len(kwargs)}")
    
    @task_postrun.connect
    def on_task_postrun(sender, task_id, task, args, kwargs, retval, state, **extra):
        """Called after a task completes"""
        logger.info(f"✅ Task completed: {task.name} ({task_id})")
        logger.info(f"   State: {state}")
    
    @task_success.connect
    def on_task_success(sender, result, **kwargs):
        """Called on task success"""
        task_name = sender.name if hasattr(sender, 'name') else 'unknown'
        task_id = sender.request.id if hasattr(sender, 'request') else 'unknown'
        logger.info(f"✅ Task succeeded: {task_name} ({task_id})")
    
    @task_failure.connect
    def on_task_failure(sender, task_id, exception, args, kwargs, traceback, einfo, **extra):
        """Called on task failure"""
        task_name = sender.name if hasattr(sender, 'name') else 'unknown'
        logger.error(f"❌ Task failed: {task_name} ({task_id})")
        logger.error(f"   Exception: {exception}")
        logger.error(f"   Traceback: {traceback}")
        
        # Send alert on critical failures
        from src.monitoring.alerting import get_alert_manager
        from src.monitoring.alerting import AlertSeverity
        
        alert_manager = get_alert_manager()
        asyncio.create_task(
            alert_manager.send_alert(
                severity=AlertSeverity.HIGH,
                message=f"Task failed: {task_name} ({task_id})",
                metadata={
                    "task_id": task_id,
                    "task_name": task_name,
                    "exception": str(exception),
                }
            )
        )
    
    @task_retry.connect
    def on_task_retry(sender, request, reason, einfo, **extra):
        """Called on task retry"""
        task_name = sender.name if hasattr(sender, 'name') else 'unknown'
        logger.warning(f"🔄 Task retry: {task_name} ({request.id})")
        logger.warning(f"   Reason: {reason}")


# ============================================================================
# 🔧 Custom Task Base Class
# ============================================================================

class BaseTask(Task):
    """
    Base task class with common functionality
    
    Features:
    - Automatic retry with exponential backoff
    - Result caching
    - Performance tracking
    - Error handling
    - Alerting
    """
    
    abstract = True
    max_retries = 3
    default_retry_delay = 60  # seconds
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Task {task_id} failed: {str(exc)}")
        
        # Cache failure
        from src.caching.redis_client import get_fast_redis_client
        redis_client = get_fast_redis_client()
        
        failure_data = {
            "task_id": task_id,
            "task_name": self.name,
            "exception": str(exc),
            "args": args,
            "kwargs": kwargs,
            "timestamp": time.time(),
        }
        
        asyncio.create_task(
            redis_client.set(
                f"task_failure:{task_id}",
                failure_data,
                ttl=86400 * 7  # 7 days
            )
        )
        
        # Send alert if critical
        if hasattr(self, 'alert_on_failure') and self.alert_on_failure:
            from src.monitoring.alerting import get_alert_manager
            from src.monitoring.alerting import AlertSeverity
            
            alert_manager = get_alert_manager()
            asyncio.create_task(
                alert_manager.send_alert(
                    severity=AlertSeverity.HIGH,
                    message=f"Critical task failed: {self.name}",
                    metadata=failure_data,
                )
            )
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry"""
        logger.warning(f"Task {task_id} retrying: {str(exc)}")
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        logger.info(f"Task {task_id} completed successfully")
        
        # Cache result if requested
        if hasattr(self, 'cache_result') and self.cache_result:
            from src.caching.redis_client import get_fast_redis_client
            redis_client = get_fast_redis_client()
            
            asyncio.create_task(
                redis_client.set(
                    f"task_result:{task_id}",
                    {
                        "result": retval,
                        "timestamp": time.time(),
                    },
                    ttl=86400  # 1 day
                )
            )
    
    def __call__(self, *args, **kwargs):
        """Execute task with retry logic"""
        try:
            return super().__call__(*args, **kwargs)
        except Exception as exc:
            if hasattr(self, 'max_retries'):
                retries = self.request.retries if hasattr(self.request, 'retries') else 0
                if retries < self.max_retries:
                    # Exponential backoff
                    delay = self.default_retry_delay * (2 ** retries)
                    self.retry(exc=exc, countdown=min(delay, 3600))
            raise


# ============================================================================
# 🔧 Task Result Caching
# ============================================================================

class CachedTask(BaseTask):
    """
    Task with automatic result caching
    
    Features:
    - Cache task results in Redis
    - Automatic cache invalidation
    - Cache TTL management
    """
    
    abstract = True
    cache_result = True
    cache_ttl = 3600  # 1 hour
    cache_key_prefix = "task_cache:"
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        import hashlib
        import json
        
        key_data = {
            "task": self.name,
            "args": args,
            "kwargs": kwargs,
        }
        key_hash = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        return f"{self.cache_key_prefix}{self.name}:{key_hash}"
    
    def __call__(self, *args, **kwargs):
        """Execute with caching"""
        if not hasattr(self, 'cache_result') or not self.cache_result:
            return super().__call__(*args, **kwargs)
        
        # Check cache
        from src.caching.redis_client import get_fast_redis_client
        
        cache_key = self.get_cache_key(*args, **kwargs)
        redis_client = get_fast_redis_client()
        
        # Try to get cached result
        cached_result = asyncio.run(redis_client.get(cache_key))
        if cached_result:
            logger.info(f"✅ Cache hit for {self.name}")
            return cached_result
        
        # Execute task
        result = super().__call__(*args, **kwargs)
        
        # Cache result
        if result is not None:
            asyncio.run(redis_client.set(cache_key, result, ttl=self.cache_ttl))
        
        return result


# ============================================================================
# 🔧 Celery App Singleton
# ============================================================================

_celery_app: Optional[Celery] = None


def get_celery_app() -> Celery:
    """Get Celery app singleton"""
    global _celery_app
    if _celery_app is None:
        _celery_app = create_celery_app()
    return _celery_app


def init_celery(worker_config: Optional[WorkerConfig] = None):
    """
    Initialize Celery worker with configuration
    
    Args:
        worker_config: Worker configuration
    """
    
    app = get_celery_app()
    
    if worker_config:
        app.conf.update(
            worker_concurrency=worker_config.concurrency,
            worker_max_tasks_per_child=worker_config.max_tasks_per_child,
            worker_prefetch_multiplier=worker_config.prefetch_multiplier,
            task_acks_late=worker_config.task_acks_late,
        )
    
    logger.info("✅ Celery worker initialized")
    return app


# ============================================================================
# 🚀 Main Entry Point
# ============================================================================

# Create app instance
celery_app = get_celery_app()

# Export for Celery worker
app = celery_app

if __name__ == "__main__":
    # Start Celery worker
    import subprocess
    import sys
    
    cmd = [
        "celery",
        "-A", "src.async_tasks.celery_worker",
        "worker",
        "--loglevel=info",
        "--concurrency=4",
        "--hostname=cdss_worker@%h",
        "--queues=default,high_priority,low_priority,ml_tasks,reporting,monitoring",
    ]
    
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    subprocess.run(cmd)