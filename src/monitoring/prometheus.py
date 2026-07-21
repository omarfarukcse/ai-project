# src/monitoring/prometheus.py
"""
Prometheus Metrics Export with Advanced Features
"""

from prometheus_client import (
    Counter as PromCounter,
    Histogram as PromHistogram,
    Gauge as PromGauge,
    Summary as PromSummary,
    generate_latest,
    REGISTRY,
    CollectorRegistry,
    multiprocess,
)
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import time
import asyncio
from functools import wraps

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


# ============================================================================
# 📊 Metrics Registry
# ============================================================================

class MetricsRegistry:
    """
    Centralized metrics registry with naming conventions
    
    Features:
    - Standardized metric naming
    - Label validation
    - Multi-process support
    - Registry management
    """
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self._metrics = {}
        self._labels = {}
        
        # Configure multiprocess mode if needed
        if config_manager.get("monitoring.multiprocess", False):
            multiprocess.MultiProcessCollector(self.registry)
        
        logger.info("📊 MetricsRegistry initialized")
    
    def create_counter(
        self,
        name: str,
        description: str,
        labels: List[str] = None,
        namespace: str = "cdss",
        subsystem: str = "healthcare",
    ) -> PromCounter:
        """Create a counter metric"""
        
        full_name = self._format_name(name, namespace, subsystem)
        counter = PromCounter(
            full_name,
            description,
            labels or [],
            registry=self.registry,
        )
        self._metrics[full_name] = counter
        self._labels[full_name] = labels or []
        return counter
    
    def create_histogram(
        self,
        name: str,
        description: str,
        labels: List[str] = None,
        buckets: List[float] = None,
        namespace: str = "cdss",
        subsystem: str = "healthcare",
    ) -> PromHistogram:
        """Create a histogram metric"""
        
        full_name = self._format_name(name, namespace, subsystem)
        histogram = PromHistogram(
            full_name,
            description,
            labels or [],
            buckets=buckets,
            registry=self.registry,
        )
        self._metrics[full_name] = histogram
        self._labels[full_name] = labels or []
        return histogram
    
    def create_gauge(
        self,
        name: str,
        description: str,
        labels: List[str] = None,
        namespace: str = "cdss",
        subsystem: str = "healthcare",
    ) -> PromGauge:
        """Create a gauge metric"""
        
        full_name = self._format_name(name, namespace, subsystem)
        gauge = PromGauge(
            full_name,
            description,
            labels or [],
            registry=self.registry,
        )
        self._metrics[full_name] = gauge
        self._labels[full_name] = labels or []
        return gauge
    
    def create_summary(
        self,
        name: str,
        description: str,
        labels: List[str] = None,
        namespace: str = "cdss",
        subsystem: str = "healthcare",
    ) -> PromSummary:
        """Create a summary metric"""
        
        full_name = self._format_name(name, namespace, subsystem)
        summary = PromSummary(
            full_name,
            description,
            labels or [],
            registry=self.registry,
        )
        self._metrics[full_name] = summary
        self._labels[full_name] = labels or []
        return summary
    
    def _format_name(self, name: str, namespace: str, subsystem: str) -> str:
        """Format metric name with namespace and subsystem"""
        if namespace:
            name = f"{namespace}_{name}"
        if subsystem:
            name = f"{subsystem}_{name}"
        return name
    
    def get_metric(self, name: str) -> Optional[PromCounter]:
        """Get a metric by name"""
        return self._metrics.get(name)
    
    def generate_latest(self) -> bytes:
        """Generate latest metrics"""
        return generate_latest(self.registry)
    
    def clear(self):
        """Clear all metrics"""
        self._metrics.clear()
        self._labels.clear()
        self.registry = CollectorRegistry()
        logger.info("🧹 Metrics registry cleared")


# ============================================================================
# 🚀 Prometheus Exporter
# ============================================================================

class PrometheusExporter:
    """
    Prometheus metrics exporter with:
    - Standardized metrics
    - Custom metrics
    - Multi-process support
    - HTTP endpoint
    - Metrics aggregation
    """
    
    def __init__(self):
        self.registry = MetricsRegistry()
        self._metrics = {}
        self._init_standard_metrics()
        
        logger.info("📊 PrometheusExporter initialized")
    
    def _init_standard_metrics(self):
        """Initialize standard metrics"""
        
        # API metrics
        self._metrics["api_requests_total"] = self.registry.create_counter(
            "requests_total",
            "Total API requests",
            labels=["endpoint", "method", "status"],
            namespace="cdss_api",
        )
        self._metrics["api_request_duration"] = self.registry.create_histogram(
            "request_duration_seconds",
            "API request duration in seconds",
            labels=["endpoint", "method"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            namespace="cdss_api",
        )
        self._metrics["api_active_requests"] = self.registry.create_gauge(
            "active_requests",
            "Current active requests",
            labels=["method"],
            namespace="cdss_api",
        )
        
        # Model metrics
        self._metrics["model_predictions_total"] = self.registry.create_counter(
            "predictions_total",
            "Total model predictions",
            labels=["model", "version", "risk_level"],
            namespace="cdss_model",
        )
        self._metrics["model_prediction_latency"] = self.registry.create_histogram(
            "prediction_latency_seconds",
            "Model prediction latency",
            labels=["model", "version"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            namespace="cdss_model",
        )
        self._metrics["model_errors_total"] = self.registry.create_counter(
            "errors_total",
            "Total model errors",
            labels=["model", "version", "error_type"],
            namespace="cdss_model",
        )
        
        # System metrics
        self._metrics["system_cpu_usage"] = self.registry.create_gauge(
            "cpu_usage_percent",
            "CPU usage percentage",
            namespace="cdss_system",
        )
        self._metrics["system_memory_usage"] = self.registry.create_gauge(
            "memory_usage_percent",
            "Memory usage percentage",
            namespace="cdss_system",
        )
        self._metrics["system_uptime"] = self.registry.create_gauge(
            "uptime_seconds",
            "System uptime in seconds",
            namespace="cdss_system",
        )
        
        # Business metrics
        self._metrics["business_risk_score"] = self.registry.create_histogram(
            "risk_score",
            "Patient risk scores",
            buckets=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
            namespace="cdss_business",
        )
        self._metrics["business_patients_total"] = self.registry.create_counter(
            "patients_total",
            "Total patients processed",
            labels=["risk_level"],
            namespace="cdss_business",
        )
    
    # ============================================================================
    # 🚀 Metric Recording Methods
    # ============================================================================
    
    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status: int,
        duration_ms: float,
    ):
        """Record API request metrics"""
        self._metrics["api_requests_total"].labels(
            endpoint=endpoint,
            method=method,
            status=str(status)
        ).inc()
        self._metrics["api_request_duration"].labels(
            endpoint=endpoint,
            method=method
        ).observe(duration_ms / 1000)
    
    def record_prediction(
        self,
        model: str,
        version: str,
        risk_level: str,
        latency_ms: float,
        success: bool = True,
    ):
        """Record model prediction metrics"""
        self._metrics["model_predictions_total"].labels(
            model=model,
            version=version,
            risk_level=risk_level
        ).inc()
        self._metrics["model_prediction_latency"].labels(
            model=model,
            version=version
        ).observe(latency_ms / 1000)
        
        if not success:
            self._metrics["model_errors_total"].labels(
                model=model,
                version=version,
                error_type="prediction_failed"
            ).inc()
    
    def record_system_metrics(
        self,
        cpu_percent: float,
        memory_percent: float,
        uptime_seconds: float,
    ):
        """Record system metrics"""
        self._metrics["system_cpu_usage"].set(cpu_percent)
        self._metrics["system_memory_usage"].set(memory_percent)
        self._metrics["system_uptime"].set(uptime_seconds)
    
    def record_business_metrics(
        self,
        risk_score: float,
        risk_level: str,
    ):
        """Record business metrics"""
        self._metrics["business_risk_score"].observe(risk_score)
        self._metrics["business_patients_total"].labels(
            risk_level=risk_level
        ).inc()
    
    # ============================================================================
    # 🔧 Utility Methods
    # ============================================================================
    
    def generate_metrics(self) -> bytes:
        """Generate Prometheus metrics"""
        return self.registry.generate_latest()
    
    def get_metric(self, name: str):
        """Get a metric by name"""
        return self._metrics.get(name)
    
    def list_metrics(self) -> List[str]:
        """List all registered metrics"""
        return list(self._metrics.keys())
    
    def clear(self):
        """Clear all metrics"""
        self.registry.clear()
        self._init_standard_metrics()
        logger.info("🧹 Metrics cleared")


# ============================================================================
# 🚀 Decorators for Automatic Metrics
# ============================================================================

def track_request(endpoint: str):
    """Decorator to track API requests"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            exporter = get_prometheus_exporter()
            start_time = time.time()
            status = 200
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = 500
                raise
            finally:
                duration = (time.time() - start_time) * 1000
                exporter.record_api_request(
                    endpoint=endpoint,
                    method="POST",
                    status=status,
                    duration_ms=duration,
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            exporter = get_prometheus_exporter()
            start_time = time.time()
            status = 200
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 500
                raise
            finally:
                duration = (time.time() - start_time) * 1000
                exporter.record_api_request(
                    endpoint=endpoint,
                    method="POST",
                    status=status,
                    duration_ms=duration,
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def track_prediction(model: str, version: str):
    """Decorator to track model predictions"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            exporter = get_prometheus_exporter()
            start_time = time.time()
            success = True
            risk_level = "unknown"
            
            try:
                result = await func(*args, **kwargs)
                if isinstance(result, dict):
                    risk_level = result.get("risk_level", "unknown")
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = (time.time() - start_time) * 1000
                exporter.record_prediction(
                    model=model,
                    version=version,
                    risk_level=risk_level,
                    latency_ms=duration,
                    success=success,
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            exporter = get_prometheus_exporter()
            start_time = time.time()
            success = True
            risk_level = "unknown"
            
            try:
                result = func(*args, **kwargs)
                if isinstance(result, dict):
                    risk_level = result.get("risk_level", "unknown")
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = (time.time() - start_time) * 1000
                exporter.record_prediction(
                    model=model,
                    version=version,
                    risk_level=risk_level,
                    latency_ms=duration,
                    success=success,
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_prometheus_exporter: Optional[PrometheusExporter] = None


def get_prometheus_exporter() -> PrometheusExporter:
    """Get Prometheus exporter singleton"""
    global _prometheus_exporter
    if _prometheus_exporter is None:
        _prometheus_exporter = PrometheusExporter()
    return _prometheus_exporter


Counter = PromCounter
Histogram = PromHistogram
Gauge = PromGauge
Summary = PromSummary