# src/monitoring/metrics_monitor.py
"""
Enterprise Metrics Monitoring with Real-time Collection
"""

import time
import psutil
import platform
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import asyncio
import threading
from prometheus_client import Counter, Histogram, Gauge

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0
    memory_percent: float = 0
    memory_used_mb: float = 0
    disk_usage_percent: float = 0
    disk_used_gb: float = 0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    open_files: int = 0
    threads: int = 0
    processes: int = 0
    load_avg: Tuple[float, float, float] = (0, 0, 0)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "disk_usage_percent": self.disk_usage_percent,
            "disk_used_gb": self.disk_used_gb,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
            "open_files": self.open_files,
            "threads": self.threads,
            "processes": self.processes,
            "load_avg": self.load_avg,
        }


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    model_name: str = ""
    model_version: str = ""
    accuracy: float = 0
    precision: float = 0
    recall: float = 0
    f1_score: float = 0
    roc_auc: float = 0
    latency_p50: float = 0
    latency_p95: float = 0
    latency_p99: float = 0
    throughput: float = 0
    error_rate: float = 0
    request_count: int = 0
    prediction_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "throughput": self.throughput,
            "error_rate": self.error_rate,
            "request_count": self.request_count,
            "prediction_count": self.prediction_count,
        }


@dataclass
class PerformanceMetrics:
    """API performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    endpoint: str = ""
    method: str = ""
    status_code: int = 0
    latency_ms: float = 0
    request_size: int = 0
    response_size: int = 0
    correlation_id: str = ""
    user_id: str = ""
    ip_address: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "endpoint": self.endpoint,
            "method": self.method,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
            "request_size": self.request_size,
            "response_size": self.response_size,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
        }


class MetricsMonitor:
    """
    Enterprise Metrics Monitor with:
    - System metrics collection
    - Model performance tracking
    - API performance monitoring
    - Real-time dashboards
    - Historical analysis
    - Alert thresholds
    """
    
    def __init__(
        self,
        collect_interval: int = 10,
        retention_hours: int = 24,
        enable_system_metrics: bool = True,
        enable_model_metrics: bool = True,
    ):
        self.collect_interval = collect_interval
        self.retention_hours = retention_hours
        self.enable_system_metrics = enable_system_metrics
        self.enable_model_metrics = enable_model_metrics
        
        # Storage
        self.system_metrics: deque = deque(maxlen=int(retention_hours * 3600 / collect_interval))
        self.model_metrics: deque = deque(maxlen=1000)
        self.performance_metrics: deque = deque(maxlen=10000)
        
        # Aggregated metrics
        self._aggregated = {
            "system": {},
            "model": {},
            "api": {},
        }
        
        # Running state
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Prometheus metrics
        self._init_prometheus_metrics()
        
        logger.info("📊 MetricsMonitor initialized")
        logger.info(f"   Collection Interval: {collect_interval}s")
        logger.info(f"   Retention: {retention_hours}h")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # System metrics
        self._cpu_gauge = Gauge("system_cpu_percent", "CPU usage percentage")
        self._memory_gauge = Gauge("system_memory_percent", "Memory usage percentage")
        self._disk_gauge = Gauge("system_disk_percent", "Disk usage percentage")
        
        # Model metrics
        self._accuracy_gauge = Gauge("model_accuracy", "Model accuracy")
        self._latency_histogram = Histogram(
            "model_latency_seconds",
            "Model inference latency",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        self._error_counter = Counter("model_errors_total", "Total model errors")
        
        # API metrics
        self._request_counter = Counter(
            "api_requests_total",
            "Total API requests",
            ["endpoint", "method", "status"]
        )
        self._request_histogram = Histogram(
            "api_request_latency_seconds",
            "API request latency",
            ["endpoint", "method"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
    
    # ============================================================================
    # 🚀 Collection Methods
    # ============================================================================
    
    async def start_collection(self):
        """Start automatic metrics collection"""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collect_loop())
        logger.info("✅ Metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            await asyncio.gather(self._collection_task, return_exceptions=True)
        logger.info("🛑 Metrics collection stopped")
    
    async def _collect_loop(self):
        """Main collection loop"""
        while self._running:
            try:
                # Collect system metrics
                if self.enable_system_metrics:
                    sys_metrics = await self._collect_system_metrics()
                    async with self._lock:
                        self.system_metrics.append(sys_metrics)
                    self._update_prometheus_system(sys_metrics)
                
                # Collect model metrics
                if self.enable_model_metrics:
                    model_metrics = await self._collect_model_metrics()
                    if model_metrics:
                        async with self._lock:
                            self.model_metrics.append(model_metrics)
                        self._update_prometheus_model(model_metrics)
                
                # Aggregate metrics
                await self._aggregate_metrics()
                
                # Check thresholds
                await self._check_thresholds()
                
                await asyncio.sleep(self.collect_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Metrics collection error: {str(e)}")
                await asyncio.sleep(self.collect_interval)
    
    # ============================================================================
    # 🔧 Metric Collection
    # ============================================================================
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics"""
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_used_gb=disk.used / (1024 ** 3),
                network_bytes_sent=net_io.bytes_sent,
                network_bytes_recv=net_io.bytes_recv,
                open_files=len(psutil.Process().open_files()),
                threads=psutil.Process().num_threads(),
                processes=len(psutil.pids()),
                load_avg=psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0),
            )
        except Exception as e:
            logger.error(f"❌ System metrics collection failed: {str(e)}")
            return SystemMetrics()
    
    async def _collect_model_metrics(self) -> Optional[ModelMetrics]:
        """Collect model performance metrics"""
        # In production, this would query the model registry and metrics store
        # For now, return placeholder
        return None
    
    async def record_prediction(
        self,
        model_name: str,
        model_version: str,
        latency_ms: float,
        success: bool,
        prediction: Dict = None,
    ):
        """Record a prediction event"""
        
        # Update model metrics
        self._latency_histogram.observe(latency_ms / 1000)
        if not success:
            self._error_counter.inc()
        
        # Store performance metrics
        performance = PerformanceMetrics(
            endpoint="predict",
            method="POST",
            status_code=200 if success else 500,
            latency_ms=latency_ms,
        )
        async with self._lock:
            self.performance_metrics.append(performance)
    
    async def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        correlation_id: str = None,
        user_id: str = None,
        ip_address: str = None,
    ):
        """Record an API request"""
        
        # Update Prometheus
        self._request_counter.labels(endpoint=endpoint, method=method, status=status_code).inc()
        self._request_histogram.labels(endpoint=endpoint, method=method).observe(latency_ms / 1000)
        
        # Store performance metrics
        performance = PerformanceMetrics(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=latency_ms,
            correlation_id=correlation_id or "",
            user_id=user_id or "",
            ip_address=ip_address or "",
        )
        async with self._lock:
            self.performance_metrics.append(performance)
    
    # ============================================================================
    # 📊 Metric Aggregation
    # ============================================================================
    
    async def _aggregate_metrics(self):
        """Aggregate collected metrics"""
        
        # System metrics aggregation
        if self.system_metrics:
            recent = list(self.system_metrics)[-100:]
            self._aggregated["system"] = {
                "avg_cpu": sum(m.cpu_percent for m in recent) / len(recent),
                "avg_memory": sum(m.memory_percent for m in recent) / len(recent),
                "max_cpu": max(m.cpu_percent for m in recent),
                "max_memory": max(m.memory_percent for m in recent),
            }
        
        # Model metrics aggregation
        if self.model_metrics:
            recent = list(self.model_metrics)[-100:]
            self._aggregated["model"] = {
                "avg_accuracy": sum(m.accuracy for m in recent) / len(recent),
                "avg_recall": sum(m.recall for m in recent) / len(recent),
                "avg_latency": sum(m.latency_p50 for m in recent) / len(recent),
            }
        
        # API metrics aggregation
        if self.performance_metrics:
            recent = list(self.performance_metrics)[-100:]
            latencies = [m.latency_ms for m in recent]
            self._aggregated["api"] = {
                "avg_latency": sum(latencies) / len(latencies),
                "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)],
                "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)],
                "error_rate": sum(1 for m in recent if m.status_code >= 400) / len(recent),
            }
    
    def _update_prometheus_system(self, metrics: SystemMetrics):
        """Update Prometheus system metrics"""
        self._cpu_gauge.set(metrics.cpu_percent)
        self._memory_gauge.set(metrics.memory_percent)
        self._disk_gauge.set(metrics.disk_usage_percent)
    
    def _update_prometheus_model(self, metrics: ModelMetrics):
        """Update Prometheus model metrics"""
        if metrics.accuracy > 0:
            self._accuracy_gauge.set(metrics.accuracy)
    
    # ============================================================================
    # 🔧 Threshold Checking
    # ============================================================================
    
    async def _check_thresholds(self):
        """Check if any metrics exceed thresholds"""
        
        alerts = []
        
        # CPU threshold
        cpu = self._aggregated["system"].get("avg_cpu", 0)
        if cpu > 80:
            alerts.append({
                "type": "cpu_threshold",
                "severity": "critical",
                "message": f"CPU usage at {cpu:.1f}% (threshold: 80%)",
                "value": cpu,
            })
        
        # Memory threshold
        memory = self._aggregated["system"].get("avg_memory", 0)
        if memory > 85:
            alerts.append({
                "type": "memory_threshold",
                "severity": "critical",
                "message": f"Memory usage at {memory:.1f}% (threshold: 85%)",
                "value": memory,
            })
        
        # Error rate threshold
        error_rate = self._aggregated["api"].get("error_rate", 0)
        if error_rate > 0.05:  # 5% error rate
            alerts.append({
                "type": "error_rate",
                "severity": "critical",
                "message": f"Error rate at {error_rate:.2%} (threshold: 5%)",
                "value": error_rate,
            })
        
        # P95 latency threshold
        latency_p95 = self._aggregated["api"].get("p95_latency", 0)
        if latency_p95 > 1000:  # 1 second
            alerts.append({
                "type": "latency_threshold",
                "severity": "warning",
                "message": f"P95 latency at {latency_p95:.1f}ms (threshold: 1000ms)",
                "value": latency_p95,
            })
        
        if alerts:
            # Send alerts
            from src.monitoring.alerting import get_alert_manager
            alert_manager = get_alert_manager()
            for alert in alerts:
                await alert_manager.send_alert(
                    severity=alert["severity"],
                    message=alert["message"],
                    metadata={"type": alert["type"], "value": alert["value"]},
                )
    
    # ============================================================================
    # 📊 Query Methods
    # ============================================================================
    
    def get_system_metrics(self, minutes: int = 30) -> List[Dict]:
        """Get recent system metrics"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            m.to_dict()
            for m in self.system_metrics
            if m.timestamp >= cutoff
        ]
    
    def get_model_metrics(self, minutes: int = 30) -> List[Dict]:
        """Get recent model metrics"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            m.to_dict()
            for m in self.model_metrics
            if m.timestamp >= cutoff
        ]
    
    def get_performance_metrics(
        self,
        minutes: int = 30,
        endpoint: Optional[str] = None
    ) -> List[Dict]:
        """Get recent performance metrics"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        metrics = [
            m.to_dict()
            for m in self.performance_metrics
            if m.timestamp >= cutoff
        ]
        if endpoint:
            metrics = [m for m in metrics if m["endpoint"] == endpoint]
        return metrics
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics"""
        return self._aggregated
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            "timestamp": datetime.now().isoformat(),
            "system": self._aggregated.get("system", {}),
            "model": self._aggregated.get("model", {}),
            "api": self._aggregated.get("api", {}),
            "stats": {
                "system_metrics_count": len(self.system_metrics),
                "model_metrics_count": len(self.model_metrics),
                "performance_metrics_count": len(self.performance_metrics),
            },
            "status": "healthy",
        }


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_metrics_monitor: Optional[MetricsMonitor] = None


def get_metrics_monitor() -> MetricsMonitor:
    """Get metrics monitor singleton"""
    global _metrics_monitor
    if _metrics_monitor is None:
        _metrics_monitor = MetricsMonitor(
            collect_interval=config_manager.get("monitoring.collect_interval", 10),
            retention_hours=config_manager.get("monitoring.retention_hours", 24),
        )
    return _metrics_monitor