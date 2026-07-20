# src/monitoring/__init__.py
"""
Monitoring Package - Enterprise Observability Stack

This package provides production-grade monitoring with:
- System Metrics: Performance monitoring with Prometheus
- Drift Detection: Data and model drift monitoring
- Bias Monitoring: Demographic bias auditing and fairness
- Alerting: Multi-channel alerts (Slack, Email, PagerDuty)
- Golden Signals: Latency, Traffic, Errors, Saturation
- SLI/SLO Tracking: Service Level Indicators and Objectives

Architecture:
    metrics_monitor.py  → System metrics collection
    prometheus.py       → Prometheus metrics export
    drift_detection.py  → Data/model drift detection
    bias_monitor.py     → Bias and fairness monitoring
    alerting.py         → Multi-channel alerting

Features:
    - Real-time metric collection
    - Automated drift detection
    - Fairness auditing
    - Proactive alerting
    - Custom dashboards
    - Historical analysis

Version: 3.0.0
"""

from src.monitoring.metrics_monitor import (
    MetricsMonitor,
    SystemMetrics,
    ModelMetrics,
    PerformanceMetrics,
    get_metrics_monitor,
)
from src.monitoring.prometheus import (
    PrometheusExporter,
    MetricsRegistry,
    Counter,
    Histogram,
    Gauge,
    Summary,
    get_prometheus_exporter,
)
from src.monitoring.drift_detection import (
    DriftDetector,
    DriftReport,
    DataDriftDetector,
    ModelDriftDetector,
    DriftType,
    get_drift_detector,
)
from src.monitoring.bias_monitor import (
    BiasMonitor,
    BiasReport,
    FairnessMetrics,
    ProtectedAttribute,
    DemographicParity,
    EqualOpportunity,
    get_bias_monitor,
)
from src.monitoring.alerting import (
    AlertManager,
    AlertSeverity,
    AlertChannel,
    AlertRule,
    AlertStatus,
    get_alert_manager,
)

__version__ = "3.0.0"
__all__ = [
    # Metrics Monitor
    "MetricsMonitor",
    "SystemMetrics",
    "ModelMetrics",
    "PerformanceMetrics",
    "get_metrics_monitor",
    
    # Prometheus
    "PrometheusExporter",
    "MetricsRegistry",
    "Counter",
    "Histogram",
    "Gauge",
    "Summary",
    "get_prometheus_exporter",
    
    # Drift Detection
    "DriftDetector",
    "DriftReport",
    "DataDriftDetector",
    "ModelDriftDetector",
    "DriftType",
    "get_drift_detector",
    
    # Bias Monitor
    "BiasMonitor",
    "BiasReport",
    "FairnessMetrics",
    "ProtectedAttribute",
    "DemographicParity",
    "EqualOpportunity",
    "get_bias_monitor",
    
    # Alerting
    "AlertManager",
    "AlertSeverity",
    "AlertChannel",
    "AlertRule",
    "AlertStatus",
    "get_alert_manager",
]

import logging
logger = logging.getLogger(__name__)
logger.info(f"🚀 Monitoring Package v{__version__} initialized")