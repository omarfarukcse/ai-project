# src/monitoring/alerting.py
"""
Multi-Channel Alerting System with Severity Levels
"""

import json
import requests
import smtplib
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertChannel(Enum):
    """Alert channels"""
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    CONSOLE = "console"


class AlertStatus(Enum):
    """Alert status"""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: Callable[[Dict], bool]
    severity: AlertSeverity = AlertSeverity.MEDIUM
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.CONSOLE])
    cooldown_seconds: int = 300  # 5 minutes
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    
    def should_trigger(self, data: Dict) -> bool:
        """Check if alert should be triggered"""
        if not self.enabled:
            return False
        
        if self.last_triggered:
            cooldown = timedelta(seconds=self.cooldown_seconds)
            if datetime.now() - self.last_triggered < cooldown:
                return False
        
        try:
            return self.condition(data)
        except Exception as e:
            logger.error(f"❌ Alert rule condition failed: {str(e)}")
            return False


@dataclass
class Alert:
    """Alert instance"""
    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    status: AlertStatus = AlertStatus.PENDING
    channel: Optional[AlertChannel] = None
    sent_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None


class AlertManager:
    """
    Enterprise Alerting System with:
    - Multi-channel notifications (Slack, Email, PagerDuty)
    - Severity-based routing
    - Deduplication and cooldown
    - Alert acknowledgment
    - Escalation policies
    - Audit trail
    """
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_history: List[Alert] = []
        self._max_history = 1000
        self._pending_alerts: List[Alert] = []
        self._notification_queue: asyncio.Queue = asyncio.Queue()
        
        # Configuration
        self._config = config_manager.get("alerting", {})
        self._init_notification_handlers()
        
        # Start notification worker
        self._worker_task = asyncio.create_task(self._notification_worker())
        
        logger.info("🔔 AlertManager initialized")
        logger.info(f"   Alert Rules: {len(self.alert_rules)}")
    
    def _init_notification_handlers(self):
        """Initialize notification handlers"""
        self._handlers = {
            AlertChannel.SLACK: self._send_slack,
            AlertChannel.EMAIL: self._send_email,
            AlertChannel.PAGERDUTY: self._send_pagerduty,
            AlertChannel.WEBHOOK: self._send_webhook,
            AlertChannel.CONSOLE: self._send_console,
        }
        
        # Register default alert rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default alert rules"""
        
        # System health rules
        self.register_rule(
            name="high_cpu_usage",
            condition=lambda d: d.get("cpu_percent", 0) > 80,
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
        )
        
        self.register_rule(
            name="high_memory_usage",
            condition=lambda d: d.get("memory_percent", 0) > 85,
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
        )
        
        self.register_rule(
            name="disk_usage_high",
            condition=lambda d: d.get("disk_percent", 0) > 90,
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.SLACK],
        )
        
        # Model performance rules
        self.register_rule(
            name="model_performance_drop",
            condition=lambda d: d.get("recall_drop", 0) > 0.1,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.SLACK, AlertChannel.PAGERDUTY],
        )
        
        self.register_rule(
            name="high_error_rate",
            condition=lambda d: d.get("error_rate", 0) > 0.05,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
        )
        
        self.register_rule(
            name="high_latency",
            condition=lambda d: d.get("p95_latency_ms", 0) > 1000,
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.SLACK],
        )
        
        # Drift detection rules
        self.register_rule(
            name="data_drift_detected",
            condition=lambda d: d.get("drift_detected", False),
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
        )
        
        self.register_rule(
            name="model_drift_detected",
            condition=lambda d: d.get("model_drift_detected", False),
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.SLACK, AlertChannel.PAGERDUTY],
        )
        
        # Bias rules
        self.register_rule(
            name="fairness_violation",
            condition=lambda d: len(d.get("fairness_violations", [])) > 0,
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
        )
    
    # ============================================================================
    # 🚀 Alert Rule Management
    # ============================================================================
    
    def register_rule(
        self,
        name: str,
        condition: Callable[[Dict], bool],
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        channels: List[AlertChannel] = None,
        cooldown_seconds: int = 300,
        enabled: bool = True,
    ) -> str:
        """Register a new alert rule"""
        
        rule = AlertRule(
            name=name,
            condition=condition,
            severity=severity,
            channels=channels or [AlertChannel.CONSOLE],
            cooldown_seconds=cooldown_seconds,
            enabled=enabled,
        )
        
        self.alert_rules[name] = rule
        logger.info(f"✅ Alert rule registered: {name}")
        
        return name
    
    def enable_rule(self, name: str):
        """Enable an alert rule"""
        if name in self.alert_rules:
            self.alert_rules[name].enabled = True
            logger.info(f"✅ Alert rule enabled: {name}")
    
    def disable_rule(self, name: str):
        """Disable an alert rule"""
        if name in self.alert_rules:
            self.alert_rules[name].enabled = False
            logger.info(f"⏹️ Alert rule disabled: {name}")
    
    def delete_rule(self, name: str):
        """Delete an alert rule"""
        if name in self.alert_rules:
            del self.alert_rules[name]
            logger.info(f"🗑️ Alert rule deleted: {name}")
    
    # ============================================================================
    # 🚀 Alert Evaluation and Sending
    # ============================================================================
    
    async def evaluate_alerts(self, data: Dict[str, Any]):
        """Evaluate all alert rules against data"""
        
        for rule_name, rule in self.alert_rules.items():
            if rule.should_trigger(data):
                # Create alert
                alert = Alert(
                    id=f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}_{rule_name}",
                    rule_name=rule_name,
                    severity=rule.severity,
                    message=self._format_message(rule_name, data),
                    details=data,
                )
                
                # Add to queue
                self._pending_alerts.append(alert)
                await self._notification_queue.put(alert)
                
                # Update rule
                rule.last_triggered = datetime.now()
                
                logger.warning(f"⚠️ Alert triggered: {rule_name} ({rule.severity.value})")
    
    async def send_alert(
        self,
        severity: AlertSeverity,
        message: str,
        metadata: Dict = None,
        channels: List[AlertChannel] = None,
    ):
        """Send an alert immediately"""
        
        alert = Alert(
            id=f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}_manual",
            rule_name="manual",
            severity=severity,
            message=message,
            details=metadata or {},
        )
        
        channels = channels or [AlertChannel.SLACK, AlertChannel.EMAIL]
        
        for channel in channels:
            if channel in self._handlers:
                try:
                    await self._handlers[channel](alert)
                    alert.status = AlertStatus.SENT
                    alert.sent_at = datetime.now()
                    alert.channel = channel
                    logger.info(f"✅ Alert sent: {channel.value} - {alert.id}")
                except Exception as e:
                    alert.status = AlertStatus.FAILED
                    logger.error(f"❌ Failed to send alert via {channel.value}: {str(e)}")
        
        self.alert_history.append(alert)
        if len(self.alert_history) > self._max_history:
            self.alert_history = self.alert_history[-self._max_history:]
    
    # ============================================================================
    # 🔧 Notification Handlers
    # ============================================================================
    
    async def _notification_worker(self):
        """Worker to process notification queue"""
        
        while True:
            try:
                alert = await self._notification_queue.get()
                
                # Send to configured channels
                rule = self.alert_rules.get(alert.rule_name)
                channels = rule.channels if rule else [AlertChannel.CONSOLE]
                
                for channel in channels:
                    if channel in self._handlers:
                        try:
                            await self._handlers[channel](alert)
                            alert.status = AlertStatus.SENT
                            alert.sent_at = datetime.now()
                            alert.channel = channel
                            logger.info(f"✅ Alert sent: {channel.value} - {alert.id}")
                        except Exception as e:
                            alert.status = AlertStatus.FAILED
                            logger.error(f"❌ Failed to send alert via {channel.value}: {str(e)}")
                
                self.alert_history.append(alert)
                if len(self.alert_history) > self._max_history:
                    self.alert_history = self.alert_history[-self._max_history:]
                
                self._notification_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Notification worker error: {str(e)}")
                await asyncio.sleep(1)
    
    async def _send_slack(self, alert: Alert):
        """Send alert to Slack"""
        
        webhook_url = self._config.get("slack_webhook_url")
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return
        
        # Determine color based on severity
        colors = {
            AlertSeverity.CRITICAL: "#ff0000",
            AlertSeverity.HIGH: "#ff6600",
            AlertSeverity.MEDIUM: "#ffcc00",
            AlertSeverity.LOW: "#00ccff",
            AlertSeverity.INFO: "#00cc66",
        }
        
        payload = {
            "attachments": [{
                "color": colors.get(alert.severity, "#cccccc"),
                "title": f"🚨 {alert.severity.value.upper()}: {alert.rule_name}",
                "text": alert.message,
                "fields": [
                    {"title": "Timestamp", "value": alert.timestamp.isoformat(), "short": True},
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                ],
                "footer": "CDSS Alert System",
                "ts": int(alert.timestamp.timestamp()),
            }]
        }
        
        # Add details
        if alert.details:
            details_text = "\n".join([f"• {k}: {v}" for k, v in alert.details.items()])
            payload["attachments"][0]["fields"].append({
                "title": "Details",
                "value": details_text,
                "short": False,
            })
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=5)
            response.raise_for_status()
        except Exception as e:
            raise Exception(f"Slack notification failed: {str(e)}")
    
    async def _send_email(self, alert: Alert):
        """Send alert via email"""
        
        smtp_config = self._config.get("email", {})
        if not smtp_config.get("enabled", False):
            return
        
        try:
            msg = MIMEMultipart()
            msg["From"] = smtp_config.get("from", "alerts@cdss-healthcare.com")
            msg["To"] = smtp_config.get("to", "admin@cdss-healthcare.com")
            msg["Subject"] = f"[CDSS] {alert.severity.value.upper()}: {alert.rule_name}"
            
            body = f"""
            Alert: {alert.rule_name}
            Severity: {alert.severity.value}
            Timestamp: {alert.timestamp.isoformat()}
            
            Message: {alert.message}
            
            Details:
            {json.dumps(alert.details, indent=2)}
            
            Please investigate and take appropriate action.
            """
            
            msg.attach(MIMEText(body, "plain"))
            
            # Send email
            server = smtplib.SMTP(
                smtp_config.get("host", "smtp.gmail.com"),
                smtp_config.get("port", 587)
            )
            server.starttls()
            
            if smtp_config.get("username") and smtp_config.get("password"):
                server.login(
                    smtp_config["username"],
                    smtp_config["password"]
                )
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            raise Exception(f"Email notification failed: {str(e)}")
    
    async def _send_pagerduty(self, alert: Alert):
        """Send alert to PagerDuty"""
        
        pd_config = self._config.get("pagerduty", {})
        if not pd_config.get("enabled", False):
            return
        
        try:
            # PagerDuty events API
            url = "https://events.pagerduty.com/v2/enqueue"
            
            payload = {
                "routing_key": pd_config.get("routing_key"),
                "event_action": "trigger",
                "payload": {
                    "summary": f"{alert.severity.value.upper()}: {alert.rule_name}",
                    "severity": alert.severity.value,
                    "source": "cdss-healthcare",
                    "component": alert.rule_name,
                    "group": "ml-pipeline",
                    "custom_details": {
                        "message": alert.message,
                        "details": alert.details,
                        "timestamp": alert.timestamp.isoformat(),
                    }
                },
                "dedup_key": f"{alert.rule_name}_{alert.timestamp.date()}",
                "client": "CDSS Alert System",
                "client_url": "https://cdss-healthcare.com",
            }
            
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            
        except Exception as e:
            raise Exception(f"PagerDuty notification failed: {str(e)}")
    
    async def _send_webhook(self, alert: Alert):
        """Send alert to custom webhook"""
        
        webhook_url = self._config.get("webhook_url")
        if not webhook_url:
            return
        
        try:
            response = requests.post(
                webhook_url,
                json={
                    "alert": {
                        "id": alert.id,
                        "rule": alert.rule_name,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "details": alert.details,
                    }
                },
                timeout=5,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            
        except Exception as e:
            raise Exception(f"Webhook notification failed: {str(e)}")
    
    async def _send_console(self, alert: Alert):
        """Log alert to console"""
        
        print(f"""
        {'='*60}
        🔔 ALERT: {alert.rule_name}
        {'='*60}
        Severity: {alert.severity.value}
        Timestamp: {alert.timestamp.isoformat()}
        Message: {alert.message}
        Details: {json.dumps(alert.details, indent=2)}
        {'='*60}
        """)
    
    def _format_message(self, rule_name: str, data: Dict) -> str:
        """Format alert message"""
        
        messages = {
            "high_cpu_usage": f"CPU usage is at {data.get('cpu_percent', 0):.1f}%",
            "high_memory_usage": f"Memory usage is at {data.get('memory_percent', 0):.1f}%",
            "disk_usage_high": f"Disk usage is at {data.get('disk_percent', 0):.1f}%",
            "model_performance_drop": f"Model recall dropped by {data.get('recall_drop', 0):.1%}",
            "high_error_rate": f"Error rate is at {data.get('error_rate', 0):.1%}",
            "high_latency": f"P95 latency is at {data.get('p95_latency_ms', 0):.0f}ms",
            "data_drift_detected": "Data drift detected, model may need retraining",
            "model_drift_detected": "Model performance drift detected, immediate action required",
            "fairness_violation": f"Fairness violations: {', '.join(data.get('fairness_violations', []))}",
        }
        
        return messages.get(rule_name, f"Alert triggered: {rule_name}")
    
    # ============================================================================
    # 🔧 Utility Methods
    # ============================================================================
    
    def get_alert_history(
        self,
        days: int = 7,
        severity: Optional[AlertSeverity] = None,
        status: Optional[AlertStatus] = None,
    ) -> List[Alert]:
        """Get alert history with filters"""
        
        cutoff = datetime.now() - timedelta(days=days)
        
        alerts = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff
        ]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str, user: str = "system"):
        """Acknowledge an alert"""
        
        for alert in self.alert_history:
            if alert.id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = user
                logger.info(f"✅ Alert acknowledged: {alert_id} by {user}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        
        for alert in self.alert_history:
            if alert.id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                logger.info(f"✅ Alert resolved: {alert_id}")
                return True
        
        return False
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        
        total = len(self.alert_history)
        by_severity = {}
        by_status = {}
        
        for alert in self.alert_history:
            severity = alert.severity.value
            status = alert.status.value
            
            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            "total_alerts": total,
            "pending_alerts": len(self._pending_alerts),
            "by_severity": by_severity,
            "by_status": by_status,
            "active_rules": len([r for r in self.alert_rules.values() if r.enabled]),
        }
    
    async def stop(self):
        """Stop the alert manager"""
        if self._worker_task:
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)
        logger.info("🛑 AlertManager stopped")


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get alert manager singleton"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager