# src/monitoring/bias_monitor.py
"""
Fairness and Bias Monitoring with Demographic Auditing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from sklearn.metrics import confusion_matrix

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


class ProtectedAttribute(Enum):
    """Protected attributes for bias monitoring"""
    AGE = "age"
    GENDER = "gender"
    RACE = "race"
    ETHNICITY = "ethnicity"
    SOCIOECONOMIC = "socioeconomic_status"
    DISABILITY = "disability"
    PREGNANCY = "pregnancy"
    RELIGION = "religion"
    SEXUAL_ORIENTATION = "sexual_orientation"


@dataclass
class FairnessMetrics:
    """Fairness metrics for bias analysis"""
    group: str = ""
    tp_rate: float = 0.0
    fp_rate: float = 0.0
    fn_rate: float = 0.0
    tn_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "group": self.group,
            "tp_rate": self.tp_rate,
            "fp_rate": self.fp_rate,
            "fn_rate": self.fn_rate,
            "tn_rate": self.tn_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "count": self.count,
        }


@dataclass
class BiasReport:
    """Comprehensive bias report"""
    timestamp: datetime = field(default_factory=datetime.now)
    protected_attribute: str = ""
    groups: List[FairnessMetrics] = field(default_factory=list)
    disparity_metrics: Dict[str, float] = field(default_factory=dict)
    fairness_violations: List[str] = field(default_factory=list)
    severity: str = "low"
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "protected_attribute": self.protected_attribute,
            "groups": [g.to_dict() for g in self.groups],
            "disparity_metrics": self.disparity_metrics,
            "fairness_violations": self.fairness_violations,
            "severity": self.severity,
            "recommendations": self.recommendations,
        }


class BiasMonitor:
    """
    Advanced Bias Monitoring with:
    - Demographic parity analysis
    - Equal opportunity analysis
    - Disparate impact calculation
    - Group fairness metrics
    - Fairness violation detection
    - Automated recommendations
    """
    
    def __init__(
        self,
        protected_attributes: List[ProtectedAttribute] = None,
        fairness_threshold: float = 0.8,
        enable_demographic_parity: bool = True,
        enable_equal_opportunity: bool = True,
    ):
        self.protected_attributes = protected_attributes or [
            ProtectedAttribute.AGE,
            ProtectedAttribute.GENDER,
        ]
        self.fairness_threshold = fairness_threshold
        self.enable_demographic_parity = enable_demographic_parity
        self.enable_equal_opportunity = enable_equal_opportunity
        
        self._bias_history: List[BiasReport] = []
        self._group_performance: Dict[str, Dict] = {}
        
        logger.info("⚖️ BiasMonitor initialized")
        logger.info(f"   Protected Attributes: {[p.value for p in self.protected_attributes]}")
        logger.info(f"   Fairness Threshold: {fairness_threshold}")
    
    # ============================================================================
    # 🚀 Bias Detection Methods
    # ============================================================================
    
    def audit_bias(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        protected_groups: np.ndarray,
        attribute_name: str = "protected_attribute",
    ) -> BiasReport:
        """
        Audit bias for a protected attribute
        
        Args:
            predictions: Model predictions
            targets: True labels
            protected_groups: Protected group labels
            attribute_name: Name of protected attribute
        """
        
        logger.info(f"🔍 Auditing bias for {attribute_name}")
        
        # Get unique groups
        unique_groups = np.unique(protected_groups)
        group_metrics = []
        
        # Calculate metrics for each group
        for group in unique_groups:
            mask = protected_groups == group
            group_preds = predictions[mask]
            group_targets = targets[mask]
            
            if len(group_preds) == 0:
                continue
            
            # Calculate metrics
            metrics = self._calculate_group_metrics(
                group_targets,
                group_preds,
                str(group)
            )
            group_metrics.append(metrics)
        
        # Calculate disparity metrics
        disparity_metrics = self._calculate_disparity_metrics(group_metrics)
        
        # Check for fairness violations
        fairness_violations = self._check_fairness_violations(
            group_metrics,
            disparity_metrics
        )
        
        # Determine severity
        severity = self._determine_severity(
            disparity_metrics,
            fairness_violations
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            group_metrics,
            disparity_metrics,
            fairness_violations
        )
        
        report = BiasReport(
            protected_attribute=attribute_name,
            groups=group_metrics,
            disparity_metrics=disparity_metrics,
            fairness_violations=fairness_violations,
            severity=severity,
            recommendations=recommendations,
        )
        
        self._bias_history.append(report)
        
        if fairness_violations:
            logger.warning(
                f"⚠️ Fairness violations detected: {len(fairness_violations)}"
            )
        
        return report
    
    def _calculate_group_metrics(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        group_name: str
    ) -> FairnessMetrics:
        """Calculate metrics for a group"""
        
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        
        return FairnessMetrics(
            group=group_name,
            tp_rate=tp / (tp + fn) if (tp + fn) > 0 else 0,
            fp_rate=fp / (fp + tn) if (fp + tn) > 0 else 0,
            fn_rate=fn / (tp + fn) if (tp + fn) > 0 else 0,
            tn_rate=tn / (fp + tn) if (fp + tn) > 0 else 0,
            precision=tp / (tp + fp) if (tp + fp) > 0 else 0,
            recall=tp / (tp + fn) if (tp + fn) > 0 else 0,
            f1=2 * (tp / (tp + fp)) * (tp / (tp + fn)) / (
                (tp / (tp + fp)) + (tp / (tp + fn))
            ) if (tp + fp) > 0 and (tp + fn) > 0 else 0,
            accuracy=(tp + tn) / (tp + tn + fp + fn),
            count=int(tp + tn + fp + fn),
        )
    
    def _calculate_disparity_metrics(
        self,
        group_metrics: List[FairnessMetrics]
    ) -> Dict[str, float]:
        """Calculate disparity metrics between groups"""
        
        if len(group_metrics) < 2:
            return {}
        
        # Find majority group (largest count)
        majority = max(group_metrics, key=lambda x: x.count)
        
        disparity_metrics = {}
        
        # Compare each group to majority
        for group in group_metrics:
            if group.group == majority.group:
                continue
            
            key = f"{group.group}_vs_{majority.group}"
            
            # Demographic parity (selection rate ratio)
            selection_rate_g = group.tp_rate + group.fp_rate
            selection_rate_m = majority.tp_rate + majority.fp_rate
            
            if selection_rate_m > 0:
                demographic_parity = selection_rate_g / selection_rate_m
            else:
                demographic_parity = 1.0
            
            # Equal opportunity (true positive rate ratio)
            if majority.tp_rate > 0:
                equal_opportunity = group.tp_rate / majority.tp_rate
            else:
                equal_opportunity = 1.0
            
            # Disparate impact (selection rate ratio)
            disparate_impact = demographic_parity
            
            disparity_metrics[f"{key}_demographic_parity"] = demographic_parity
            disparity_metrics[f"{key}_equal_opportunity"] = equal_opportunity
            disparity_metrics[f"{key}_disparate_impact"] = disparate_impact
        
        return disparity_metrics
    
    def _check_fairness_violations(
        self,
        group_metrics: List[FairnessMetrics],
        disparity_metrics: Dict[str, float]
    ) -> List[str]:
        """Check for fairness violations"""
        
        violations = []
        
        # Check demographic parity
        if self.enable_demographic_parity:
            for key, value in disparity_metrics.items():
                if 'demographic_parity' in key:
                    if value < self.fairness_threshold:
                        violations.append(
                            f"Demographic parity violation: {key} = {value:.3f}"
                        )
        
        # Check equal opportunity
        if self.enable_equal_opportunity:
            for key, value in disparity_metrics.items():
                if 'equal_opportunity' in key:
                    if value < self.fairness_threshold:
                        violations.append(
                            f"Equal opportunity violation: {key} = {value:.3f}"
                        )
        
        # Check disparate impact (4/5 rule)
        for key, value in disparity_metrics.items():
            if 'disparate_impact' in key:
                if value < 0.8:  # 4/5 rule
                    violations.append(
                        f"Disparate impact violation (4/5 rule): {key} = {value:.3f}"
                    )
        
        # Check for group size bias
        for group in group_metrics:
            if group.count < len(group_metrics) * 0.05:  # Less than 5% of total
                violations.append(f"Small group size: {group.group} ({group.count})")
        
        return violations
    
    def _determine_severity(
        self,
        disparity_metrics: Dict[str, float],
        violations: List[str]
    ) -> str:
        """Determine severity of fairness issues"""
        
        if not violations:
            return "low"
        
        if len(violations) > 3:
            return "critical"
        
        # Check for severe violations
        severe_violations = [
            v for v in violations
            if '0.8' in v or '0.6' in v
        ]
        
        if severe_violations:
            return "high"
        
        if len(violations) > 1:
            return "medium"
        
        return "low"
    
    def _generate_recommendations(
        self,
        group_metrics: List[FairnessMetrics],
        disparity_metrics: Dict[str, float],
        violations: List[str]
    ) -> List[str]:
        """Generate recommendations for fairness issues"""
        
        recommendations = []
        
        if not violations:
            recommendations.append("No fairness violations detected")
            return recommendations
        
        # Group-specific recommendations
        for group in group_metrics:
            if group.recall < 0.7:
                recommendations.append(
                    f"Improve recall for group {group.group} (current: {group.recall:.3f})"
                )
            if group.precision < 0.7:
                recommendations.append(
                    f"Improve precision for group {group.group} (current: {group.precision:.3f})"
                )
        
        # Bias mitigation strategies
        if any('demographic_parity' in k for k in disparity_metrics):
            recommendations.append(
                "Consider reweighting or resampling to improve demographic parity"
            )
        
        if any('equal_opportunity' in k for k in disparity_metrics):
            recommendations.append(
                "Consider calibrating thresholds per group for equal opportunity"
            )
        
        if any('disparate_impact' in k for k in disparity_metrics):
            recommendations.append(
                "Review feature engineering and model architecture for bias"
            )
        
        # General recommendations
        recommendations.extend([
            "Collect more diverse training data for underrepresented groups",
            "Regularly audit model performance across demographic groups",
            "Implement fairness constraints in model training",
            "Document and monitor fairness metrics over time",
        ])
        
        return recommendations
    
    # ============================================================================
    # 🔧 Utility Methods
    # ============================================================================
    
    def get_bias_history(self, days: int = 30) -> List[BiasReport]:
        """Get bias audit history"""
        cutoff = datetime.now() - timedelta(days=days)
        return [
            report for report in self._bias_history
            if report.timestamp >= cutoff
        ]
    
    def get_bias_summary(self) -> Dict[str, Any]:
        """Get bias monitoring summary"""
        
        recent_reports = self.get_bias_history(days=7)
        
        if not recent_reports:
            return {"status": "no_data", "message": "No bias reports available"}
        
        violations = []
        for report in recent_reports:
            violations.extend(report.fairness_violations)
        
        return {
            "total_reports": len(recent_reports),
            "violations_detected": len(violations),
            "unique_violations": len(set(violations)),
            "latest_severity": recent_reports[-1].severity if recent_reports else "unknown",
            "attributes_audited": list(set(r.protected_attribute for r in recent_reports)),
        }
    
    def get_group_performance(
        self,
        protected_attribute: str
    ) -> Dict[str, FairnessMetrics]:
        """Get performance metrics by group"""
        
        # Find latest report for this attribute
        reports = [
            r for r in self._bias_history
            if r.protected_attribute == protected_attribute
        ]
        
        if not reports:
            return {}
        
        latest = reports[-1]
        return {g.group: g for g in latest.groups}
    
    def check_fairness_thresholds(
        self,
        protected_attribute: str,
        threshold: Optional[float] = None
    ) -> Dict[str, bool]:
        """Check if all groups meet fairness thresholds"""
        
        threshold = threshold or self.fairness_threshold
        
        group_performance = self.get_group_performance(protected_attribute)
        
        if not group_performance:
            return {}
        
        results = {}
        for group, metrics in group_performance.items():
            results[group] = {
                "recall_ok": metrics.recall >= threshold,
                "precision_ok": metrics.precision >= threshold,
                "f1_ok": metrics.f1 >= threshold,
                "all_ok": (
                    metrics.recall >= threshold and
                    metrics.precision >= threshold and
                    metrics.f1 >= threshold
                ),
            }
        
        return results
    
    def export_report(self, report: BiasReport, path: str = "outputs/reports/bias_report.json"):
        """Export bias report to file"""
        import json
        from pathlib import Path
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"📄 Bias report exported to {path}")


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_bias_monitor: Optional[BiasMonitor] = None


def get_bias_monitor() -> BiasMonitor:
    """Get bias monitor singleton"""
    global _bias_monitor
    if _bias_monitor is None:
        _bias_monitor = BiasMonitor(
            fairness_threshold=config_manager.get("monitoring.fairness_threshold", 0.8),
        )
    return _bias_monitor