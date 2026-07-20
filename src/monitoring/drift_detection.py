# src/monitoring/drift_detection.py
"""
Advanced Data and Model Drift Detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


class DriftType(Enum):
    """Types of drift"""
    DATA = "data_drift"
    MODEL = "model_drift"
    CONCEPT = "concept_drift"
    FEATURE = "feature_drift"
    TARGET = "target_drift"


@dataclass
class DriftReport:
    """Drift detection report"""
    timestamp: datetime = field(default_factory=datetime.now)
    drift_type: DriftType = DriftType.DATA
    drift_detected: bool = False
    drift_score: float = 0.0
    threshold: float = 0.2
    features_drifted: List[str] = field(default_factory=list)
    feature_scores: Dict[str, float] = field(default_factory=dict)
    severity: str = "low"  # low, medium, high
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "drift_type": self.drift_type.value,
            "drift_detected": self.drift_detected,
            "drift_score": self.drift_score,
            "threshold": self.threshold,
            "features_drifted": self.features_drifted,
            "feature_scores": self.feature_scores,
            "severity": self.severity,
            "recommendations": self.recommendations,
        }


class DriftDetector:
    """
    Advanced Drift Detection with:
    - Data drift detection (statistical tests)
    - Model drift detection (performance monitoring)
    - Concept drift detection (feature-target relationship)
    - Feature drift detection (per-feature analysis)
    - Target drift detection (label distribution)
    - Automated recommendations
    """
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        threshold: float = 0.2,
        confidence_level: float = 0.95,
        detection_methods: List[str] = None,
    ):
        self.reference_data = reference_data
        self.threshold = threshold
        self.confidence_level = confidence_level
        self.detection_methods = detection_methods or ['wasserstein', 'ks', 'chi2']
        
        self._reference_stats: Dict[str, Dict] = {}
        self._drift_history: List[DriftReport] = []
        self._baseline_performance: Dict[str, float] = {}
        
        if reference_data is not None:
            self._compute_reference_stats(reference_data)
        
        logger.info("📉 DriftDetector initialized")
        logger.info(f"   Threshold: {threshold}")
        logger.info(f"   Confidence Level: {confidence_level}")
    
    def _compute_reference_stats(self, reference_data: pd.DataFrame):
        """Compute reference statistics"""
        
        for column in reference_data.columns:
            data = reference_data[column].dropna()
            
            if pd.api.types.is_numeric_dtype(data):
                self._reference_stats[column] = {
                    'type': 'numeric',
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'q1': data.quantile(0.25),
                    'median': data.median(),
                    'q3': data.quantile(0.75),
                    'skew': data.skew(),
                    'kurtosis': data.kurtosis(),
                    'distribution': self._fit_distribution(data),
                }
            elif pd.api.types.is_categorical_dtype(data) or data.dtype == 'object':
                self._reference_stats[column] = {
                    'type': 'categorical',
                    'categories': data.value_counts().to_dict(),
                    'n_categories': len(data.unique()),
                    'entropy': stats.entropy(data.value_counts().values / len(data)),
                }
            else:
                self._reference_stats[column] = {'type': 'unknown'}
        
        logger.info(f"✅ Reference stats computed for {len(self._reference_stats)} columns")
    
    def _fit_distribution(self, data: pd.Series) -> str:
        """Fit statistical distribution to data"""
        # Simple distribution fitting
        try:
            from scipy.stats import norm, expon, uniform, gamma, beta
            
            distributions = {
                'normal': norm,
                'exponential': expon,
                'uniform': uniform,
                'gamma': gamma,
                'beta': beta,
            }
            
            best_dist = 'normal'
            best_p_value = 0
            
            for name, dist in distributions.items():
                try:
                    params = dist.fit(data)
                    ks_stat, p_value = stats.kstest(
                        data,
                        dist.cdf,
                        args=params
                    )
                    if p_value > best_p_value:
                        best_p_value = p_value
                        best_dist = name
                except:
                    continue
            
            return best_dist
        except:
            return 'unknown'
    
    # ============================================================================
    # 🚀 Drift Detection Methods
    # ============================================================================
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> DriftReport:
        """
        Detect data drift between reference and current data
        """
        
        if self.reference_data is None:
            logger.warning("No reference data available")
            return DriftReport(drift_type=DriftType.DATA)
        
        threshold = threshold or self.threshold
        
        feature_scores = {}
        drifted_features = []
        drift_score = 0.0
        
        for column in current_data.columns:
            if column not in self._reference_stats:
                continue
            
            ref_stats = self._reference_stats[column]
            current_values = current_data[column].dropna()
            
            if len(current_values) == 0:
                continue
            
            # Detect drift based on column type
            if ref_stats['type'] == 'numeric':
                score = self._detect_numeric_drift(
                    current_values,
                    ref_stats,
                    self.detection_methods
                )
            elif ref_stats['type'] == 'categorical':
                score = self._detect_categorical_drift(
                    current_values,
                    ref_stats
                )
            else:
                score = 0.0
            
            feature_scores[column] = score
            
            if score > threshold:
                drifted_features.append(column)
            
            drift_score += score
        
        drift_score = drift_score / len(feature_scores) if feature_scores else 0
        
        # Determine severity
        if drift_score > threshold * 2:
            severity = "high"
        elif drift_score > threshold:
            severity = "medium"
        else:
            severity = "low"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            drifted_features,
            feature_scores,
            DriftType.DATA
        )
        
        report = DriftReport(
            drift_type=DriftType.DATA,
            drift_detected=drift_score > threshold,
            drift_score=drift_score,
            threshold=threshold,
            features_drifted=drifted_features,
            feature_scores=feature_scores,
            severity=severity,
            recommendations=recommendations,
        )
        
        self._drift_history.append(report)
        
        if report.drift_detected:
            logger.warning(
                f"⚠️ Data drift detected: score={drift_score:.3f}, "
                f"features={len(drifted_features)}"
            )
        
        return report
    
    def _detect_numeric_drift(
        self,
        current: pd.Series,
        ref_stats: Dict,
        methods: List[str]
    ) -> float:
        """Detect drift for numeric features"""
        
        scores = []
        
        # Method 1: Wasserstein distance (Earth Mover's Distance)
        if 'wasserstein' in methods:
            try:
                from scipy.stats import wasserstein_distance
                w_dist = wasserstein_distance(current, self.reference_data[current.name].dropna())
                scores.append(w_dist / (ref_stats['max'] - ref_stats['min'] + 1e-6))
            except:
                pass
        
        # Method 2: Kolmogorov-Smirnov test
        if 'ks' in methods:
            try:
                ks_stat, p_value = stats.ks_2samp(
                    current,
                    self.reference_data[current.name].dropna()
                )
                scores.append(1 - p_value if p_value < 1 else 0)
            except:
                pass
        
        # Method 3: Population stability index (PSI)
        if 'psi' in methods:
            try:
                psi = self._calculate_psi(current, self.reference_data[current.name])
                scores.append(psi)
            except:
                pass
        
        return max(scores) if scores else 0.0
    
    def _detect_categorical_drift(
        self,
        current: pd.Series,
        ref_stats: Dict
    ) -> float:
        """Detect drift for categorical features"""
        
        current_counts = current.value_counts()
        ref_counts = pd.Series(ref_stats['categories'])
        
        # Chi-square test
        try:
            # Create contingency table
            all_categories = set(current_counts.index) | set(ref_counts.index)
            observed = []
            expected = []
            
            for cat in all_categories:
                observed.append(current_counts.get(cat, 0))
                expected.append(ref_counts.get(cat, 0))
            
            # Chi-square
            chi2, p_value, _, _ = stats.chi2_contingency(
                [observed, expected]
            )
            return 1 - p_value if p_value < 1 else 0
            
        except:
            # Jensen-Shannon divergence
            try:
                js_div = jensenshannon(
                    current_counts.values / len(current),
                    ref_counts.values / len(self.reference_data)
                )
                return js_div / np.log(2)  # Normalize to 0-1
            except:
                return 0.0
    
    def _calculate_psi(self, current: pd.Series, reference: pd.Series) -> float:
        """Calculate Population Stability Index"""
        
        # Create bins
        n_bins = 10
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        ref_counts = np.histogram(reference, bins=bins)[0]
        cur_counts = np.histogram(current, bins=bins)[0]
        
        # Calculate PSI
        ref_dist = ref_counts / len(reference)
        cur_dist = cur_counts / len(current)
        
        psi = 0
        for ref_p, cur_p in zip(ref_dist, cur_dist):
            if ref_p > 0 and cur_p > 0:
                psi += (cur_p - ref_p) * np.log(cur_p / ref_p)
        
        return psi / 5  # Normalize to ~0-1
    
    def _generate_recommendations(
        self,
        drifted_features: List[str],
        feature_scores: Dict[str, float],
        drift_type: DriftType
    ) -> List[str]:
        """Generate recommendations based on drift"""
        
        recommendations = []
        
        if not drifted_features:
            recommendations.append("No drift detected. Continue monitoring.")
            return recommendations
        
        recommendations.append(f"Drift detected in {len(drifted_features)} features")
        
        # Top drifted features
        top_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        if top_features:
            recommendations.append(
                f"Top drifted features: {', '.join([f[0] for f in top_features])}"
            )
        
        # Specific recommendations
        if drift_type == DriftType.DATA:
            recommendations.append("Consider retraining model with new data")
            recommendations.append("Review data pipeline for changes")
            recommendations.append("Validate data quality and consistency")
        
        elif drift_type == DriftType.MODEL:
            recommendations.append("Model performance may be degrading")
            recommendations.append("Consider model retraining or recalibration")
            recommendations.append("Evaluate model on recent data")
        
        elif drift_type == DriftType.CONCEPT:
            recommendations.append("Feature-target relationship has changed")
            recommendations.append("Consider feature engineering review")
            recommendations.append("Explore alternative model architectures")
        
        return recommendations
    
    # ============================================================================
    # 🔧 Model Drift Detection
    # ============================================================================
    
    def detect_model_drift(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> DriftReport:
        """Detect model performance drift"""
        
        baseline = baseline_metrics or self._baseline_performance
        
        if not baseline:
            # Store as baseline
            self._baseline_performance = current_metrics.copy()
            return DriftReport(
                drift_type=DriftType.MODEL,
                drift_detected=False,
                severity="low",
                recommendations=["Baseline model performance recorded"]
            )
        
        drift_score = 0.0
        drifted_metrics = []
        
        for metric, current_value in current_metrics.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                
                # Calculate relative change
                if baseline_value != 0:
                    change = abs(current_value - baseline_value) / baseline_value
                else:
                    change = abs(current_value - baseline_value)
                
                if change > self.threshold:
                    drifted_metrics.append(metric)
                
                drift_score += change
        
        drift_score = drift_score / len(current_metrics) if current_metrics else 0
        
        # Determine severity
        if drift_score > self.threshold * 2:
            severity = "high"
        elif drift_score > self.threshold:
            severity = "medium"
        else:
            severity = "low"
        
        recommendations = self._generate_recommendations(
            drifted_metrics,
            {},
            DriftType.MODEL
        )
        
        report = DriftReport(
            drift_type=DriftType.MODEL,
            drift_detected=drift_score > self.threshold,
            drift_score=drift_score,
            threshold=self.threshold,
            features_drifted=drifted_metrics,
            severity=severity,
            recommendations=recommendations,
        )
        
        self._drift_history.append(report)
        
        if report.drift_detected:
            logger.warning(f"⚠️ Model drift detected: score={drift_score:.3f}")
        
        return report
    
    # ============================================================================
    # 🔧 Utility Methods
    # ============================================================================
    
    def update_reference(self, new_data: pd.DataFrame):
        """Update reference data"""
        self.reference_data = new_data
        self._compute_reference_stats(new_data)
        logger.info("✅ Reference data updated")
    
    def get_drift_history(self, days: int = 30) -> List[DriftReport]:
        """Get drift detection history"""
        cutoff = datetime.now() - timedelta(days=days)
        return [
            report for report in self._drift_history
            if report.timestamp >= cutoff
        ]
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get drift detection summary"""
        
        recent_reports = self.get_drift_history(days=7)
        
        if not recent_reports:
            return {"status": "no_data", "message": "No drift reports available"}
        
        drift_count = sum(1 for r in recent_reports if r.drift_detected)
        
        return {
            "total_reports": len(recent_reports),
            "drift_detected_count": drift_count,
            "drift_detected_ratio": drift_count / len(recent_reports),
            "latest_drift_score": recent_reports[-1].drift_score if recent_reports else 0,
            "latest_severity": recent_reports[-1].severity if recent_reports else "unknown",
            "most_drifted_features": self._get_most_drifted_features(recent_reports),
        }
    
    def _get_most_drifted_features(self, reports: List[DriftReport]) -> List[str]:
        """Get most frequently drifted features"""
        feature_counts = {}
        
        for report in reports:
            for feature in report.features_drifted:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        sorted_features = sorted(
            feature_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [f[0] for f in sorted_features[:10]]


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_drift_detector: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    """Get drift detector singleton"""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector(
            threshold=config_manager.get("monitoring.drift_threshold", 0.2),
            confidence_level=config_manager.get("monitoring.confidence_level", 0.95),
        )
    return _drift_detector