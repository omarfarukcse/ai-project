# src/utils/metrics_utils.py
"""
Advanced Metrics Calculations for Clinical ML
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    log_loss,
    brier_score_loss,
    matthews_corrcoef,
    cohen_kappa_score,
)
from sklearn.calibration import calibration_curve
from scipy import stats
from dataclasses import dataclass, field

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClinicalMetrics:
    """Clinical-focused metrics with sensitivity priority"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0  # Sensitivity - PRIMARY METRIC
    specificity: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    log_loss: float = 0.0
    brier_score: float = 0.0
    mcc: float = 0.0
    kappa: float = 0.0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'specificity': self.specificity,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'pr_auc': self.pr_auc,
            'log_loss': self.log_loss,
            'brier_score': self.brier_score,
            'mcc': self.mcc,
            'kappa': self.kappa,
            'confusion_matrix': {
                'tn': self.tn,
                'fp': self.fp,
                'fn': self.fn,
                'tp': self.tp,
            },
        }
    
    def summary(self) -> str:
        """Get summary string"""
        return (
            f"Recall: {self.recall:.3f} | "
            f"Accuracy: {self.accuracy:.3f} | "
            f"F1: {self.f1_score:.3f} | "
            f"ROC-AUC: {self.roc_auc:.3f}"
        )


@dataclass
class PerformanceMetrics:
    """Performance metrics for API monitoring"""
    endpoint: str = ""
    method: str = ""
    latency_ms: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    request_count: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'endpoint': self.endpoint,
            'method': self.method,
            'latency_ms': self.latency_ms,
            'throughput': self.throughput,
            'error_rate': self.error_rate,
            'p50_latency': self.p50_latency,
            'p95_latency': self.p95_latency,
            'p99_latency': self.p99_latency,
            'request_count': self.request_count,
            'error_count': self.error_count,
        }


class MetricsCalculator:
    """
    Advanced Metrics Calculator with:
    - Clinical metrics (sensitivity priority)
    - Performance metrics
    - Calibration metrics
    - Statistical tests
    - Model comparison
    - Visualization helpers
    """
    
    @staticmethod
    def calculate_clinical_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> ClinicalMetrics:
        """
        Calculate clinical-focused metrics
        
        Primary metric: RECALL (Sensitivity)
        - In healthcare, false negatives are most dangerous
        """
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Advanced metrics
        roc_auc = 0.0
        pr_auc = 0.0
        log_loss_val = 0.0
        brier = 0.0
        
        if y_prob is not None:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = MetricsCalculator._calculate_pr_auc(y_true, y_prob)
            log_loss_val = log_loss(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
        
        # Additional metrics
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        return ClinicalMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            specificity=specificity,
            f1_score=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            log_loss=log_loss_val,
            brier_score=brier,
            mcc=mcc,
            kappa=kappa,
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
        )
    
    @staticmethod
    def _calculate_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Precision-Recall AUC"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        return np.trapz(precision, recall)
    
    @staticmethod
    def calculate_calibration_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Calculate calibration metrics
        
        Returns:
            ECE: Expected Calibration Error
            MCE: Maximum Calibration Error
            Brier Score
        """
        
        # Brier score
        brier = brier_score_loss(y_true, y_prob)
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        # ECE (Expected Calibration Error)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_accuracies.append(y_true[mask].mean())
                bin_confidences.append(y_prob[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        bin_counts = np.array(bin_counts)
        
        # ECE (weighted by bin sizes)
        weighted_errors = np.abs(bin_accuracies - bin_confidences) * bin_counts
        ece = weighted_errors.sum() / bin_counts.sum() if bin_counts.sum() > 0 else 0
        
        # MCE (Maximum Calibration Error)
        mce = np.max(np.abs(bin_accuracies - bin_confidences)) if bin_counts.sum() > 0 else 0
        
        return {
            'brier_score': float(brier),
            'ece': float(ece),
            'mce': float(mce),
            'n_bins': n_bins,
        }
    
    @staticmethod
    def calculate_performance_metrics(
        latencies: List[float],
        request_count: int,
        error_count: int,
        endpoint: str = "",
        method: str = "",
    ) -> PerformanceMetrics:
        """Calculate API performance metrics"""
        
        if not latencies:
            return PerformanceMetrics(
                endpoint=endpoint,
                method=method,
                request_count=request_count,
                error_count=error_count,
                error_rate=error_count / max(1, request_count),
            )
        
        sorted_latencies = sorted(latencies)
        p50_idx = int(len(sorted_latencies) * 0.50)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)
        
        return PerformanceMetrics(
            endpoint=endpoint,
            method=method,
            latency_ms=sum(latencies) / len(latencies),
            throughput=request_count / (sum(latencies) / 1000) if latencies else 0,
            error_rate=error_count / max(1, request_count),
            p50_latency=sorted_latencies[p50_idx] if p50_idx < len(sorted_latencies) else 0,
            p95_latency=sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else 0,
            p99_latency=sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else 0,
            request_count=request_count,
            error_count=error_count,
        )
    
    @staticmethod
    def compare_models(
        metrics_dict: Dict[str, ClinicalMetrics]
    ) -> Dict[str, Any]:
        """Compare multiple models"""
        
        comparison = {}
        
        for model_name, metrics in metrics_dict.items():
            comparison[model_name] = {
                'recall': metrics.recall,
                'precision': metrics.precision,
                'accuracy': metrics.accuracy,
                'f1': metrics.f1_score,
                'roc_auc': metrics.roc_auc,
                'summary': metrics.summary(),
            }
        
        # Find best model by recall (clinical priority)
        if comparison:
            best_model = max(comparison.keys(), key=lambda x: comparison[x]['recall'])
            comparison['best_model'] = best_model
            comparison['best_recall'] = comparison[best_model]['recall']
        
        return comparison
    
    @staticmethod
    def calculate_statistical_tests(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Calculate statistical tests for model evaluation"""
        
        results = {}
        
        # McNemar's test (for binary classification)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Chi-square test for independence
        chi2, p_value = stats.chi2_contingency([[tn, fp], [fn, tp]])[:2]
        results['chi_square'] = {
            'statistic': float(chi2),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
        }
        
        # If probabilities available
        if y_prob is not None:
            # Hosmer-Lemeshow test (goodness of fit)
            from sklearn.calibration import calibration_curve
            
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            # Simplified version - in production use proper HL test
            results['calibration'] = {
                'ece': calculate_ece(y_true, y_prob),
            }
        
        # Confidence intervals for metrics
        metrics = MetricsCalculator.calculate_clinical_metrics(y_true, y_pred, y_prob)
        
        # Bootstrap confidence intervals (simplified)
        results['confidence_intervals'] = {
            'recall_ci': MetricsCalculator._bootstrap_ci(
                y_true, y_pred, recall_score
            ),
            'precision_ci': MetricsCalculator._bootstrap_ci(
                y_true, y_pred, precision_score
            ),
        }
        
        return results
    
    @staticmethod
    def _bootstrap_ci(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn: callable,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        
        n = len(y_true)
        scores = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            score = metric_fn(y_true[indices], y_pred[indices])
            scores.append(score)
        
        lower = np.percentile(scores, (1 - ci) / 2 * 100)
        upper = np.percentile(scores, (1 + ci) / 2 * 100)
        
        return float(lower), float(upper)


# ============================================================================
# 🔧 Convenience Functions
# ============================================================================

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy"""
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate precision"""
    return precision_score(y_true, y_pred, zero_division=0)

def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate recall (sensitivity)"""
    return recall_score(y_true, y_pred, zero_division=0)

def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate F1 score"""
    return f1_score(y_true, y_pred, zero_division=0)

def calculate_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate ROC-AUC"""
    return roc_auc_score(y_true, y_prob)

def calculate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, int]:
    """Calculate confusion matrix"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp),
    }

def calculate_calibration_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Calculate calibration metrics"""
    return MetricsCalculator.calculate_calibration_metrics(y_true, y_prob)

def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Brier score"""
    return brier_score_loss(y_true, y_prob)

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracies.append(y_true[mask].mean())
            bin_confidences.append(y_prob[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
    
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)
    
    weighted_errors = np.abs(bin_accuracies - bin_confidences) * bin_counts
    ece = weighted_errors.sum() / bin_counts.sum() if bin_counts.sum() > 0 else 0
    return float(ece)


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_metrics_calculator: Optional[MetricsCalculator] = None


def get_metrics_calculator() -> MetricsCalculator:
    """Get metrics calculator singleton"""
    global _metrics_calculator
    if _metrics_calculator is None:
        _metrics_calculator = MetricsCalculator()
    return _metrics_calculator