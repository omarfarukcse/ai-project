# src/components/model_evaluation.py
"""
Comprehensive Model Evaluation with Clinical Metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    log_loss, brier_score_loss, matthews_corrcoef,
    cohen_kappa_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import json

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelMetrics:
    """Comprehensive model metrics"""
    accuracy: float = 0
    precision: float = 0
    recall: float = 0
    specificity: float = 0
    f1_score: float = 0
    roc_auc: float = 0
    pr_auc: float = 0
    log_loss: float = 0
    brier_score: float = 0
    mcc: float = 0
    kappa: float = 0
    confusion_matrix: Dict[str, int] = field(default_factory=dict)
    
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
            'confusion_matrix': self.confusion_matrix,
        }


class ModelEvaluator:
    """
    Comprehensive Model Evaluator with:
    - Multiple metrics
    - Clinical focus
    - Visualization
    - Model comparison
    - Calibration assessment
    """
    
    def __init__(self):
        self.results = {}
        
    # ============================================================================
    # 🚀 Evaluation Methods
    # ============================================================================
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model",
        probability_threshold: float = 0.5,
    ) -> ModelMetrics:
        """
        Comprehensive evaluation of a single model
        """
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_prob, probability_threshold)
        
        self.results[model_name] = metrics
        
        return metrics
    
    def evaluate_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        probability_threshold: float = 0.5,
    ) -> Dict[str, ModelMetrics]:
        """
        Evaluate multiple models
        """
        
        results = {}
        
        for model_name, model in models.items():
            metrics = self.evaluate_model(
                model, X_test, y_test, model_name, probability_threshold
            )
            results[model_name] = metrics
            
            logger.info(f"📊 {model_name}:")
            logger.info(f"   Accuracy: {metrics.accuracy:.3f}")
            logger.info(f"   Recall: {metrics.recall:.3f} ⭐")
            logger.info(f"   Specificity: {metrics.specificity:.3f}")
            logger.info(f"   F1: {metrics.f1_score:.3f}")
            logger.info(f"   ROC-AUC: {metrics.roc_auc:.3f}")
        
        return results
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> ModelMetrics:
        """
        Calculate all metrics
        """
        
        metrics = ModelMetrics()
        
        # Basic metrics
        metrics.accuracy = accuracy_score(y_true, y_pred)
        metrics.precision = precision_score(y_true, y_pred, zero_division=0)
        metrics.recall = recall_score(y_true, y_pred, zero_division=0)
        metrics.f1_score = f1_score(y_true, y_pred, zero_division=0)
        
        # Specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Confusion matrix
        metrics.confusion_matrix = {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
        }
        
        # Advanced metrics (if probabilities available)
        if y_prob is not None:
            metrics.roc_auc = roc_auc_score(y_true, y_prob)
            metrics.pr_auc = self._calculate_pr_auc(y_true, y_prob)
            metrics.log_loss = log_loss(y_true, y_prob)
            metrics.brier_score = brier_score_loss(y_true, y_prob)
        
        # Additional metrics
        metrics.mcc = matthews_corrcoef(y_true, y_pred)
        metrics.kappa = cohen_kappa_score(y_true, y_pred)
        
        return metrics
    
    def _calculate_pr_auc(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Precision-Recall AUC"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        return np.trapz(precision, recall)
    
    # ============================================================================
    # 📊 Visualization Methods
    # ============================================================================
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Healthy', 'Disease'],
            yticklabels=['Healthy', 'Disease']
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ):
        """Plot ROC curve"""
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 ROC curve saved to {save_path}")
        
        plt.close()
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ):
        """Plot calibration curve"""
        
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=model_name)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Calibration curve saved to {save_path}")
        
        plt.close()
    
    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_path: Optional[str] = None,
    ):
        """Compare multiple models with bar charts"""
        
        # Evaluate all models
        results = self.evaluate_models(models, X_test, y_test)
        
        # Prepare data
        metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score']
        data = {}
        for model_name, metrics_obj in results.items():
            data[model_name] = [getattr(metrics_obj, m, 0) for m in metrics]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.8 / len(data)
        
        for idx, (model_name, scores) in enumerate(data.items()):
            offset = (idx - len(data)/2 + 0.5) * width
            ax.bar(x + offset, scores, width, label=model_name)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Model comparison saved to {save_path}")
        
        plt.close()
        
        return results
    
    def generate_report(self, model_name: str, metrics: ModelMetrics) -> Dict:
        """Generate detailed evaluation report"""
        
        return {
            'model_name': model_name,
            'metrics': metrics.to_dict(),
            'summary': {
                'overall_performance': 'Good' if metrics.accuracy > 0.8 else 'Average' if metrics.accuracy > 0.7 else 'Poor',
                'clinical_priority': metrics.recall,
                'balanced_performance': (metrics.recall + metrics.specificity) / 2,
                'recommendations': self._generate_recommendations(metrics),
            }
        }
    
    def _generate_recommendations(self, metrics: ModelMetrics) -> List[str]:
        """Generate recommendations based on metrics"""
        
        recommendations = []
        
        if metrics.recall < 0.7:
            recommendations.append("Consider improving model recall (sensitivity)")
        if metrics.specificity < 0.7:
            recommendations.append("Consider improving model specificity")
        if metrics.roc_auc < 0.7:
            recommendations.append("Model discrimination is poor, consider feature engineering")
        if metrics.brier_score > 0.2:
            recommendations.append("Model calibration needs improvement")
        
        return recommendations