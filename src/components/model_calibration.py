# src/components/model_calibration.py
"""
Model Calibration with Platt Scaling and Isotonic Regression
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from dataclasses import dataclass, field

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationMetrics:
    """Calibration performance metrics"""
    brier_score: float = 0
    log_loss: float = 0
    ece: float = 0  # Expected Calibration Error
    mce: float = 0  # Maximum Calibration Error
    reliability_diagram: Dict = field(default_factory=dict)


class ModelCalibrator:
    """
    Advanced Model Calibration with:
    - Platt Scaling (sigmoid)
    - Isotonic Regression
    - Temperature Scaling
    - Calibration visualization
    - Cross-validation calibration
    """
    
    def __init__(
        self,
        method: str = 'platt',  # platt, isotonic, temperature
        cv_folds: int = 5,
        ensemble: bool = True,
        save_calibrated: bool = True,
    ):
        self.method = method
        self.cv_folds = cv_folds
        self.ensemble = ensemble
        self.save_calibrated = save_calibrated
        
        self.calibrated_model = None
        self.calibration_metrics = None
        self._fitted = False
        
        logger.info(f"🎯 ModelCalibrator initialized: {method}")
    
    # ============================================================================
    # 🚀 Calibration Methods
    # ============================================================================
    
    def calibrate(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Any:
        """
        Calibrate model probabilities
        
        Args:
            model: Base model to calibrate
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Calibrated model
        """
        
        logger.info(f"🔄 Calibrating model with {self.method}...")
        
        if self.method == 'platt':
            self.calibrated_model = self._platt_scaling(model, X_train, y_train)
        elif self.method == 'isotonic':
            self.calibrated_model = self._isotonic_regression(model, X_train, y_train)
        elif self.method == 'temperature':
            self.calibrated_model = self._temperature_scaling(model, X_train, y_train)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self._fitted = True
        
        # Evaluate calibration
        if X_val is not None and y_val is not None:
            self.calibration_metrics = self.evaluate_calibration(
                self.calibrated_model, X_val, y_val
            )
            logger.info(f"✅ Calibration complete:")
            logger.info(f"   Brier Score: {self.calibration_metrics.brier_score:.4f}")
            logger.info(f"   ECE: {self.calibration_metrics.ece:.4f}")
        else:
            # Use training data for metrics
            y_pred_proba = self.calibrated_model.predict_proba(X_train)[:, 1]
            self.calibration_metrics = CalibrationMetrics(
                brier_score=brier_score_loss(y_train, y_pred_proba),
                log_loss=log_loss(y_train, y_pred_proba),
            )
        
        return self.calibrated_model
    
    def _platt_scaling(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
        """Apply Platt Scaling (sigmoid calibration)"""
        
        if self.ensemble:
            # Use CalibratedClassifierCV for ensemble calibration
            calibrated = CalibratedClassifierCV(
                model,
                method='sigmoid',
                cv=self.cv_folds,
                ensemble=True,
            )
            calibrated.fit(X, y)
            return calibrated
        else:
            # Manual Platt scaling
            # Get cross-validated predictions
            y_pred_proba = cross_val_predict(
                model, X, y,
                cv=self.cv_folds,
                method='predict_proba'
            )[:, 1]
            
            # Fit sigmoid to predictions
            sigmoid = LogisticRegression(C=1e10)
            sigmoid.fit(y_pred_proba.reshape(-1, 1), y)
            
            # Create wrapper model
            class PlattWrapper:
                def __init__(self, base_model, sigmoid):
                    self.base_model = base_model
                    self.sigmoid = sigmoid
                    self._fitted = False
                
                def fit(self, X, y):
                    self.base_model.fit(X, y)
                    return self
                
                def predict_proba(self, X):
                    base_proba = self.base_model.predict_proba(X)[:, 1]
                    calibrated_proba = self.sigmoid.predict_proba(
                        base_proba.reshape(-1, 1)
                    )[:, 1]
                    return np.column_stack([1 - calibrated_proba, calibrated_proba])
                
                def predict(self, X):
                    return self.predict_proba(X)[:, 1] > 0.5
            
            wrapper = PlattWrapper(model, sigmoid)
            wrapper.fit(X, y)
            return wrapper
    
    def _isotonic_regression(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
        """Apply Isotonic Regression calibration"""
        
        # Get cross-validated predictions
        y_pred_proba = cross_val_predict(
            model, X, y,
            cv=self.cv_folds,
            method='predict_proba'
        )[:, 1]
        
        # Fit isotonic regression
        isotonic = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds='clip'
        )
        isotonic.fit(y_pred_proba, y)
        
        # Create wrapper model
        class IsotonicWrapper:
            def __init__(self, base_model, isotonic):
                self.base_model = base_model
                self.isotonic = isotonic
                self._fitted = False
            
            def fit(self, X, y):
                self.base_model.fit(X, y)
                return self
            
            def predict_proba(self, X):
                base_proba = self.base_model.predict_proba(X)[:, 1]
                calibrated_proba = self.isotonic.transform(base_proba)
                return np.column_stack([1 - calibrated_proba, calibrated_proba])
            
            def predict(self, X):
                return self.predict_proba(X)[:, 1] > 0.5
        
        wrapper = IsotonicWrapper(model, isotonic)
        wrapper.fit(X, y)
        return wrapper
    
    def _temperature_scaling(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
        """Apply Temperature Scaling"""
        
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Get predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred_logits = np.log(y_pred_proba / (1 - y_pred_proba + 1e-6))
        
        # Convert to torch tensors
        logits = torch.tensor(y_pred_logits, dtype=torch.float32).reshape(-1, 1)
        labels = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)
        
        # Learn temperature
        class TemperatureScaler(nn.Module):
            def __init__(self):
                super().__init__()
                self.temperature = nn.Parameter(torch.ones(1))
            
            def forward(self, logits):
                return torch.sigmoid(logits / self.temperature)
        
        scaler = TemperatureScaler()
        optimizer = optim.Adam(scaler.parameters(), lr=0.01)
        
        for epoch in range(100):
            optimizer.zero_grad()
            probs = scaler(logits)
            loss = nn.BCELoss()(probs, labels)
            loss.backward()
            optimizer.step()
        
        temperature = scaler.temperature.item()
        
        # Create wrapper model
        class TemperatureWrapper:
            def __init__(self, base_model, temperature):
                self.base_model = base_model
                self.temperature = temperature
            
            def predict_proba(self, X):
                base_proba = self.base_model.predict_proba(X)[:, 1]
                logits = np.log(base_proba / (1 - base_proba + 1e-6))
                calibrated_proba = 1 / (1 + np.exp(-logits / self.temperature))
                return np.column_stack([1 - calibrated_proba, calibrated_proba])
            
            def predict(self, X):
                return self.predict_proba(X)[:, 1] > 0.5
        
        return TemperatureWrapper(model, temperature)
    
    # ============================================================================
    # 📊 Evaluation Methods
    # ============================================================================
    
    def evaluate_calibration(
        self,
        calibrated_model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """
        Evaluate calibration quality
        
        Returns:
            CalibrationMetrics with Brier score, Log loss, ECE, MCE
        """
        
        # Get predictions
        y_pred_proba = calibrated_model.predict_proba(X)[:, 1]
        
        # Brier score
        brier = brier_score_loss(y, y_pred_proba)
        log_loss_val = log_loss(y, y_pred_proba)
        
        # Expected Calibration Error
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred_proba, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_accuracies.append(y[mask].mean())
                bin_confidences.append(y_pred_proba[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        bin_counts = np.array(bin_counts)
        
        # ECE and MCE
        weighted_errors = np.abs(bin_accuracies - bin_confidences) * bin_counts
        ece = weighted_errors.sum() / bin_counts.sum() if bin_counts.sum() > 0 else 0
        mce = np.max(np.abs(bin_accuracies - bin_confidences)) if bin_counts.sum() > 0 else 0
        
        self.calibration_metrics = CalibrationMetrics(
            brier_score=float(brier),
            log_loss=float(log_loss_val),
            ece=float(ece),
            mce=float(mce),
            reliability_diagram={
                'bin_edges': bin_edges.tolist(),
                'bin_accuracies': bin_accuracies.tolist(),
                'bin_confidences': bin_confidences.tolist(),
                'bin_counts': bin_counts.tolist(),
            }
        )
        
        return self.calibration_metrics
    
    # ============================================================================
    # 📊 Visualization Methods
    # ============================================================================
    
    def plot_calibration_curve(
        self,
        calibrated_model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        save_path: Optional[str] = None,
    ):
        """Plot calibration curve with reliability diagram"""
        
        y_pred_proba = calibrated_model.predict_proba(X)[:, 1]
        
        from sklearn.calibration import calibration_curve
        
        prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calibration curve
        axes[0].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Calibrated')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect')
        axes[0].set_xlabel('Mean Predicted Probability')
        axes[0].set_ylabel('Fraction of Positives')
        axes[0].set_title('Calibration Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Reliability histogram
        bin_edges = np.linspace(0, 1, 11)
        hist, _ = np.histogram(y_pred_proba, bins=bin_edges)
        axes[1].bar(bin_edges[:-1], hist, width=0.09, alpha=0.7, color='steelblue')
        axes[1].set_xlabel('Predicted Probability')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Prediction Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Calibration curve saved to {save_path}")
        
        plt.close()
    
    def get_metrics(self) -> Dict[str, float]:
        """Get calibration metrics"""
        if self.calibration_metrics:
            return {
                'brier_score': self.calibration_metrics.brier_score,
                'log_loss': self.calibration_metrics.log_loss,
                'ece': self.calibration_metrics.ece,
                'mce': self.calibration_metrics.mce,
            }
        return {}
    
    def save(self, path: str = "models/calibrator.pkl"):
        """Save calibrator"""
        joblib.dump(self, path)
        logger.info(f"✅ Calibrator saved to {path}")
    
    @classmethod
    def load(cls, path: str = "models/calibrator.pkl"):
        """Load calibrator"""
        return joblib.load(path)