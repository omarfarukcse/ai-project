# src/monitoring/comprehensive_drift.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy import stats
import json
from datetime import datetime

from src.logger import get_logger
from src.monitoring.drift_detection import DriftDetector

logger = get_logger(__name__)

class ComprehensiveDriftDetector:
    """
    Comprehensive drift detection with multiple metrics:
    - Feature Drift (PSI, KL Divergence, JS Distance)
    - Data Drift (K-S Test)
    - Concept Drift
    - Prediction Drift
    """
    
    def __init__(self):
        self.drift_metrics = {}
        self.drift_history = []
        self.thresholds = {
            'psi': 0.2,        # Population Stability Index
            'kl_div': 0.1,     # KL Divergence
            'js_distance': 0.1, # Jensen-Shannon Distance
            'ks_test': 0.1,    # Kolmogorov-Smirnov Test
            'concept': 0.15    # Concept Drift
        }
    
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray,
                     bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        # Create bins from expected distribution
        bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
        
        # Count expected and actual in each bin
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)
        
        # Add small epsilon to avoid division by zero
        expected_counts = expected_counts + 1e-6
        actual_counts = actual_counts + 1e-6
        
        # Calculate proportions
        expected_prop = expected_counts / len(expected)
        actual_prop = actual_counts / len(actual)
        
        # Calculate PSI
        psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
        
        return float(psi)
    
    def calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL Divergence between two distributions"""
        # Compute histograms
        bins = 50
        p_hist, bin_edges = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
        
        # Add small epsilon
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10
        
        # Calculate KL Divergence
        kl_div = np.sum(p_hist * np.log(p_hist / q_hist))
        
        return float(kl_div)
    
    def calculate_js_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Jensen-Shannon Distance"""
        # Calculate KL Divergence
        kl_p = self.calculate_kl_divergence(p, (p + q) / 2)
        kl_q = self.calculate_kl_divergence(q, (p + q) / 2)
        
        js_div = 0.5 * kl_p + 0.5 * kl_q
        js_distance = np.sqrt(js_div)
        
        return float(js_distance)
    
    def detect_drift(self, reference_data: pd.DataFrame,
                    current_data: pd.DataFrame,
                    predictions: Dict = None) -> Dict[str, Any]:
        """
        Detect all types of drift
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'feature_drift': {},
            'data_drift': {},
            'prediction_drift': {},
            'concept_drift': {},
            'overall_risk': 'LOW'
        }
        
        # 1. Feature Drift
        for col in reference_data.columns:
            if col in current_data.columns:
                expected = reference_data[col].dropna().values
                actual = current_data[col].dropna().values
                
                if len(expected) > 0 and len(actual) > 0:
                    # Calculate multiple metrics
                    psi = self.calculate_psi(expected, actual)
                    kl_div = self.calculate_kl_divergence(expected, actual)
                    js_dist = self.calculate_js_distance(expected, actual)
                    
                    # KS Test
                    ks_stat, ks_pvalue = stats.ks_2samp(expected, actual)
                    
                    results['feature_drift'][col] = {
                        'psi': psi,
                        'kl_divergence': kl_div,
                        'js_distance': js_dist,
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pvalue,
                        'drift_detected': (
                            psi > self.thresholds['psi'] or
                            kl_div > self.thresholds['kl_div'] or
                            js_dist > self.thresholds['js_distance'] or
                            ks_stat > self.thresholds['ks_test']
                        )
                    }
        
        # 2. Concept Drift (if predictions available)
        if predictions and 'y_true' in predictions and 'y_pred' in predictions:
            concept_drift = self.detect_concept_drift(
                predictions['y_true'],
                predictions['y_pred']
            )
            results['concept_drift'] = concept_drift
        
        # 3. Overall Risk Assessment
        drift_count = sum(
            1 for metrics in results['feature_drift'].values()
            if metrics.get('drift_detected', False)
        )
        
        total_features = len(results['feature_drift'])
        if total_features > 0:
            drift_ratio = drift_count / total_features
            
            if drift_ratio > 0.3:
                results['overall_risk'] = 'HIGH'
            elif drift_ratio > 0.15:
                results['overall_risk'] = 'MEDIUM'
            else:
                results['overall_risk'] = 'LOW'
        
        # Store history
        self.drift_history.append(results)
        self._save_drift_history()
        
        return results
    
    def detect_concept_drift(self, y_true: np.ndarray, y_pred: np.ndarray,
                           window_size: int = 100) -> Dict[str, Any]:
        """
        Detect concept drift using sliding window
        """
        if len(y_true) < window_size * 2:
            return {'drift_detected': False, 'message': 'Insufficient data'}
        
        # Calculate accuracy in windows
        window_accuracies = []
        
        for i in range(0, len(y_true) - window_size, window_size // 2):
            window_true = y_true[i:i+window_size]
            window_pred = y_pred[i:i+window_size]
            accuracy = np.mean(window_true == window_pred)
            window_accuracies.append(accuracy)
        
        # Check for significant drop
        if len(window_accuracies) > 1:
            recent_avg = np.mean(window_accuracies[-3:])
            historical_avg = np.mean(window_accuracies[:-3])
            drop = historical_avg - recent_avg
            
            return {
                'drift_detected': drop > self.thresholds['concept'],
                'drop': float(drop),
                'recent_accuracy': float(recent_avg),
                'historical_accuracy': float(historical_avg),
                'window_size': window_size
            }
        
        return {'drift_detected': False, 'message': 'Not enough windows'}
    
    def _save_drift_history(self):
        """Save drift history"""
        history_file = "outputs/drift_history.json"
        Path("outputs").mkdir(exist_ok=True)
        
        if Path(history_file).exists():
            with open(history_file, 'r') as f:
                existing = json.load(f)
            self.drift_history = existing + self.drift_history[-100:]
        
        with open(history_file, 'w') as f:
            json.dump(self.drift_history[-1000:], f, indent=2)
    
    def get_drift_report(self) -> Dict[str, Any]:
        """Generate comprehensive drift report"""
        if not self.drift_history:
            return {'message': 'No drift data available'}
        
        recent = self.drift_history[-10:]
        
        return {
            'total_checks': len(self.drift_history),
            'recent_checks': len(recent),
            'recent_drift_count': sum(
                1 for r in recent
                if r.get('overall_risk') == 'HIGH'
            ),
            'overall_risk': self.drift_history[-1].get('overall_risk', 'UNKNOWN'),
            'drifting_features': [
                col for col, metrics in self.drift_history[-1].get('feature_drift', {}).items()
                if metrics.get('drift_detected', False)
            ],
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on drift"""
        recommendations = []
        
        if not self.drift_history:
            return ['No drift data available']
        
        last = self.drift_history[-1]
        
        if last.get('overall_risk') == 'HIGH':
            recommendations.append("🚨 High drift detected - immediate action required")
            recommendations.append("Consider retraining model with recent data")
        
        # Check specific features
        for col, metrics in last.get('feature_drift', {}).items():
            if metrics.get('drift_detected', False):
                if metrics.get('psi', 0) > 0.3:
                    recommendations.append(f"⚠️ High PSI for {col} - distribution shift detected")
                elif metrics.get('ks_statistic', 0) > 0.2:
                    recommendations.append(f"⚠️ Significant distribution change in {col}")
        
        # Concept drift
        concept = last.get('concept_drift', {})
        if concept.get('drift_detected', False):
            recommendations.append(
                f"⚠️ Concept drift detected - accuracy dropped by {concept.get('drop', 0):.2%}"
            )
        
        if not recommendations:
            recommendations.append("✅ No significant drift detected")
        
        return recommendations