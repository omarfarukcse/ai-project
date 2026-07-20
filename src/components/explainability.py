# src/components/explainability.py
"""
Advanced SHAP Explainability with Clinical Interpretation
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple, Union
import json
from pathlib import Path
import logging

from src.logger import get_logger

logger = get_logger(__name__)


class ClinicalSHAPExplainer:
    """
    Clinical SHAP Explainability with:
    - Global explanations
    - Local explanations
    - Clinical interpretation
    - Visualization generation
    - Batch explanation caching
    """
    
    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        model_type: Optional[str] = None,
        cache_explanations: bool = True,
    ):
        self.model = model
        self.X_train = X_train
        self.model_type = model_type or self._detect_model_type()
        self.cache_explanations = cache_explanations
        
        self.explainer = None
        self.shap_values = None
        self.base_value = None
        self.feature_names = X_train.columns.tolist()
        self._cache = {}
        
        self._initialize_explainer()
        
        logger.info(f"🧠 ClinicalSHAPExplainer initialized")
        logger.info(f"   Model type: {self.model_type}")
        logger.info(f"   Features: {len(self.feature_names)}")
    
    def _detect_model_type(self) -> str:
        """Detect model type for appropriate explainer"""
        model_class = self.model.__class__.__name__
        
        if any(x in model_class for x in ['XGB', 'RandomForest', 'GradientBoosting']):
            return 'tree'
        elif 'LogisticRegression' in model_class or 'Linear' in model_class:
            return 'linear'
        elif 'LGBM' in model_class:
            return 'tree'
        elif 'CatBoost' in model_class:
            return 'tree'
        else:
            return 'kernel'
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer"""
        
        logger.info(f"🔄 Initializing SHAP explainer ({self.model_type})...")
        
        try:
            if self.model_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(self.X_train[:100])
                self.base_value = self.explainer.expected_value
                
            elif self.model_type == 'linear':
                self.explainer = shap.LinearExplainer(
                    self.model,
                    self.X_train[:100],
                    feature_dependence='independent'
                )
                self.shap_values = self.explainer.shap_values(self.X_train[:100])
                self.base_value = self.explainer.expected_value
                
            else:
                # KernelExplainer for black-box models
                def predict_proba(X):
                    return self.model.predict_proba(X)[:, 1]
                
                background = shap.sample(self.X_train, min(50, len(self.X_train)))
                self.explainer = shap.KernelExplainer(
                    predict_proba,
                    background
                )
                self.shap_values = self.explainer.shap_values(self.X_train[:50])
                self.base_value = self.explainer.expected_value
            
            logger.info("✅ SHAP explainer initialized")
            
        except Exception as e:
            logger.error(f"❌ SHAP explainer initialization failed: {str(e)}")
            raise
    
    # ============================================================================
    # 🚀 Explanation Generation
    # ============================================================================
    
    def generate_global_explanations(self, n_samples: int = 100) -> Dict[str, Any]:
        """Generate global feature importance explanations"""
        
        logger.info("📊 Generating global explanations...")
        
        # Use cached values if available
        cache_key = f"global_{n_samples}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get SHAP values for sample
        X_sample = self.X_train.sample(min(n_samples, len(self.X_train)))
        
        if self.model_type == 'tree':
            shap_vals = self.explainer.shap_values(X_sample)
            if isinstance(shap_vals, list):
                shap_importance = np.abs(shap_vals[0]).mean(0)
            else:
                shap_importance = np.abs(shap_vals).mean(0)
        else:
            shap_vals = self.explainer.shap_values(X_sample)
            shap_importance = np.abs(shap_vals).mean(0)
        
        # Feature ranking
        feature_ranking = dict(zip(self.feature_names, shap_importance))
        sorted_ranking = dict(
            sorted(feature_ranking.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Clinical interpretation
        clinical_notes = self._generate_clinical_notes(sorted_ranking)
        
        result = {
            'feature_importance': sorted_ranking,
            'top_features': list(sorted_ranking.keys())[:10],
            'shap_summary': {
                'mean_abs_shap': shap_importance.tolist(),
                'features': self.feature_names,
            },
            'clinical_notes': clinical_notes,
        }
        
        # Cache
        if self.cache_explanations:
            self._cache[cache_key] = result
        
        logger.info(f"✅ Global explanations generated")
        logger.info(f"   Top features: {result['top_features'][:5]}")
        
        return result
    
    def generate_local_explanations(
        self,
        X_instance: pd.DataFrame,
        instance_index: int = 0,
    ) -> Dict[str, Any]:
        """Generate explanations for a single patient"""
        
        logger.info(f"🔍 Generating local explanation for patient {instance_index}")
        
        # Get single instance
        instance = X_instance.iloc[[instance_index]]
        
        # Generate SHAP values
        if self.model_type == 'tree':
            shap_vals = self.explainer.shap_values(instance)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0][0]
            else:
                shap_vals = shap_vals[0]
        else:
            shap_vals = self.explainer.shap_values(instance)[0]
        
        # Feature contributions
        contributions = dict(zip(self.feature_names, shap_vals))
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Prediction
        probability = self.model.predict_proba(instance)[0][1]
        risk_score = probability * 100
        risk_level = self._classify_risk(risk_score)
        
        # Clinical explanation
        clinical_explanation = self._generate_clinical_explanation(
            instance.iloc[0],
            sorted_contributions
        )
        
        result = {
            'instance_index': instance_index,
            'risk_score': round(risk_score, 1),
            'risk_level': risk_level,
            'probability': float(probability),
            'base_value': float(self.base_value) if self.base_value is not None else 0.5,
            'shap_values': shap_vals.tolist(),
            'contributing_factors': [
                {
                    'feature': feature,
                    'contribution': float(val),
                    'direction': 'positive' if val > 0 else 'negative',
                    'impact': 'increases' if val > 0 else 'decreases',
                    'magnitude': abs(float(val)),
                }
                for feature, val in sorted_contributions[:10]
            ],
            'top_factors': [feature for feature, _ in sorted_contributions[:5]],
            'clinical_explanation': clinical_explanation,
            'feature_values': instance.iloc[0].to_dict(),
        }
        
        # Cache
        if self.cache_explanations:
            cache_key = f"local_{instance_index}"
            self._cache[cache_key] = result
        
        return result
    
    def generate_batch_explanations(
        self,
        X_batch: pd.DataFrame,
        max_instances: int = 100,
    ) -> List[Dict[str, Any]]:
        """Generate explanations for multiple patients"""
        
        explanations = []
        n = min(max_instances, len(X_batch))
        
        for i in range(n):
            try:
                explanation = self.generate_local_explanations(X_batch, i)
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"❌ Failed to generate explanation for {i}: {str(e)}")
                continue
        
        logger.info(f"✅ Generated {len(explanations)} explanations")
        
        return explanations
    
    # ============================================================================
    # 🔧 Clinical Interpretation
    # ============================================================================
    
    def _generate_clinical_notes(self, feature_ranking: Dict) -> Dict[str, str]:
        """Generate clinical notes for features"""
        
        notes = {}
        
        # Clinical knowledge base
        clinical_kb = {
            'glucose': 'Glucose level - Normal: 70-100 mg/dL, Diabetic: >126 mg/dL',
            'bmi': 'BMI - Normal: 18.5-24.9, Overweight: 25-29.9, Obese: ≥30',
            'age': 'Age - Risk increases significantly after age 45',
            'blood_pressure': 'Blood Pressure - Normal: <120/80 mmHg, Hypertensive: ≥140/90',
            'cholesterol': 'Cholesterol - Desirable: <200 mg/dL, High: ≥240',
            'pregnancies': 'Pregnancies - Higher parity associated with increased risk',
            'insulin': 'Insulin - 2-Hour serum insulin, Normal: <25 mu U/ml',
            'skin_thickness': 'Skin Thickness - Triceps skin fold thickness',
            'diabetes_pedigree': 'Diabetes Pedigree - Family history score',
            'thalach': 'Max Heart Rate - Lower max heart rate indicates higher risk',
            'exang': 'Exercise Angina - Presence indicates higher risk',
            'oldpeak': 'ST Depression - Higher values indicate higher risk',
        }
        
        for feature in feature_ranking:
            if feature in clinical_kb:
                notes[feature] = clinical_kb[feature]
            else:
                notes[feature] = f'Clinical significance of {feature} should be evaluated'
        
        return notes
    
    def _generate_clinical_explanation(
        self,
        patient_data: pd.Series,
        contributions: List[Tuple[str, float]],
    ) -> str:
        """Generate natural language clinical explanation"""
        
        explanations = []
        
        # Risk factors (positive contributions)
        risk_factors = []
        protective_factors = []
        
        for feature, value in contributions[:5]:
            if value > 0.05:
                risk_factors.append(feature)
            elif value < -0.05:
                protective_factors.append(feature)
        
        # Build explanation
        if risk_factors:
            explanation = f"Risk is primarily driven by: {', '.join(risk_factors)}. "
        else:
            explanation = "No significant risk factors identified. "
        
        if protective_factors:
            explanation += f"Protective factors include: {', '.join(protective_factors)}. "
        
        # Add clinical context with actual values
        clinical_context = []
        
        if 'glucose' in patient_data.index and patient_data['glucose'] > 126:
            clinical_context.append(
                f"Glucose ({patient_data['glucose']:.1f} mg/dL) is elevated (>126 mg/dL)"
            )
        if 'bmi' in patient_data.index and patient_data['bmi'] > 30:
            clinical_context.append(
                f"BMI ({patient_data['bmi']:.1f}) indicates obesity"
            )
        if 'age' in patient_data.index and patient_data['age'] > 60:
            clinical_context.append(
                f"Age ({patient_data['age']:.0f} years) is a significant risk factor"
            )
        if 'blood_pressure' in patient_data.index and patient_data['blood_pressure'] > 90:
            clinical_context.append(
                f"Blood pressure ({patient_data['blood_pressure']:.1f}) is elevated"
            )
        if 'cholesterol' in patient_data.index and patient_data['cholesterol'] > 240:
            clinical_context.append(
                f"Cholesterol ({patient_data['cholesterol']:.1f} mg/dL) is high"
            )
        
        if clinical_context:
            explanation += "Clinically relevant findings: " + ". ".join(clinical_context)
        
        return explanation
    
    def _classify_risk(self, risk_score: float) -> str:
        """Classify risk level"""
        if risk_score < 30:
            return "Low Risk"
        elif risk_score < 60:
            return "Moderate Risk"
        else:
            return "High Risk"
    
    # ============================================================================
    # 📊 Visualization Methods
    # ============================================================================
    
    def plot_summary(
        self,
        X_sample: Optional[pd.DataFrame] = None,
        max_display: int = 10,
        save_path: Optional[str] = None,
    ):
        """Generate SHAP summary plot"""
        
        X_plot = X_sample if X_sample is not None else self.X_train[:100]
        
        plt.figure(figsize=(12, 8))
        
        if self.model_type == 'tree':
            shap_vals = self.explainer.shap_values(X_plot)
            if isinstance(shap_vals, list):
                shap.summary_plot(shap_vals[0], X_plot, max_display=max_display, show=False)
            else:
                shap.summary_plot(shap_vals, X_plot, max_display=max_display, show=False)
        else:
            shap_vals = self.explainer.shap_values(X_plot)
            shap.summary_plot(shap_vals, X_plot, max_display=max_display, show=False)
        
        plt.title('SHAP Feature Importance Summary', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Summary plot saved to {save_path}")
        
        plt.close()
    
    def plot_waterfall(
        self,
        X_instance: pd.DataFrame,
        index: int = 0,
        save_path: Optional[str] = None,
    ):
        """Generate waterfall plot"""
        
        instance = X_instance.iloc[[index]]
        
        if self.model_type == 'tree':
            shap_vals = self.explainer.shap_values(instance)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0][0]
            else:
                shap_vals = shap_vals[0]
        else:
            shap_vals = self.explainer.shap_values(instance)[0]
        
        # Create Explanation object
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=self.base_value if self.base_value is not None else 0.5,
            data=instance.iloc[0].values,
            feature_names=instance.columns.tolist()
        )
        
        plt.figure(figsize=(14, 8))
        shap.waterfall_plot(explanation, show=False, max_display=8)
        plt.title(f'SHAP Waterfall Plot - Patient {index}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Waterfall plot saved to {save_path}")
        
        plt.close()
    
    def plot_force(
        self,
        X_instance: pd.DataFrame,
        index: int = 0,
        save_path: Optional[str] = None,
    ):
        """Generate force plot"""
        
        instance = X_instance.iloc[[index]]
        
        if self.model_type == 'tree':
            shap_vals = self.explainer.shap_values(instance)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            else:
                shap_vals = shap_vals
        else:
            shap_vals = self.explainer.shap_values(instance)
        
        plt.figure(figsize=(16, 6))
        
        shap.force_plot(
            base_value=self.base_value,
            shap_values=shap_vals,
            features=instance,
            matplotlib=True,
            show=False,
        )
        
        plt.title(f'SHAP Force Plot - Patient {index}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Force plot saved to {save_path}")
        
        plt.close()
    
    # ============================================================================
    # 🔧 Utility Methods
    # ============================================================================
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_names
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'cached_explanations': len(self._cache),
            'cache_keys': list(self._cache.keys())[:10],
        }
    
    def clear_cache(self):
        """Clear explanation cache"""
        self._cache.clear()
        logger.info("🧹 Cache cleared")