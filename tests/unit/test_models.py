# tests/unit/test_models.py
"""
Unit tests for model components
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.components.model_training import ClinicalModelTrainer
from src.components.model_evaluation import ModelEvaluator


class TestModelTraining:
    """Test model training components"""
    
    def test_model_initialization(self):
        """Test model trainer initialization"""
        trainer = ClinicalModelTrainer(
            model_names=['logistic_regression', 'random_forest']
        )
        assert len(trainer.models) == 2
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        # Create sample data
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        
        # Train model
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        
        # Get importance
        trainer = ClinicalModelTrainer()
        importance = trainer.get_feature_importance(model, X.columns)
        
        assert len(importance) == 3
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns


class TestModelEvaluation:
    """Test model evaluation components"""
    
    def test_metrics_calculation(self):
        """Test metrics calculation"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        
        evaluator = ModelEvaluator()
        metrics = evaluator._calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'specificity' in metrics