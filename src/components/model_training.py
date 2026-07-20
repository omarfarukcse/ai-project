# src/components/model_training.py
"""
Advanced Model Training with Hyperparameter Optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import make_scorer, recall_score
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib
import json
import time

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


class ClinicalModelTrainer:
    """
    Advanced Model Training with:
    - Multiple model support
    - Hyperparameter optimization (Grid/Random/Optuna)
    - Cross-validation with clinical focus
    - Model persistence
    - Performance tracking
    """
    
    MODELS = {
        'logistic_regression': {
            'class': LogisticRegression,
            'default_params': {'random_state': 42, 'max_iter': 1000, 'class_weight': 'balanced'},
            'param_grid': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': [None, 'balanced'],
            },
            'description': 'Baseline interpretable model',
        },
        'random_forest': {
            'class': RandomForestClassifier,
            'default_params': {'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'},
            'param_grid': {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': [None, 'balanced', 'balanced_subsample'],
            },
            'description': 'Ensemble method with feature importance',
        },
        'xgboost': {
            'class': XGBClassifier,
            'default_params': {
                'random_state': 42,
                'n_jobs': -1,
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'scale_pos_weight': 1,
            },
            'param_grid': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.3],
                'scale_pos_weight': [1, 2, 3, 5],
            },
            'description': 'High-performance gradient boosting',
        },
        'lightgbm': {
            'class': LGBMClassifier,
            'default_params': {'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'},
            'param_grid': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [-1, 10, 20, 30],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'num_leaves': [31, 50, 100],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'min_child_samples': [5, 10, 20],
                'class_weight': [None, 'balanced'],
            },
            'description': 'Lightweight gradient boosting',
        },
        'gradient_boosting': {
            'class': GradientBoostingClassifier,
            'default_params': {'random_state': 42},
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10],
            },
            'description': 'Traditional gradient boosting',
        },
        'catboost': {
            'class': CatBoostClassifier,
            'default_params': {'random_state': 42, 'verbose': False},
            'param_grid': {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'l2_leaf_reg': [1, 3, 5, 7],
            },
            'description': 'High-performance categorical boosting',
        },
    }
    
    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        optimization: str = 'grid',  # grid, random, optuna
        cv_folds: int = 5,
        n_trials: int = 50,
        primary_metric: str = 'recall',
    ):
        self.model_names = model_names or ['logistic_regression', 'random_forest', 'xgboost']
        self.optimization = optimization
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.primary_metric = primary_metric
        
        self.models = {}
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.cv_results = {}
        self.training_history = []
        
        # Create scorer
        self.scorer = make_scorer(recall_score)
        
        # Setup Optuna if using
        if optimization == 'optuna':
            self.setup_optuna()
        
        logger.info(f"🤖 ModelTrainer initialized: {len(self.model_names)} models")
    
    def setup_optuna(self):
        """Setup Optuna study"""
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(),
        )
    
    # ============================================================================
    # 🚀 Training Methods
    # ============================================================================
    
    def create_models(self) -> Dict[str, Dict]:
        """Initialize model configurations"""
        self.models = {
            name: self.MODELS[name]
            for name in self.model_names
            if name in self.MODELS
        }
        return self.models
    
    def train_with_optimization(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train models with hyperparameter optimization
        
        Returns:
            Dictionary of trained models with their metrics
        """
        
        results = {}
        start_time = time.time()
        
        for model_name, model_config in self.models.items():
            logger.info(f"🔄 Training {model_name}...")
            
            if self.optimization == 'grid':
                result = self._train_with_grid_search(
                    model_name, model_config, X_train, y_train
                )
            elif self.optimization == 'random':
                result = self._train_with_random_search(
                    model_name, model_config, X_train, y_train
                )
            elif self.optimization == 'optuna':
                result = self._train_with_optuna(
                    model_name, model_config, X_train, y_train
                )
            else:
                result = self._train_baseline(
                    model_name, model_config, X_train, y_train
                )
            
            results[model_name] = result
            self.trained_models[model_name] = result['model']
            
            logger.info(f"✅ {model_name} trained: {result['cv_score']:.3f} ± {result['cv_std']:.3f}")
        
        logger.info(f"✅ All models trained in {time.time() - start_time:.2f}s")
        return results
    
    def _train_with_grid_search(
        self,
        model_name: str,
        model_config: Dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Dict[str, Any]:
        """Train with Grid Search"""
        
        base_model = model_config['class'](**model_config['default_params'])
        
        grid_search = GridSearchCV(
            base_model,
            model_config['param_grid'],
            cv=self.cv_folds,
            scoring=self.scorer,
            n_jobs=-1,
            verbose=0,
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            best_model, X_train, y_train,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
            scoring=self.scorer
        )
        
        return {
            'model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'search_results': grid_search.cv_results_,
        }
    
    def _train_with_random_search(
        self,
        model_name: str,
        model_config: Dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Dict[str, Any]:
        """Train with Random Search"""
        
        base_model = model_config['class'](**model_config['default_params'])
        
        random_search = RandomizedSearchCV(
            base_model,
            model_config['param_grid'],
            n_iter=50,
            cv=self.cv_folds,
            scoring=self.scorer,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        
        random_search.fit(X_train, y_train)
        
        return {
            'model': random_search.best_estimator_,
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_score': random_search.best_score_,
            'cv_std': 0,
            'cv_scores': [],
            'search_results': random_search.cv_results_,
        }
    
    def _train_with_optuna(
        self,
        model_name: str,
        model_config: Dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Dict[str, Any]:
        """Train with Optuna optimization"""
        
        def objective(trial):
            # Suggest hyperparameters
            params = {}
            for param_name, param_values in model_config['param_grid'].items():
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(
                        param_name,
                        min(param_values),
                        max(param_values),
                        step=1 if len(param_values) > 1 else 1
                    )
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(
                        param_name,
                        min(param_values),
                        max(param_values),
                        log=True if param_values[0] < 0.1 else False
                    )
                else:
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_values
                    )
            
            # Train model
            model = model_config['class'](**params)
            scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring=self.scorer
            )
            
            return scores.mean()
        
        # Run optimization
        self.study.optimize(objective, n_trials=self.n_trials)
        
        # Train best model
        best_params = self.study.best_params
        model = model_config['class'](**best_params)
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'best_params': best_params,
            'best_score': self.study.best_value,
            'cv_score': self.study.best_value,
            'cv_std': 0,
            'cv_scores': [],
            'search_results': self.study.trials,
        }
    
    def _train_baseline(
        self,
        model_name: str,
        model_config: Dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Dict[str, Any]:
        """Train baseline model without optimization"""
        
        model = model_config['class'](**model_config['default_params'])
        model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
            scoring=self.scorer
        )
        
        return {
            'model': model,
            'best_params': model_config['default_params'],
            'best_score': cv_scores.mean(),
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'search_results': {},
        }
    
    # ============================================================================
    # 🔧 Model Selection
    # ============================================================================
    
    def select_best_model(
        self,
        trained_models: Dict,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metric: str = 'recall',
    ) -> Tuple[Any, str]:
        """
        Select best model based on metric
        """
        
        from src.components.model_evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_models(trained_models, X_test, y_test)
        
        # Select best
        best_model_name = max(
            evaluation_results,
            key=lambda x: evaluation_results[x].get(metric, 0)
        )
        
        self.best_model = trained_models[best_model_name]
        self.best_model_name = best_model_name
        
        logger.info(f"🏆 Best model: {best_model_name}")
        logger.info(f"   {metric}: {evaluation_results[best_model_name].get(metric, 0):.3f}")
        
        return self.best_model, best_model_name
    
    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """
        Extract feature importance from model
        """
        
        importance_dict = {}
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_dict = dict(zip(feature_names, importance))
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                importance = np.abs(model.coef_)
            else:
                importance = np.abs(model.coef_[0])
            importance_dict = dict(zip(feature_names, importance))
        elif hasattr(model, 'best_estimator_'):
            return self.get_feature_importance(model.best_estimator_, feature_names)
        else:
            logger.warning("Model does not provide feature importance")
            return pd.DataFrame()
        
        # Sort
        importance_df = pd.DataFrame(
            list(importance_dict.items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        
        return importance_df
    
    # ============================================================================
    # 🔧 Persistence
    # ============================================================================
    
    def save_models(self, path: str = "models/"):
        """Save all trained models"""
        
        from pathlib import Path
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            joblib.dump(model, path / f"{model_name}.pkl")
        
        # Save best model separately
        if self.best_model:
            joblib.dump(self.best_model, path / "best_model.pkl")
        
        logger.info(f"✅ Models saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> Any:
        """Load a model"""
        return joblib.load(path)