# src/components/preprocessing.py
"""
Advanced Preprocessing with Feature Engineering and Optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    PowerTransformer, QuantileTransformer
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import joblib
import json
from pathlib import Path

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


class ClinicalPreprocessor:
    """
    Advanced Clinical Preprocessor with:
    - Multiple imputation strategies
    - Feature scaling options
    - Class balancing (SMOTE variants)
    - Feature engineering
    - Feature selection
    - PCA dimensionality reduction
    - Pipeline persistence
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        feature_engineering: bool = True,
    ):
        self.config = config or self._default_config()
        self.feature_engineering = feature_engineering
        
        # Components
        self.imputer = None
        self.scaler = None
        self.sampler = None
        self.selector = None
        self.pca = None
        self._pipeline = None
        
        # State
        self._fitted = False
        self._feature_names = None
        self._preprocessing_report = {}
        
        # Configure
        self._configure_components()
        
        logger.info("🔧 ClinicalPreprocessor initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'imputation': {
                'strategy': 'median',
                'method': 'simple',  # simple, knn, iterative
                'knn_neighbors': 5,
            },
            'scaling': {
                'method': 'standard',  # standard, minmax, robust, power, quantile
            },
            'balancing': {
                'method': 'smote',  # smote, adasyn, borderline, none
                'k_neighbors': 5,
                'sampling_strategy': 'auto',
            },
            'feature_selection': {
                'enabled': False,
                'method': 'kbest',  # kbest, mutual_info
                'n_features': 15,
            },
            'pca': {
                'enabled': False,
                'n_components': 0.95,  # Number of components or variance ratio
            },
        }
    
    def _configure_components(self):
        """Configure preprocessing components"""
        
        # Imputation
        impute_config = self.config['imputation']
        if impute_config['method'] == 'simple':
            strategy = impute_config['strategy']
            self.imputer = SimpleImputer(strategy=strategy)
        elif impute_config['method'] == 'knn':
            self.imputer = KNNImputer(n_neighbors=impute_config['knn_neighbors'])
        elif impute_config['method'] == 'iterative':
            self.imputer = IterativeImputer()
        else:
            raise ValueError(f"Unknown imputation method: {impute_config['method']}")
        
        # Scaling
        scale_config = self.config['scaling']
        if scale_config['method'] == 'standard':
            self.scaler = StandardScaler()
        elif scale_config['method'] == 'minmax':
            self.scaler = MinMaxScaler()
        elif scale_config['method'] == 'robust':
            self.scaler = RobustScaler()
        elif scale_config['method'] == 'power':
            self.scaler = PowerTransformer()
        elif scale_config['method'] == 'quantile':
            self.scaler = QuantileTransformer()
        else:
            raise ValueError(f"Unknown scaling method: {scale_config['method']}")
        
        # Balancing
        balance_config = self.config['balancing']
        if balance_config['method'] == 'smote':
            self.sampler = SMOTE(
                k_neighbors=balance_config['k_neighbors'],
                sampling_strategy=balance_config['sampling_strategy'],
                random_state=42
            )
        elif balance_config['method'] == 'adasyn':
            self.sampler = ADASYN(
                n_neighbors=balance_config['k_neighbors'],
                sampling_strategy=balance_config['sampling_strategy'],
                random_state=42
            )
        elif balance_config['method'] == 'borderline':
            self.sampler = BorderlineSMOTE(
                k_neighbors=balance_config['k_neighbors'],
                sampling_strategy=balance_config['sampling_strategy'],
                random_state=42
            )
        elif balance_config['method'] == 'none':
            self.sampler = None
        else:
            raise ValueError(f"Unknown balancing method: {balance_config['method']}")
    
    # ============================================================================
    # 🚀 Main Processing Methods
    # ============================================================================
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform the data
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Tuple of (transformed features, target)
        """
        
        logger.info("🔄 Starting preprocessing...")
        start_time = time.time()
        
        self._feature_names = X.columns.tolist()
        
        # Create report
        self._preprocessing_report = {
            'original_shape': X.shape,
            'features': X.columns.tolist(),
            'missing_values_original': X.isnull().sum().to_dict(),
            'class_distribution_original': y.value_counts().to_dict() if y is not None else {},
        }
        
        # Step 1: Feature engineering
        if self.feature_engineering:
            X = self._engineer_features(X)
            self._preprocessing_report['feature_engineering'] = {
                'new_features': [col for col in X.columns if col not in self._feature_names]
            }
            self._feature_names = X.columns.tolist()
        
        # Step 2: Handle missing values
        X_imputed = self._impute_missing(X)
        self._preprocessing_report['imputation'] = {
            'method': self.config['imputation']['method'],
            'strategy': self.config['imputation']['strategy'],
            'missing_after': X_imputed.isnull().sum().to_dict(),
        }
        
        # Step 3: Feature scaling
        X_scaled = self._scale_features(X_imputed)
        self._preprocessing_report['scaling'] = {
            'method': self.config['scaling']['method'],
        }
        
        # Step 4: Feature selection
        if self.config['feature_selection']['enabled']:
            X_scaled = self._select_features(X_scaled, y)
            self._preprocessing_report['feature_selection'] = {
                'n_features_original': len(self._feature_names),
                'n_features_selected': X_scaled.shape[1],
                'selected_features': X_scaled.columns.tolist(),
            }
        
        # Step 5: PCA (optional)
        if self.config['pca']['enabled']:
            X_scaled = self._apply_pca(X_scaled)
            self._preprocessing_report['pca'] = {
                'n_components': self.pca.n_components_,
                'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
            }
        
        # Step 6: Class balancing
        if self.sampler is not None and y is not None:
            X_balanced, y_balanced = self._balance_data(X_scaled, y)
            self._preprocessing_report['balancing'] = {
                'method': self.config['balancing']['method'],
                'original_distribution': self._preprocessing_report['class_distribution_original'],
                'balanced_distribution': y_balanced.value_counts().to_dict(),
            }
        else:
            X_balanced, y_balanced = X_scaled, y
        
        self._fitted = True
        
        logger.info("✅ Preprocessing complete")
        logger.info(f"   Time: {(time.time() - start_time):.2f}s")
        logger.info(f"   Shape: {X_balanced.shape}")
        
        return X_balanced, y_balanced
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor"""
        
        if not self._fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        # Feature engineering
        if self.feature_engineering:
            X = self._engineer_features(X)
        
        # Imputation
        X = self._impute_missing(X, fit=False)
        
        # Scaling
        X = self._scale_features(X, fit=False)
        
        # Feature selection
        if self.config['feature_selection']['enabled'] and self.selector:
            X = self._select_features(X, fit=False)
        
        # PCA
        if self.config['pca']['enabled'] and self.pca:
            X = self._apply_pca(X, fit=False)
        
        return X
    
    # ============================================================================
    # 🔧 Feature Engineering
    # ============================================================================
    
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones"""
        
        X_eng = X.copy()
        
        # Clinical domain-specific features
        if 'glucose' in X.columns and 'bmi' in X.columns:
            # Glucose-BMI interaction
            X_eng['glucose_bmi_ratio'] = X['glucose'] / (X['bmi'] + 1e-6)
        
        if 'age' in X.columns and 'glucose' in X.columns:
            # Age-Glucose interaction
            X_eng['age_glucose'] = X['age'] * X['glucose']
        
        if 'bmi' in X.columns:
            # BMI categories
            X_eng['bmi_category'] = pd.cut(
                X['bmi'],
                bins=[0, 18.5, 24.9, 29.9, 100],
                labels=['underweight', 'normal', 'overweight', 'obese']
            )
            X_eng = pd.get_dummies(X_eng, columns=['bmi_category'], prefix='bmi')
        
        if 'age' in X.columns:
            # Age categories
            X_eng['age_category'] = pd.cut(
                X['age'],
                bins=[0, 30, 45, 60, 100],
                labels=['young', 'middle', 'senior', 'elderly']
            )
            X_eng = pd.get_dummies(X_eng, columns=['age_category'], prefix='age')
        
        if 'blood_pressure' in X.columns:
            # Blood pressure categories
            X_eng['bp_high'] = (X['blood_pressure'] > 80).astype(int)
        
        if 'cholesterol' in X.columns:
            # Cholesterol categories
            X_eng['chol_high'] = (X['cholesterol'] > 240).astype(int)
        
        return X_eng
    
    # ============================================================================
    # 🔧 Private Methods
    # ============================================================================
    
    def _impute_missing(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Impute missing values"""
        if fit:
            self.imputer.fit(X)
        return pd.DataFrame(self.imputer.transform(X), columns=X.columns)
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features"""
        if fit:
            self.scaler.fit(X)
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns)
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = True) -> pd.DataFrame:
        """Select top features"""
        if fit:
            if self.config['feature_selection']['method'] == 'kbest':
                self.selector = SelectKBest(
                    score_func=f_classif,
                    k=self.config['feature_selection']['n_features']
                )
            elif self.config['feature_selection']['method'] == 'mutual_info':
                self.selector = SelectKBest(
                    score_func=mutual_info_classif,
                    k=self.config['feature_selection']['n_features']
                )
            else:
                raise ValueError(f"Unknown selection method: {self.config['feature_selection']['method']}")
            
            self.selector.fit(X, y)
        
        # Get selected features
        mask = self.selector.get_support()
        selected_cols = X.columns[mask].tolist()
        
        return X[selected_cols]
    
    def _apply_pca(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Apply PCA dimensionality reduction"""
        if fit:
            self.pca = PCA(
                n_components=self.config['pca']['n_components'],
                random_state=42
            )
            self.pca.fit(X)
        
        components = self.pca.transform(X)
        pca_cols = [f'PC{i+1}' for i in range(components.shape[1])]
        
        return pd.DataFrame(components, columns=pca_cols, index=X.index)
    
    def _balance_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance dataset"""
        X_balanced, y_balanced = self.sampler.fit_resample(X, y)
        return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)
    
    # ============================================================================
    # 🔧 Persistence
    # ============================================================================
    
    def save(self, path: str = "models/preprocessor.pkl"):
        """Save preprocessor to disk"""
        import joblib
        joblib.dump(self, path)
        logger.info(f"✅ Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: str = "models/preprocessor.pkl") -> 'ClinicalPreprocessor':
        """Load preprocessor from disk"""
        import joblib
        return joblib.load(path)
    
    def get_report(self) -> Dict[str, Any]:
        """Get preprocessing report"""
        return self._preprocessing_report