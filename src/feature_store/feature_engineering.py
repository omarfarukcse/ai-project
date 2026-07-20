# src/feature_store/feature_engineering.py
"""
Advanced Feature Engineering with Automated Feature Creation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib
from datetime import datetime, timedelta
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PolynomialFeatures, KBinsDiscretizer
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


class FeatureType(Enum):
    """Feature data types"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    DATETIME = "datetime"
    TEXT = "text"
    ARRAY = "array"
    EMBEDDING = "embedding"


class FeatureStatus(Enum):
    """Feature status"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class FeatureDefinition:
    """Feature definition with metadata"""
    name: str
    description: str
    feature_type: FeatureType
    status: FeatureStatus = FeatureStatus.ACTIVE
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    owner: str = "system"
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    compute_fn: Optional[Callable] = None
    validation_fn: Optional[Callable] = None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "feature_type": self.feature_type.value,
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "owner": self.owner,
            "tags": self.tags,
            "dependencies": self.dependencies,
        }


@dataclass
class FeatureGroup:
    """Group of related features"""
    name: str
    description: str
    features: List[FeatureDefinition] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)


class FeatureEngineer:
    """
    Advanced Feature Engineering with:
    - Automatic feature creation
    - Interaction features
    - Aggregation features
    - Derived features
    - Feature validation
    - Transformation pipelines
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.feature_registry: Dict[str, FeatureDefinition] = {}
        self.transformers: Dict[str, Any] = {}
        self._fitted = False
        
        # Initialize default transformers
        self._init_default_transformers()
        
        logger.info("🔧 FeatureEngineer initialized")
    
    def _init_default_transformers(self):
        """Initialize default transformers"""
        self.transformers.update({
            'standard_scaler': StandardScaler(),
            'minmax_scaler': MinMaxScaler(),
            'robust_scaler': RobustScaler(),
            'polynomial': PolynomialFeatures(degree=2, include_bias=False),
            'discretizer': KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile'),
        })
    
    # ============================================================================
    # 🚀 Feature Creation Methods
    # ============================================================================
    
    def create_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create clinical domain-specific features
        
        Args:
            df: Input DataFrame with raw clinical data
            
        Returns:
            DataFrame with engineered features
        """
        
        logger.info("🏥 Creating clinical features...")
        df_eng = df.copy()
        
        # BMI-related features
        if 'bmi' in df.columns:
            df_eng['bmi_category'] = pd.cut(
                df['bmi'],
                bins=[0, 18.5, 24.9, 29.9, 100],
                labels=['underweight', 'normal', 'overweight', 'obese']
            )
            df_eng['bmi_risk'] = (df['bmi'] > 25).astype(int)
            df_eng['bmi_squared'] = df['bmi'] ** 2
        
        # Glucose-related features
        if 'glucose' in df.columns:
            df_eng['glucose_risk'] = (df['glucose'] > 126).astype(int)
            df_eng['glucose_category'] = pd.cut(
                df['glucose'],
                bins=[0, 70, 100, 126, 300],
                labels=['low', 'normal', 'pre_diabetic', 'diabetic']
            )
        
        # Blood pressure features
        if 'blood_pressure' in df.columns:
            df_eng['bp_risk'] = (df['blood_pressure'] > 80).astype(int)
            df_eng['bp_category'] = pd.cut(
                df['blood_pressure'],
                bins=[0, 60, 80, 90, 200],
                labels=['low', 'normal', 'pre_hypertensive', 'hypertensive']
            )
        
        # Age-related features
        if 'age' in df.columns:
            df_eng['age_squared'] = df['age'] ** 2
            df_eng['age_log'] = np.log1p(df['age'])
            df_eng['age_category'] = pd.cut(
                df['age'],
                bins=[0, 30, 45, 60, 120],
                labels=['young', 'middle', 'senior', 'elderly']
            )
        
        # Interaction features
        if 'glucose' in df.columns and 'bmi' in df.columns:
            df_eng['glucose_bmi_interaction'] = df['glucose'] * df['bmi']
            df_eng['glucose_bmi_ratio'] = df['glucose'] / (df['bmi'] + 1e-6)
        
        if 'age' in df.columns and 'glucose' in df.columns:
            df_eng['age_glucose_interaction'] = df['age'] * df['glucose']
        
        if 'bmi' in df.columns and 'blood_pressure' in df.columns:
            df_eng['bmi_bp_interaction'] = df['bmi'] * df['blood_pressure']
        
        # Create one-hot encodings for categorical features
        categorical_cols = [col for col in df_eng.columns if '_category' in col]
        for col in categorical_cols:
            if col in df_eng.columns:
                dummies = pd.get_dummies(df_eng[col], prefix=col, drop_first=True)
                df_eng = pd.concat([df_eng, dummies], axis=1)
                df_eng = df_eng.drop(col, axis=1)
        
        logger.info(f"✅ Created {len(df_eng.columns) - len(df.columns)} new features")
        
        return df_eng
    
    def create_aggregation_features(
        self,
        df: pd.DataFrame,
        group_col: str,
        agg_cols: List[str],
        agg_funcs: List[str],
        prefix: str = "agg"
    ) -> pd.DataFrame:
        """
        Create aggregation features for grouped data
        
        Args:
            df: Input DataFrame
            group_col: Column to group by
            agg_cols: Columns to aggregate
            agg_funcs: Aggregation functions
            prefix: Prefix for new feature names
            
        Returns:
            DataFrame with aggregation features
        """
        
        agg_dict = {}
        for col in agg_cols:
            for func in agg_funcs:
                agg_dict[col] = agg_dict.get(col, []) + [func]
        
        grouped = df.groupby(group_col).agg(agg_dict)
        grouped.columns = [f"{prefix}_{col}_{func}" for col, func in grouped.columns]
        
        return grouped
    
    def create_time_series_features(
        self,
        df: pd.DataFrame,
        time_col: str,
        value_cols: List[str],
        windows: List[int],
        agg_funcs: List[str]
    ) -> pd.DataFrame:
        """
        Create time series features (rolling windows, lags, etc.)
        
        Args:
            df: Input DataFrame with time series data
            time_col: Time column name
            value_cols: Value columns to transform
            windows: Rolling window sizes
            agg_funcs: Aggregation functions
            
        Returns:
            DataFrame with time series features
        """
        
        df_ts = df.copy()
        df_ts = df_ts.sort_values(time_col)
        
        for col in value_cols:
            # Rolling statistics
            for window in windows:
                for func in agg_funcs:
                    if func == 'mean':
                        df_ts[f'{col}_rolling_{window}_{func}'] = df_ts[col].rolling(window).mean()
                    elif func == 'std':
                        df_ts[f'{col}_rolling_{window}_{func}'] = df_ts[col].rolling(window).std()
                    elif func == 'min':
                        df_ts[f'{col}_rolling_{window}_{func}'] = df_ts[col].rolling(window).min()
                    elif func == 'max':
                        df_ts[f'{col}_rolling_{window}_{func}'] = df_ts[col].rolling(window).max()
            
            # Lag features
            for lag in [1, 3, 7]:
                df_ts[f'{col}_lag_{lag}'] = df_ts[col].shift(lag)
            
            # Difference features
            df_ts[f'{col}_diff_1'] = df_ts[col].diff(1)
            df_ts[f'{col}_diff_3'] = df_ts[col].diff(3)
            df_ts[f'{col}_pct_change'] = df_ts[col].pct_change()
        
        return df_ts
    
    def create_embedding_features(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        embedding_dim: int = 50
    ) -> pd.DataFrame:
        """
        Create embedding features from categorical variables
        
        Note: This requires training embeddings on the data
        """
        
        # Placeholder - in production, use learned embeddings
        df_emb = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                # Simple target encoding as proxy for embeddings
                target_enc = df.groupby(col).size().rank(pct=True)
                df_emb[f'{col}_embedding'] = df[col].map(target_enc)
        
        return df_emb
    
    # ============================================================================
    # 🔧 Feature Transformation Methods
    # ============================================================================
    
    def transform_features(
        self,
        df: pd.DataFrame,
        methods: List[str] = None
    ) -> pd.DataFrame:
        """
        Apply transformations to features
        
        Args:
            df: Input DataFrame
            methods: List of transformation methods
            
        Returns:
            DataFrame with transformed features
        """
        
        if methods is None:
            methods = ['standard_scaler']
        
        df_trans = df.copy()
        
        for method in methods:
            if method in self.transformers:
                transformer = self.transformers[method]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    if self._fitted:
                        transformed = transformer.transform(df[numeric_cols])
                    else:
                        transformed = transformer.fit_transform(df[numeric_cols])
                    
                    # Create new column names
                    new_cols = [f'{col}_{method}' for col in numeric_cols]
                    df_trans[new_cols] = transformed
        
        self._fitted = True
        return df_trans
    
    def apply_pca(
        self,
        df: pd.DataFrame,
        n_components: int = 5
    ) -> pd.DataFrame:
        """
        Apply PCA dimensionality reduction
        
        Args:
            df: Input DataFrame
            n_components: Number of PCA components
            
        Returns:
            DataFrame with PCA features
        """
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < n_components:
            n_components = len(numeric_cols)
        
        pca = PCA(n_components=n_components, random_state=42)
        
        if self._fitted:
            components = pca.transform(df[numeric_cols])
        else:
            components = pca.fit_transform(df[numeric_cols])
        
        pca_cols = [f'pca_{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(components, columns=pca_cols, index=df.index)
        
        return pd.concat([df, df_pca], axis=1)
    
    # ============================================================================
    # 🔧 Feature Selection Methods
    # ============================================================================
    
    def select_best_features(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        k: int = 20
    ) -> List[str]:
        """
        Select top k features using statistical tests
        
        Args:
            df: Input DataFrame
            target: Target variable
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return []
        
        selector = SelectKBest(score_func=f_classif, k=min(k, len(numeric_cols)))
        selector.fit(df[numeric_cols], target)
        
        # Get selected indices
        selected_indices = selector.get_support(indices=True)
        selected_features = [numeric_cols[i] for i in selected_indices]
        
        # Get scores
        scores = selector.scores_
        feature_scores = dict(zip(numeric_cols, scores))
        
        logger.info(f"✅ Selected {len(selected_features)} features")
        
        return selected_features
    
    # ============================================================================
    # 🔧 Utility Methods
    # ============================================================================
    
    def register_feature(
        self,
        name: str,
        compute_fn: Callable,
        description: str,
        feature_type: FeatureType = FeatureType.NUMERICAL,
        dependencies: List[str] = None
    ):
        """Register a custom feature"""
        
        feature = FeatureDefinition(
            name=name,
            description=description,
            feature_type=feature_type,
            dependencies=dependencies or [],
            compute_fn=compute_fn,
        )
        
        self.feature_registry[name] = feature
        logger.info(f"✅ Registered feature: {name}")
    
    def compute_feature(self, name: str, df: pd.DataFrame) -> pd.Series:
        """Compute a registered feature"""
        
        if name not in self.feature_registry:
            raise ValueError(f"Feature not found: {name}")
        
        feature = self.feature_registry[name]
        if feature.compute_fn:
            return feature.compute_fn(df)
        else:
            raise ValueError(f"Feature {name} has no compute function")
    
    def get_feature_names(self) -> List[str]:
        """Get all registered feature names"""
        return list(self.feature_registry.keys())
    
    def get_feature_info(self, name: str) -> Dict:
        """Get feature information"""
        if name in self.feature_registry:
            return self.feature_registry[name].to_dict()
        return {}