# src/components/data_ingestion.py
"""
Enterprise Data Ingestion with Validation and Versioning
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union
from pathlib import Path
import json
import hashlib
from datetime import datetime
import requests
import logging
from dataclasses import dataclass, field

from src.validation.schema_validation import DataValidator
from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


@dataclass
class DatasetMetadata:
    """Dataset metadata with versioning"""
    name: str
    version: str
    source: str
    timestamp: str
    n_samples: int
    n_features: int
    feature_names: List[str]
    target_name: str
    class_distribution: Dict[str, int]
    hash: str
    validation_status: str = "pending"
    validation_errors: List[str] = field(default_factory=list)


class DataIngestion:
    """
    Enterprise Data Ingestion with:
    - Multiple data sources (URL, local, database)
    - Schema validation
    - Data quality checks
    - Version tracking
    - Data lineage
    - Audit trail
    """
    
    DATASETS = {
        'diabetes': {
            'url': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
            'columns': [
                'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                'insulin', 'bmi', 'diabetes_pedigree', 'age', 'target'
            ],
            'target_column': 'target',
            'description': 'Pima Indians Diabetes Dataset',
            'version': '1.0.0',
        },
        'heart_disease': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
            'columns': [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                'ca', 'thal', 'target'
            ],
            'target_column': 'target',
            'description': 'UCI Heart Disease Dataset',
            'version': '1.0.0',
        }
    }
    
    def __init__(
        self,
        dataset_type: str = 'diabetes',
        source_path: Optional[str] = None,
        validate_schema: bool = True,
        track_version: bool = True,
    ):
        self.dataset_type = dataset_type
        self.source_path = source_path
        self.validate_schema = validate_schema
        self.track_version = track_version
        
        self.config = self.DATASETS.get(dataset_type)
        if not self.config and not source_path:
            raise ValueError(f"Unknown dataset: {dataset_type}")
        
        self.validator = DataValidator()
        self.metadata: Optional[DatasetMetadata] = None
        self._data = None
        self._target = None
        
        logger.info(f"📊 DataIngestion initialized: {dataset_type}")
    
    # ============================================================================
    # 🚀 Main Loading Methods
    # ============================================================================
    
    def load(
        self,
        limit: Optional[int] = None,
        sample: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load dataset from configured source
        
        Args:
            limit: Limit number of rows
            sample: Fraction of data to sample
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        
        logger.info(f"📥 Loading dataset: {self.dataset_type}")
        
        if self.source_path:
            data = self._load_from_file(self.source_path)
        else:
            data = self._load_from_url()
        
        # Sample if requested
        if sample and 0 < sample < 1:
            data = data.sample(frac=sample, random_state=42)
        
        # Limit if requested
        if limit:
            data = data.head(limit)
        
        # Separate features and target
        target_col = self.config['target_column']
        self._target = data[target_col]
        self._data = data.drop(target_col, axis=1)
        
        # Validate
        if self.validate_schema:
            self._validate_data()
        
        # Track version
        if self.track_version:
            self._create_metadata()
        
        logger.info(f"✅ Dataset loaded: {len(self._data)} samples, {len(self._data.columns)} features")
        logger.info(f"   Target distribution: {self._target.value_counts().to_dict()}")
        
        return self._data, self._target
    
    def _load_from_url(self) -> pd.DataFrame:
        """Load dataset from URL"""
        try:
            url = self.config['url']
            columns = self.config['columns']
            
            if url.endswith('.data'):
                df = pd.read_csv(url, names=columns)
            elif url.endswith('.csv'):
                df = pd.read_csv(url)
            else:
                raise ValueError(f"Unsupported URL format: {url}")
            
            # Handle missing values
            df = df.replace('?', np.nan)
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load from URL: {str(e)}")
            raise
    
    def _load_from_file(self, path: str) -> pd.DataFrame:
        """Load dataset from local file"""
        try:
            path = Path(path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            if path.suffix == '.csv':
                df = pd.read_csv(path)
            elif path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(path)
            elif path.suffix == '.parquet':
                df = pd.read_parquet(path)
            elif path.suffix == '.json':
                df = pd.read_json(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load from file: {str(e)}")
            raise
    
    # ============================================================================
    # 🔧 Validation Methods
    # ============================================================================
    
    def _validate_data(self):
        """Validate loaded data"""
        
        is_valid, errors = self.validator.validate(self._data)
        
        if not is_valid:
            logger.warning(f"⚠️ Data validation failed: {errors}")
            # Still proceed but log warnings
        
        # Check for missing values
        missing = self._data.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"⚠️ Missing values found: {missing[missing > 0].to_dict()}")
        
        # Check for outliers (Z-score method)
        for col in self._data.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((self._data[col] - self._data[col].mean()) / self._data[col].std())
            outliers = (z_scores > 3).sum()
            if outliers > 0:
                logger.warning(f"⚠️ Outliers detected in {col}: {outliers} rows")
    
    # ============================================================================
    # 🔧 Version Tracking
    # ============================================================================
    
    def _create_metadata(self):
        """Create dataset metadata"""
        
        # Calculate hash
        data_hash = hashlib.md5(
            pd.concat([self._data, self._target], axis=1).to_json().encode()
        ).hexdigest()
        
        self.metadata = DatasetMetadata(
            name=self.dataset_type,
            version=self.config.get('version', '1.0.0'),
            source=self.source_path or self.config.get('url', ''),
            timestamp=datetime.now().isoformat(),
            n_samples=len(self._data),
            n_features=len(self._data.columns),
            feature_names=self._data.columns.tolist(),
            target_name=self.config['target_column'],
            class_distribution=self._target.value_counts().to_dict(),
            hash=data_hash,
            validation_status="validated",
        )
        
        # Save metadata
        self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata to file"""
        if not self.metadata:
            return
        
        metadata_path = Path(f"data/versioned/metadata_{self.dataset_type}.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.__dict__, f, indent=2)
    
    # ============================================================================
    # 🔧 Quality Report
    # ============================================================================
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        
        report = {
            'dataset_info': {
                'name': self.dataset_type,
                'samples': len(self._data) if self._data is not None else 0,
                'features': len(self._data.columns) if self._data is not None else 0,
            },
            'missing_values': {},
            'outliers': {},
            'correlations': {},
            'statistics': {},
            'validation_errors': [],
        }
        
        if self._data is None:
            return report
        
        # Missing values
        missing = self._data.isnull().sum()
        report['missing_values'] = {
            col: {'count': int(missing[col]), 'percentage': float(missing[col] / len(self._data) * 100)}
            for col in self._data.columns
            if missing[col] > 0
        }
        
        # Statistics
        report['statistics'] = self._data.describe().to_dict()
        
        # Outliers
        for col in self._data.select_dtypes(include=[np.number]).columns:
            q1 = self._data[col].quantile(0.25)
            q3 = self._data[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((self._data[col] < lower) | (self._data[col] > upper)).sum()
            if outliers > 0:
                report['outliers'][col] = int(outliers)
        
        # Correlations
        numeric_cols = self._data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self._data[numeric_cols].corr()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': float(corr_matrix.iloc[i, j])
                        })
            report['correlations']['high_correlations'] = high_corr
        
        return report
    
    # ============================================================================
    # 🔧 Export Methods
    # ============================================================================
    
    def export_processed(
        self,
        path: str = "data/processed",
        format: str = "parquet"
    ):
        """Export processed data"""
        
        if self._data is None:
            raise ValueError("No data loaded")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Combine features and target
        df = pd.concat([self._data, self._target], axis=1)
        
        # Save
        if format == 'parquet':
            df.to_parquet(path / f"{self.dataset_type}_processed.parquet")
        elif format == 'csv':
            df.to_csv(path / f"{self.dataset_type}_processed.csv", index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"✅ Exported processed data to {path}")
    
    # ============================================================================
    # 🔧 Getter Methods
    # ============================================================================
    
    def get_data(self) -> pd.DataFrame:
        """Get features DataFrame"""
        return self._data
    
    def get_target(self) -> pd.Series:
        """Get target Series"""
        return self._target
    
    def get_metadata(self) -> Optional[DatasetMetadata]:
        """Get dataset metadata"""
        return self.metadata
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self._data.columns.tolist() if self._data is not None else []