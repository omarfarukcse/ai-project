# src/data/pipeline.py
"""
Data Pipeline with Version Control Integration
"""

import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from src.data.data_manager import get_data_manager
from src.components.data_ingestion import DataIngestion
from src.components.preprocessing import ClinicalPreprocessor
from src.logger import get_logger

logger = get_logger(__name__)


class DataPipeline:
    """
    Complete Data Pipeline with:
    - Data loading (raw)
    - Preprocessing
    - Feature engineering
    - Train/Test split
    - Data versioning
    - Reference data creation
    """
    
    def __init__(self):
        self.data_manager = get_data_manager()
        self.ingestion = DataIngestion()
        self.preprocessor = ClinicalPreprocessor()
        
        logger.info("🔧 DataPipeline initialized")
    
    def run_full_pipeline(
        self,
        dataset_type: str = "diabetes",
        version: bool = True,
        create_reference: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete data pipeline
        
        Args:
            dataset_type: Dataset type (diabetes, heart_disease)
            version: Version the data
            create_reference: Create reference data
            
        Returns:
            Pipeline results
        """
        
        logger.info(f"🚀 Running data pipeline for {dataset_type}")
        
        # Step 1: Load raw data
        raw_df = self._load_raw_data(dataset_type)
        
        # Step 2: Preprocess
        X_train, X_test, y_train, y_test, report = self._preprocess_data(raw_df)
        
        # Step 3: Save processed data
        processed_paths = self._save_processed_data(
            X_train, X_test, y_train, y_test, dataset_type
        )
        
        # Step 4: Create reference data
        if create_reference:
            reference_path = self._create_reference_data(X_train, y_train)
        
        # Step 5: Version data
        if version:
            version_info = self._version_data(dataset_type, processed_paths)
        
        # Step 6: Generate data quality report
        quality_report = self.data_manager.get_data_quality_report(raw_df)
        
        results = {
            "dataset_type": dataset_type,
            "X_train_shape": X_train.shape,
            "X_test_shape": X_test.shape,
            "processed_paths": processed_paths,
            "reference_path": reference_path if create_reference else None,
            "version_info": version_info if version else None,
            "quality_report": quality_report,
            "preprocessing_report": report,
        }
        
        logger.info("✅ Data pipeline complete")
        
        return results
    
    def _load_raw_data(self, dataset_type: str) -> pd.DataFrame:
        """Load raw data from source"""
        
        if dataset_type == "diabetes":
            df = self.ingestion.load_diabetes_dataset()
        elif dataset_type == "heart_disease":
            df = self.ingestion.load_heart_disease_dataset()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Save raw data
        self.data_manager.save_raw_data(df, dataset_type)
        
        return df
    
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple:
        """Preprocess data"""
        
        X_train, X_test, y_train, y_test, report = self.preprocessor.create_pipeline(df)
        
        return X_train, X_test, y_train, y_test, report
    
    def _save_processed_data(self, X_train, X_test, y_train, y_test, name: str) -> Dict:
        """Save processed data"""
        
        feature_names = X_train.columns.tolist()
        
        paths = self.data_manager.save_processed_data(
            X_train, X_test, y_train, y_test,
            name=name,
            feature_names=feature_names,
        )
        
        return paths
    
    def _create_reference_data(self, X_train, y_train) -> Path:
        """Create reference data for drift detection"""
        
        # Combine features and target
        reference_df = X_train.copy()
        reference_df['target'] = y_train
        
        # Save as reference
        path = self.data_manager.save_reference_data(
            reference_df,
            name="reference_data",
            description="Reference data for drift detection",
        )
        
        return path
    
    def _version_data(self, dataset_type: str, processed_paths: Dict) -> Dict:
        """Version the data"""
        
        versions = {}
        
        for name, path in processed_paths.items():
            version_info = self.data_manager.version_data(
                name=f"{dataset_type}_{name}",
                path=path,
                commit_message=f"Processed {dataset_type} data",
            )
            versions[name] = version_info
        
        return versions
    
    def load_latest_data(
        self,
        dataset_type: str = "diabetes",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
        """
        Load the latest processed data
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, metadata)
        """
        
        return self.data_manager.load_processed_data(dataset_type)
    
    def get_data_version_history(self, dataset_type: str) -> List[Dict]:
        """Get version history for a dataset"""
        
        versions = []
        for dvc_file in (self.data_manager.versioned_dir / f"{dataset_type}_*").glob("*.dvc"):
            with open(dvc_file, 'r') as f:
                versions.append(json.load(f))
        
        return sorted(versions, key=lambda x: x["timestamp"], reverse=True)