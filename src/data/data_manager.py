# src/data/data_manager.py
"""
Enterprise Data Management Layer with Version Control
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import hashlib
import shutil
import logging

from src.logger import get_logger
from src.config_manager import get_config_manager

logger = get_logger(__name__)


class DataManager:
    """
    Enterprise Data Manager with:
    - Multi-format data loading (CSV, Parquet, JSON, Excel)
    - Data versioning
    - Data quality validation
    - Data lineage tracking
    - Feature store management
    - Reference data management
    - Data export/import
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.feature_store_dir = self.base_dir / "feature_store"
        self.versioned_dir = self.base_dir / "versioned"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, 
                        self.feature_store_dir, self.versioned_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._metadata_cache: Dict[str, Dict] = {}
        
        logger.info(f"📁 DataManager initialized: {self.base_dir}")
    
    # ============================================================================
    # 🚀 Raw Data Management
    # ============================================================================
    
    def save_raw_data(
        self,
        df: pd.DataFrame,
        name: str,
        source: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Save raw data with metadata
        
        Args:
            df: DataFrame to save
            name: Dataset name
            source: Data source
            metadata: Additional metadata
            
        Returns:
            Path to saved file
        """
        
        # Create dataset directory
        dataset_dir = self.raw_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save data
        file_path = dataset_dir / f"{name}_{timestamp}.parquet"
        df.to_parquet(file_path, index=False)
        
        # Save metadata
        metadata_path = dataset_dir / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
        else:
            existing_metadata = {}
        
        existing_metadata[timestamp] = {
            "timestamp": timestamp,
            "source": source,
            "rows": len(df),
            "columns": len(df.columns),
            "columns_list": df.columns.tolist(),
            "file_path": str(file_path),
            "hash": self._calculate_hash(df),
            "metadata": metadata or {},
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(existing_metadata, f, indent=2)
        
        logger.info(f"✅ Raw data saved: {name} ({len(df)} rows)")
        
        return file_path
    
    def load_raw_data(
        self,
        name: str,
        timestamp: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Load raw data with metadata
        
        Args:
            name: Dataset name
            timestamp: Specific timestamp (latest if None)
            
        Returns:
            Tuple of (DataFrame, metadata)
        """
        
        dataset_dir = self.raw_dir / name
        metadata_path = dataset_dir / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"No data found for: {name}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get latest if no timestamp provided
        if timestamp is None:
            timestamp = sorted(metadata.keys())[-1]
        
        if timestamp not in metadata:
            raise ValueError(f"Timestamp not found: {timestamp}")
        
        file_path = Path(metadata[timestamp]["file_path"])
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data
        df = pd.read_parquet(file_path)
        
        logger.info(f"✅ Raw data loaded: {name} ({len(df)} rows)")
        
        return df, metadata[timestamp]
    
    def list_raw_datasets(self) -> List[str]:
        """List all available raw datasets"""
        return [d.name for d in self.raw_dir.iterdir() if d.is_dir()]
    
    # ============================================================================
    # 🔧 Processed Data Management
    # ============================================================================
    
    def save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        name: str = "default",
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Path]:
        """
        Save processed train/test data
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            name: Dataset name
            feature_names: Feature names
            metadata: Additional metadata
            
        Returns:
            Dictionary of saved file paths
        """
        
        # Create dataset directory
        dataset_dir = self.processed_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save data
        paths = {
            "X_train": dataset_dir / f"X_train_{timestamp}.parquet",
            "X_test": dataset_dir / f"X_test_{timestamp}.parquet",
            "y_train": dataset_dir / f"y_train_{timestamp}.parquet",
            "y_test": dataset_dir / f"y_test_{timestamp}.parquet",
        }
        
        X_train.to_parquet(paths["X_train"], index=False)
        X_test.to_parquet(paths["X_test"], index=False)
        
        # Save targets as DataFrame
        pd.DataFrame(y_train).to_parquet(paths["y_train"], index=False)
        pd.DataFrame(y_test).to_parquet(paths["y_test"], index=False)
        
        # Save feature names
        if feature_names:
            feature_path = dataset_dir / "feature_names.json"
            with open(feature_path, 'w') as f:
                json.dump(feature_names, f, indent=2)
        
        # Save metadata
        metadata_path = dataset_dir / "metadata.json"
        
        metadata_data = {
            "timestamp": timestamp,
            "name": name,
            "X_train_shape": X_train.shape,
            "X_test_shape": X_test.shape,
            "y_train_shape": y_train.shape,
            "y_test_shape": y_test.shape,
            "class_distribution_train": y_train.value_counts().to_dict(),
            "class_distribution_test": y_test.value_counts().to_dict(),
            "feature_names": feature_names or X_train.columns.tolist(),
            "hash": self._calculate_hash(pd.concat([X_train, X_test], axis=0)),
            "metadata": metadata or {},
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_data, f, indent=2)
        
        logger.info(f"✅ Processed data saved: {name} (Train: {X_train.shape}, Test: {X_test.shape})")
        
        return paths
    
    def load_processed_data(
        self,
        name: str = "default",
        timestamp: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
        """
        Load processed train/test data
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, metadata)
        """
        
        dataset_dir = self.processed_dir / name
        metadata_path = dataset_dir / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"No processed data found for: {name}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get latest if no timestamp
        if timestamp is None:
            timestamp = metadata.get("timestamp", 
                                    sorted([f.stem.split('_')[1] for f in dataset_dir.glob("X_train_*.parquet")])[-1])
        
        # Load data
        X_train = pd.read_parquet(dataset_dir / f"X_train_{timestamp}.parquet")
        X_test = pd.read_parquet(dataset_dir / f"X_test_{timestamp}.parquet")
        y_train = pd.read_parquet(dataset_dir / f"y_train_{timestamp}.parquet").iloc[:, 0]
        y_test = pd.read_parquet(dataset_dir / f"y_test_{timestamp}.parquet").iloc[:, 0]
        
        logger.info(f"✅ Processed data loaded: {name} (Train: {X_train.shape}, Test: {X_test.shape})")
        
        return X_train, X_test, y_train, y_test, metadata
    
    # ============================================================================
    # 🏪 Feature Store Management
    # ============================================================================
    
    def save_features(
        self,
        entity_id: str,
        features: Dict[str, Any],
        feature_group: str = "default",
        timestamp: Optional[datetime] = None,
    ) -> Path:
        """
        Save features to feature store
        
        Args:
            entity_id: Entity identifier
            features: Feature dictionary
            feature_group: Feature group name
            timestamp: Feature timestamp
            
        Returns:
            Path to saved features
        """
        
        timestamp = timestamp or datetime.now()
        
        # Create feature group directory
        group_dir = self.feature_store_dir / "features" / feature_group
        group_dir.mkdir(parents=True, exist_ok=True)
        
        # Create entity file
        entity_file = group_dir / f"{entity_id}_{timestamp.strftime('%Y%m%d')}.parquet"
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            "entity_id": entity_id,
            "timestamp": timestamp,
            **features
        }])
        
        df.to_parquet(entity_file, index=False)
        
        # Update metadata
        self._update_feature_registry(entity_id, feature_group, timestamp)
        
        logger.debug(f"💾 Features saved: {entity_id} ({feature_group})")
        
        return entity_file
    
    def load_features(
        self,
        entity_id: str,
        feature_group: str = "default",
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load features from feature store
        
        Args:
            entity_id: Entity identifier
            feature_group: Feature group name
            date: Date to load (latest if None)
            
        Returns:
            Feature dictionary
        """
        
        group_dir = self.feature_store_dir / "features" / feature_group
        
        if not group_dir.exists():
            return {}
        
        # Find entity files
        entity_files = list(group_dir.glob(f"{entity_id}_*.parquet"))
        
        if not entity_files:
            return {}
        
        # Get latest or specific date
        if date:
            target_file = group_dir / f"{entity_id}_{date}.parquet"
            if target_file.exists():
                df = pd.read_parquet(target_file)
            else:
                return {}
        else:
            # Get latest
            entity_files.sort(key=lambda x: x.stem.split('_')[1], reverse=True)
            df = pd.read_parquet(entity_files[0])
        
        # Convert to dict (excluding entity_id and timestamp)
        result = df.iloc[0].to_dict()
        result.pop('entity_id', None)
        result.pop('timestamp', None)
        
        return result
    
    def load_feature_batch(
        self,
        entity_ids: List[str],
        feature_group: str = "default",
    ) -> pd.DataFrame:
        """
        Load features for multiple entities
        
        Args:
            entity_ids: List of entity identifiers
            feature_group: Feature group name
            
        Returns:
            DataFrame with features
        """
        
        dfs = []
        
        for entity_id in entity_ids:
            features = self.load_features(entity_id, feature_group)
            if features:
                dfs.append(pd.DataFrame([{
                    "entity_id": entity_id,
                    **features
                }]))
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        
        return pd.DataFrame()
    
    def _update_feature_registry(self, entity_id: str, feature_group: str, timestamp: datetime):
        """Update feature registry"""
        
        registry_path = self.feature_store_dir / "metadata" / "feature_registry.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {"entities": {}, "groups": {}}
        
        # Update entity
        if entity_id not in registry["entities"]:
            registry["entities"][entity_id] = {"groups": [], "last_updated": timestamp.isoformat()}
        
        if feature_group not in registry["entities"][entity_id]["groups"]:
            registry["entities"][entity_id]["groups"].append(feature_group)
        
        registry["entities"][entity_id]["last_updated"] = timestamp.isoformat()
        
        # Update group
        if feature_group not in registry["groups"]:
            registry["groups"][feature_group] = {"entities": [], "last_updated": timestamp.isoformat()}
        
        if entity_id not in registry["groups"][feature_group]["entities"]:
            registry["groups"][feature_group]["entities"].append(entity_id)
        
        registry["groups"][feature_group]["last_updated"] = timestamp.isoformat()
        
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
    
    # ============================================================================
    # 📊 Reference Data Management
    # ============================================================================
    
    def save_reference_data(
        self,
        df: pd.DataFrame,
        name: str = "reference_data",
        description: str = "",
    ) -> Path:
        """
        Save reference data for drift detection
        
        Args:
            df: Reference DataFrame
            name: Reference name
            description: Description
            
        Returns:
            Path to saved reference data
        """
        
        # Save as CSV and Parquet
        csv_path = self.base_dir / f"{name}.csv"
        parquet_path = self.base_dir / f"{name}.parquet"
        
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        
        # Save metadata
        metadata_path = self.base_dir / f"{name}_metadata.json"
        
        metadata = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "rows": len(df),
            "columns": len(df.columns),
            "columns_list": df.columns.tolist(),
            "hash": self._calculate_hash(df),
            "statistics": df.describe().to_dict(),
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Reference data saved: {name} ({len(df)} rows)")
        
        return csv_path
    
    def load_reference_data(self, name: str = "reference_data") -> Tuple[pd.DataFrame, Dict]:
        """
        Load reference data
        
        Returns:
            Tuple of (DataFrame, metadata)
        """
        
        parquet_path = self.base_dir / f"{name}.parquet"
        metadata_path = self.base_dir / f"{name}_metadata.json"
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Reference data not found: {name}")
        
        df = pd.read_parquet(parquet_path)
        
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        logger.info(f"✅ Reference data loaded: {name} ({len(df)} rows)")
        
        return df, metadata
    
    # ============================================================================
    # 🔄 Version Control (DVC Integration)
    # ============================================================================
    
    def version_data(
        self,
        name: str,
        path: Union[str, Path],
        commit_message: str = "",
    ) -> Dict[str, Any]:
        """
        Version data using DVC
        
        Args:
            name: Data name
            path: Path to data file
            commit_message: Commit message
            
        Returns:
            Version information
        """
        
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        # Create versioned directory
        versioned_path = self.versioned_dir / path.name
        shutil.copy2(path, versioned_path)
        
        # Create DVC file
        dvc_file = self.versioned_dir / f"{path.name}.dvc"
        
        # Calculate hash
        file_hash = self._calculate_file_hash(path)
        
        # Create DVC metadata
        dvc_data = {
            "name": name,
            "path": str(path),
            "versioned_path": str(versioned_path),
            "hash": file_hash,
            "timestamp": datetime.now().isoformat(),
            "commit_message": commit_message,
            "file_size": path.stat().st_size,
        }
        
        with open(dvc_file, 'w') as f:
            json.dump(dvc_data, f, indent=2)
        
        logger.info(f"✅ Data versioned: {name} ({file_hash[:8]})")
        
        return dvc_data
    
    def get_version_history(self) -> List[Dict]:
        """Get version history"""
        
        versions = []
        
        for dvc_file in self.versioned_dir.glob("*.dvc"):
            with open(dvc_file, 'r') as f:
                data = json.load(f)
                versions.append(data)
        
        return sorted(versions, key=lambda x: x["timestamp"], reverse=True)
    
    # ============================================================================
    # 🔧 Utility Methods
    # ============================================================================
    
    def _calculate_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of DataFrame"""
        return hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()
    
    def _calculate_file_hash(self, path: Path) -> str:
        """Calculate hash of file"""
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data quality report
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Data quality report
        """
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": {},
            "duplicates": int(df.duplicated().sum()),
            "column_types": df.dtypes.astype(str).to_dict(),
            "outliers": {},
            "statistics": {},
        }
        
        # Missing values
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                report["missing_values"][col] = {
                    "count": int(missing),
                    "percentage": float(missing / len(df) * 100),
                }
        
        # Outliers (numeric columns)
        for col in df.select_dtypes(include=[np.number]).columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                report["outliers"][col] = int(outliers)
        
        # Statistics
        report["statistics"] = df.describe().to_dict()
        
        return report
    
    def export_data(self, df: pd.DataFrame, path: Union[str, Path], format: str = "csv"):
        """
        Export data to various formats
        
        Args:
            df: DataFrame to export
            path: Export path
            format: Export format (csv, parquet, json, excel)
        """
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "csv":
            df.to_csv(path, index=False)
        elif format == "parquet":
            df.to_parquet(path, index=False)
        elif format == "json":
            df.to_json(path, orient="records", indent=2)
        elif format == "excel":
            df.to_excel(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"✅ Data exported: {path} ({format})")
    
    def import_data(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Import data from various formats
        
        Args:
            path: Import path
            
        Returns:
            DataFrame
        """
        
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            return pd.read_csv(path)
        elif suffix in ['.parquet', '.pq']:
            return pd.read_parquet(path)
        elif suffix == '.json':
            return pd.read_json(path)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def clean_old_data(self, days: int = 30):
        """
        Clean old raw data
        
        Args:
            days: Keep data from last N days
        """
        
        import time
        cutoff = time.time() - (days * 86400)
        cleaned = 0
        
        for dataset_dir in self.raw_dir.iterdir():
            if dataset_dir.is_dir():
                for file in dataset_dir.glob("*.parquet"):
                    if file.stat().st_mtime < cutoff:
                        file.unlink()
                        cleaned += 1
        
        logger.info(f"🧹 Cleaned {cleaned} old data files")
        
        return cleaned


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_data_manager: Optional[DataManager] = None


def get_data_manager() -> DataManager:
    """Get data manager singleton"""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager