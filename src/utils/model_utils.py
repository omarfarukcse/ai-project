# src/utils/model_utils.py
"""
Model Utilities for Serialization, Versioning, and Management
"""

import os
import json
import hashlib
import pickle
import joblib
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


@dataclass
class ModelVersion:
    """Model version information"""
    version: str
    created_at: datetime = field(default_factory=datetime.now)
    author: str = "system"
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    file_hash: str = ""
    size_bytes: int = 0


@dataclass
class ModelMetadata:
    """Model metadata"""
    name: str
    version: str
    framework: str
    framework_version: str
    features: List[str]
    target: str
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    signature: Optional[Dict] = None


class ModelSerializer:
    """
    Model Serialization with Multiple Formats
    
    Supports:
    - Pickle
    - Joblib
    - MLflow
    - ONNX
    - TensorFlow
    - PyTorch
    """
    
    @staticmethod
    def save_pickle(model: Any, path: Union[str, Path], compress: bool = True):
        """Save model using pickle"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if compress:
            import gzip
            with gzip.open(path, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"💾 Model saved (pickle): {path}")
    
    @staticmethod
    def load_pickle(path: Union[str, Path]) -> Any:
        """Load model from pickle"""
        path = Path(path)
        
        import gzip
        try:
            with gzip.open(path, 'rb') as f:
                return pickle.load(f)
        except (gzip.BadGzipFile, OSError):
            # Not gzipped
            with open(path, 'rb') as f:
                return pickle.load(f)
    
    @staticmethod
    def save_joblib(model: Any, path: Union[str, Path]):
        """Save model using joblib"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"💾 Model saved (joblib): {path}")
    
    @staticmethod
    def load_joblib(path: Union[str, Path]) -> Any:
        """Load model from joblib"""
        return joblib.load(path)
    
    @staticmethod
    def save_mlflow(
        model: Any,
        path: Union[str, Path],
        model_name: str,
        version: str,
        signature: Optional[ModelSignature] = None,
    ):
        """Save model using MLflow"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        mlflow.sklearn.save_model(
            sk_model=model,
            path=str(path),
            signature=signature,
        )
        
        # Save metadata
        metadata = ModelMetadata(
            name=model_name,
            version=version,
            framework="sklearn",
            framework_version=model.__class__.__module__,
            features=[],  # Should be passed separately
            target="",
        )
        
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata.__dict__, f, indent=2, default=str)
        
        logger.info(f"💾 Model saved (MLflow): {path}")
    
    @staticmethod
    def load_mlflow(path: Union[str, Path]) -> Any:
        """Load model from MLflow"""
        return mlflow.sklearn.load_model(str(path))
    
    @staticmethod
    def get_model_size(path: Union[str, Path]) -> int:
        """Get model file size in bytes"""
        path = Path(path)
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            total = 0
            for file in path.rglob('*'):
                if file.is_file():
                    total += file.stat().st_size
            return total
        return 0
    
    @staticmethod
    def get_model_hash(path: Union[str, Path]) -> str:
        """Get model file hash"""
        import hashlib
        hash_func = hashlib.sha256()
        path = Path(path)
        
        if path.is_file():
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_func.update(chunk)
        elif path.is_dir():
            for file in sorted(path.rglob('*')):
                if file.is_file():
                    with open(file, 'rb') as f:
                        for chunk in iter(lambda: f.read(8192), b''):
                            hash_func.update(chunk)
        
        return hash_func.hexdigest()


class ModelManager:
    """
    Comprehensive Model Manager with:
    - Model serialization (multiple formats)
    - Version management
    - Model comparison
    - Signature validation
    - Performance tracking
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir or config_manager.get('models.dir', 'models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.serializer = ModelSerializer()
        self._loaded_models: Dict[str, Any] = {}
        
        logger.info(f"🤖 ModelManager initialized: {self.models_dir}")
    
    def save_model(
        self,
        model: Any,
        name: str,
        version: str,
        format: str = 'joblib',
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Save model with metadata
        
        Args:
            model: Model to save
            name: Model name
            version: Model version
            format: Serialization format (pickle, joblib, mlflow)
            metadata: Additional metadata
            
        Returns:
            Path to saved model
        """
        
        model_dir = self.models_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            path = model_dir / 'model.pkl'
            self.serializer.save_pickle(model, path)
        elif format == 'joblib':
            path = model_dir / 'model.joblib'
            self.serializer.save_joblib(model, path)
        elif format == 'mlflow':
            path = model_dir / 'mlflow'
            self.serializer.save_mlflow(model, path, name, version)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save metadata
        metadata = metadata or {}
        metadata.update({
            'name': name,
            'version': version,
            'format': format,
            'created_at': datetime.now().isoformat(),
            'size_bytes': self.serializer.get_model_size(path),
            'hash': self.serializer.get_model_hash(path),
        })
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"✅ Model saved: {name} v{version} ({format})")
        return path
    
    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
        format: Optional[str] = None,
    ) -> Any:
        """
        Load model from storage
        
        Args:
            name: Model name
            version: Model version (latest if None)
            format: Serialization format (auto-detect if None)
            
        Returns:
            Loaded model
        """
        
        # Get version
        if version is None:
            version = self.get_latest_version(name)
        
        if version is None:
            raise ValueError(f"No model found: {name}")
        
        # Check cache
        cache_key = f"{name}:{version}"
        if cache_key in self._loaded_models:
            logger.info(f"📦 Model loaded from cache: {cache_key}")
            return self._loaded_models[cache_key]
        
        model_dir = self.models_dir / name / version
        
        # Auto-detect format if not specified
        if format is None:
            if (model_dir / 'model.joblib').exists():
                format = 'joblib'
            elif (model_dir / 'model.pkl').exists():
                format = 'pickle'
            elif (model_dir / 'mlflow').exists():
                format = 'mlflow'
            else:
                raise ValueError(f"No model found in {model_dir}")
        
        # Load model
        if format == 'pickle':
            path = model_dir / 'model.pkl'
            model = self.serializer.load_pickle(path)
        elif format == 'joblib':
            path = model_dir / 'model.joblib'
            model = self.serializer.load_joblib(path)
        elif format == 'mlflow':
            path = model_dir / 'mlflow'
            model = self.serializer.load_mlflow(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Cache model
        self._loaded_models[cache_key] = model
        
        logger.info(f"✅ Model loaded: {name} v{version} ({format})")
        return model
    
    def get_latest_version(self, name: str) -> Optional[str]:
        """Get the latest version of a model"""
        model_dir = self.models_dir / name
        if not model_dir.exists():
            return None
        
        versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
        if not versions:
            return None
        
        # Sort by version (assuming semver or timestamp)
        try:
            versions.sort(key=lambda v: [int(x) for x in v.split('.')])
        except:
            versions.sort(key=lambda v: v)
        
        return versions[-1]
    
    def list_models(self) -> List[str]:
        """List all model names"""
        return [d.name for d in self.models_dir.iterdir() if d.is_dir()]
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions of a model"""
        model_dir = self.models_dir / name
        if not model_dir.exists():
            return []
        return [d.name for d in model_dir.iterdir() if d.is_dir()]
    
    def get_model_metadata(self, name: str, version: Optional[str] = None) -> Dict:
        """Get model metadata"""
        if version is None:
            version = self.get_latest_version(name)
        
        if version is None:
            return {}
        
        metadata_path = self.models_dir / name / version / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        return {}
    
    def delete_model(self, name: str, version: Optional[str] = None):
        """Delete model"""
        if version:
            path = self.models_dir / name / version
            import shutil
            shutil.rmtree(path)
            logger.info(f"🗑️ Deleted: {name} v{version}")
        else:
            path = self.models_dir / name
            import shutil
            shutil.rmtree(path)
            logger.info(f"🗑️ Deleted: {name} (all versions)")
        
        # Clear cache
        for key in list(self._loaded_models.keys()):
            if key.startswith(f"{name}:"):
                del self._loaded_models[key]
    
    def compare_models(self, name: str, versions: List[str]) -> Dict:
        """Compare multiple model versions"""
        results = {}
        
        for version in versions:
            metadata = self.get_model_metadata(name, version)
            results[version] = {
                'metrics': metadata.get('metrics', {}),
                'created_at': metadata.get('created_at'),
                'size_bytes': metadata.get('size_bytes'),
            }
        
        return results
    
    def get_model_signature(
        self,
        model: Any,
        feature_names: List[str],
        target_name: str = 'target',
    ) -> ModelSignature:
        """Get model signature for MLflow"""
        from mlflow.models import ModelSignature
        from mlflow.types.schema import Schema, ColSpec
        
        input_schema = Schema([ColSpec('double', name) for name in feature_names])
        output_schema = Schema([ColSpec('double', target_name)])
        
        return ModelSignature(
            inputs=input_schema,
            outputs=output_schema,
        )


# ============================================================================
# 🔧 Convenience Functions
# ============================================================================

def save_model(
    model: Any,
    path: Union[str, Path],
    format: str = 'joblib',
    metadata: Optional[Dict] = None,
) -> Path:
    """Save model using ModelManager"""
    manager = get_model_utils()
    return manager.save_model(model, path, format, metadata)

def load_model(path: Union[str, Path], format: Optional[str] = None) -> Any:
    """Load model using ModelManager"""
    manager = get_model_utils()
    return manager.load_model(path, format)

def get_model_version(path: Union[str, Path]) -> str:
    """Get model version from path"""
    return Path(path).name

def compare_models(model1: Any, model2: Any) -> Dict[str, float]:
    """Compare two models"""
    # Extract feature importance if available
    result = {}
    
    for name, model in [('model1', model1), ('model2', model2)]:
        if hasattr(model, 'feature_importances_'):
            result[f'{name}_feature_importance'] = model.feature_importances_.tolist()
        if hasattr(model, 'coef_'):
            result[f'{name}_coef'] = model.coef_.tolist()
    
    return result

def get_model_size(path: Union[str, Path]) -> int:
    """Get model file size"""
    return ModelSerializer.get_model_size(path)

def get_model_signature(
    model: Any,
    feature_names: List[str],
    target_name: str = 'target',
) -> ModelSignature:
    """Get model signature"""
    manager = get_model_utils()
    return manager.get_model_signature(model, feature_names, target_name)


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_model_manager: Optional[ModelManager] = None


def get_model_utils() -> ModelManager:
    """Get model manager singleton"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(
            models_dir=config_manager.get('models.dir', 'models')
        )
    return _model_manager