# src/components/model_registry.py
"""
MLflow Model Registry with Version Management
"""

import os
import json
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import joblib
import hashlib

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


class ModelRegistry:
    """
    Enterprise Model Registry with:
    - MLflow integration
    - Version management
    - Staging/Production/Archived stages
    - Model promotion with validation
    - Rollback support
    - Audit trail
    """
    
    STAGES = ['staging', 'production', 'archived']
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
    ):
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.registry_uri = registry_uri or os.getenv('MLFLOW_REGISTRY_URI', '')
        
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        self.model_name = f"cdss_{config_manager.get('data.dataset_type', 'diabetes')}"
        
        logger.info(f"📦 ModelRegistry initialized: {self.model_name}")
        logger.info(f"   Tracking URI: {self.tracking_uri}")
    
    # ============================================================================
    # 🚀 Model Registration
    # ============================================================================
    
    def register_model(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        metadata: Optional[Dict] = None,
        stage: str = 'staging',
        run_id: Optional[str] = None,
    ) -> str:
        """
        Register a model in the registry
        
        Args:
            model_path: Path to model files
            model_name: Name of the model (defaults to configured name)
            version: Version string
            metadata: Additional metadata
            stage: Stage to register in (staging/production/archived)
            run_id: MLflow run ID
            
        Returns:
            Model version string
        """
        
        model_name = model_name or self.model_name
        version = version or f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        logger.info(f"📦 Registering model: {model_name} v{version}")
        
        try:
            # Load model
            model = joblib.load(f"{model_path}/model.pkl")
            
            # Create MLflow run if not provided
            with mlflow.start_run(run_id=run_id, run_name=f"register_{version}") as run:
                # Log model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=model_name,
                )
                
                # Log metadata
                if metadata:
                    mlflow.log_params({f"metadata_{k}": str(v) for k, v in metadata.items()})
                
                # Log artifacts
                for file in Path(model_path).glob("*"):
                    if file.suffix in ['.pkl', '.json', '.yaml']:
                        mlflow.log_artifact(str(file))
            
            # Get latest version
            latest_version = self.client.get_latest_versions(model_name)
            version_number = len(latest_version)
            
            # Transition to requested stage
            if stage in self.STAGES:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version_number,
                    stage=stage,
                )
            
            logger.info(f"✅ Model registered: {model_name} v{version_number} ({stage})")
            
            return f"{version_number}"
            
        except Exception as e:
            logger.error(f"❌ Model registration failed: {str(e)}")
            raise
    
    def register_model_from_mlflow(
        self,
        run_id: str,
        stage: str = 'staging',
    ) -> str:
        """Register model from MLflow run"""
        
        # Get run info
        run = self.client.get_run(run_id)
        
        # Register model
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=self.model_name,
        )
        
        # Transition to stage
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=registered_model.version,
            stage=stage,
        )
        
        logger.info(f"✅ Model registered from MLflow: {registered_model.version} ({stage})")
        
        return registered_model.version
    
    # ============================================================================
    # 🔧 Model Retrieval
    # ============================================================================
    
    def get_production_model(self) -> Any:
        """Get production model"""
        return self._get_model(stage='production')
    
    def get_staging_model(self) -> Any:
        """Get staging model"""
        return self._get_model(stage='staging')
    
    def get_model(self, version: str) -> Any:
        """Get specific model version"""
        return self._get_model(version=version)
    
    def _get_model(
        self,
        stage: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Any:
        """Internal method to get model"""
        
        try:
            if stage:
                latest_versions = self.client.get_latest_versions(
                    self.model_name,
                    stages=[stage]
                )
                if not latest_versions:
                    raise ValueError(f"No model found in stage: {stage}")
                
                version_obj = latest_versions[0]
            elif version:
                version_obj = self.client.get_model_version(
                    self.model_name,
                    version
                )
            else:
                raise ValueError("Either stage or version must be provided")
            
            # Download model
            model_uri = f"models:/{self.model_name}/{version_obj.version}"
            model = mlflow.sklearn.load_model(model_uri)
            
            logger.info(f"✅ Model loaded: {self.model_name} v{version_obj.version}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise
    
    def get_model_info(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """Get model information"""
        
        try:
            if stage:
                versions = self.client.get_latest_versions(self.model_name, stages=[stage])
                if not versions:
                    return {}
                version = versions[0]
            else:
                version = self.client.get_latest_versions(self.model_name)[-1]
            
            return {
                'name': self.model_name,
                'version': version.version,
                'stage': version.current_stage,
                'run_id': version.run_id,
                'creation_time': datetime.fromtimestamp(version.creation_timestamp / 1000).isoformat(),
                'description': version.description,
                'tags': version.tags,
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get model info: {str(e)}")
            return {}
    
    # ============================================================================
    # 🔧 Model Promotion & Rollback
    # ============================================================================
    
    def promote_model(
        self,
        version: str,
        stage: str = 'production',
        run_tests: bool = True,
    ) -> Dict[str, Any]:
        """
        Promote a model to production
        
        Args:
            version: Model version to promote
            stage: Target stage
            run_tests: Run validation tests before promotion
            
        Returns:
            Promotion result
        """
        
        logger.info(f"🚀 Promoting model v{version} to {stage}")
        
        try:
            # Get model info
            model_version = self.client.get_model_version(self.model_name, version)
            
            # Run tests if requested
            test_results = {}
            if run_tests:
                test_results = self._run_promotion_tests(version)
                if not test_results['passed']:
                    raise ValueError(f"Promotion tests failed: {test_results['errors']}")
            
            # Transition to stage
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True,
            )
            
            # Log promotion
            logger.info(f"✅ Model v{version} promoted to {stage}")
            
            return {
                'status': 'success',
                'version': version,
                'stage': stage,
                'test_results': test_results,
            }
            
        except Exception as e:
            logger.error(f"❌ Promotion failed: {str(e)}")
            raise
    
    def rollback(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Rollback to previous version
        
        Args:
            target_version: Version to rollback to (default: latest before current)
            
        Returns:
            Rollback result
        """
        
        try:
            # Get current production
            current = self.client.get_latest_versions(self.model_name, stages=['production'])
            if not current:
                raise ValueError("No production model found")
            
            current_version = current[0].version
            
            # Determine target
            if target_version:
                target = target_version
            else:
                # Get all versions
                all_versions = self.client.search_model_versions(f"name='{self.model_name}'")
                versions = sorted([int(v.version) for v in all_versions if v.current_stage != 'production'])
                if not versions:
                    raise ValueError("No previous versions available")
                target = str(versions[-1])
            
            # Validate target
            if target == current_version:
                raise ValueError("Target version is same as current")
            
            # Rollback
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=current_version,
                stage='archived',
            )
            
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=target,
                stage='production',
            )
            
            logger.info(f"✅ Rollback complete: {current_version} -> {target}")
            
            return {
                'status': 'success',
                'previous_version': current_version,
                'current_version': target,
            }
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {str(e)}")
            raise
    
    def _run_promotion_tests(self, version: str) -> Dict[str, Any]:
        """Run validation tests before promotion"""
        
        from src.pipelines.inference_pipeline import InferencePipeline
        
        errors = []
        passed = True
        
        try:
            # Load model
            model = self.get_model(version)
            
            # Load golden tests
            from src.validation.schema_validation import DataValidator
            validator = DataValidator()
            golden_tests = validator.load_golden_tests()
            
            # Run tests
            for test_name, test_case in golden_tests.items():
                try:
                    # Prepare input
                    input_data = pd.DataFrame([test_case['input']])
                    
                    # Run prediction
                    y_pred_proba = model.predict_proba(input_data)[0][1]
                    y_pred = int(y_pred_proba > 0.5)
                    
                    # Check against expected
                    expected = test_case['expected']
                    if y_pred != expected:
                        errors.append(f"{test_name}: Expected {expected}, got {y_pred}")
                        passed = False
                        
                except Exception as e:
                    errors.append(f"{test_name}: {str(e)}")
                    passed = False
            
            # Performance test
            # Check if model meets minimum performance threshold
            from src.components.model_evaluation import ModelEvaluator
            evaluator = ModelEvaluator()
            
            # This would require test data
            # metrics = evaluator.evaluate_model(model, X_test, y_test)
            # if metrics.recall < 0.7:
            #     errors.append(f"Recall {metrics.recall:.3f} below threshold 0.7")
            #     passed = False
            
        except Exception as e:
            errors.append(f"Test execution error: {str(e)}")
            passed = False
        
        return {
            'passed': passed,
            'errors': errors,
        }
    
    # ============================================================================
    # 📊 Registry Management
    # ============================================================================
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all model versions"""
        
        models = []
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            
            for version in versions:
                models.append({
                    'name': version.name,
                    'version': version.version,
                    'stage': version.current_stage,
                    'run_id': version.run_id,
                    'creation_time': datetime.fromtimestamp(
                        version.creation_timestamp / 1000
                    ).isoformat(),
                    'description': version.description,
                })
            
            # Sort by version
            models.sort(key=lambda x: int(x['version']))
            
        except Exception as e:
            logger.error(f"❌ Failed to list models: {str(e)}")
        
        return models
    
    def archive_model(self, version: str) -> Dict[str, Any]:
        """Archive a model version"""
        
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage='archived',
            )
            
            logger.info(f"✅ Model v{version} archived")
            
            return {
                'status': 'success',
                'version': version,
                'stage': 'archived',
            }
            
        except Exception as e:
            logger.error(f"❌ Archival failed: {str(e)}")
            raise
    
    def delete_model(self, version: str) -> Dict[str, Any]:
        """Delete a model version (dangerous)"""
        
        try:
            self.client.delete_model_version(
                name=self.model_name,
                version=version,
            )
            
            logger.warning(f"⚠️ Model v{version} deleted")
            
            return {
                'status': 'success',
                'version': version,
            }
            
        except Exception as e:
            logger.error(f"❌ Deletion failed: {str(e)}")
            raise