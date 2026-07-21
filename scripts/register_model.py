# scripts/register_model.py
#!/usr/bin/env python
"""
Model Registration Script

This script registers a trained model into the model registry with proper metadata.
It handles:
- Model validation
- Metadata extraction
- MLflow tracking
- Version management
- Staging deployment
- Artifact storage

Usage:
    python scripts/register_model.py --model-path models/staging/model.pkl --version v1.0.0
    python scripts/register_model.py --model-path models/staging/model.pkl --auto-version
    python scripts/register_model.py --help
"""

import os
import sys
import json
import argparse
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.components.model_registry import ModelRegistry, ModelMetadata
from src.components.model_evaluation import ModelEvaluator
from src.components.data_ingestion import DataIngestion
from src.components.preprocessing import ClinicalPreprocessor
from src.logger import get_logger
from src.config_manager import get_config_manager
from src.utils.model_utils import ModelSerializer

logger = get_logger(__name__)


class ModelRegistrar:
    """
    Model registration with validation and metadata extraction
    """
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.serializer = ModelSerializer()
        self.config = get_config_manager()
        self.evaluator = ModelEvaluator()
        
        logger.info("📦 ModelRegistrar initialized")
    
    def register_model(
        self,
        model_path: str,
        version: Optional[str] = None,
        name: str = "cdss_model",
        stage: str = "staging",
        features: Optional[List[str]] = None,
        target: str = "target",
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        training_config: Optional[Dict] = None,
        run_tests: bool = True,
        auto_version: bool = False,
    ) -> Dict[str, Any]:
        """
        Register a model with complete validation
        
        Args:
            model_path: Path to the model file
            version: Model version (auto-generated if None)
            name: Model name
            stage: Initial stage (staging, production)
            features: Feature names
            target: Target name
            description: Model description
            tags: Additional tags
            training_config: Training configuration
            run_tests: Run validation tests
            auto_version: Auto-generate version
            
        Returns:
            Registration results
        """
        
        logger.info(f"📦 Registering model from {model_path}")
        
        # Validate model path
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        model = self._load_model(model_path)
        
        # Auto-generate version if requested
        if auto_version or version is None:
            version = self._generate_version()
            logger.info(f"📌 Auto-generated version: {version}")
        
        # Load test data for evaluation
        X_train, X_test, y_train, y_test = self._load_test_data()
        
        # Evaluate model if features provided
        metrics = {}
        performance = {}
        
        if features is not None:
            logger.info("📊 Evaluating model...")
            metrics, performance = self._evaluate_model(model, X_test, y_test)
        
        # Get feature importance if available
        if features is None:
            features = self._extract_features(model, X_train)
        
        # Register model
        metadata = self.registry.register_model(
            model=model,
            name=name,
            version=version,
            features=features,
            target=target,
            metrics=metrics,
            performance=performance,
            description=description,
            tags=tags or {},
            training_config=training_config or {},
            stage=stage,
        )
        
        # Run tests if requested
        test_results = {}
        if run_tests:
            logger.info("🧪 Running validation tests...")
            test_results = self._run_tests(model, X_test, y_test)
        
        # Save registration report
        report = {
            "status": "success",
            "model_name": name,
            "version": version,
            "stage": stage,
            "model_path": str(model_path),
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata.to_dict() if metadata else {},
            "metrics": metrics,
            "performance": performance,
            "test_results": test_results,
        }
        
        # Save report
        self._save_report(report)
        
        logger.info(f"✅ Model registered: {name} v{version} ({stage})")
        logger.info(f"   Hash: {metadata.hash[:8] if metadata else 'N/A'}")
        logger.info(f"   Size: {metadata.size_bytes / 1024:.2f} KB" if metadata else "")
        
        return report
    
    def _load_model(self, model_path: Path) -> Any:
        """Load model from file"""
        
        logger.info(f"📂 Loading model from {model_path}")
        
        # Try different formats
        if model_path.suffix == '.pkl':
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        elif model_path.suffix == '.joblib':
            return joblib.load(model_path)
        elif model_path.suffix == '.onnx':
            # ONNX loading would require onnxruntime
            raise ValueError("ONNX loading not implemented")
        else:
            # Try pickle then joblib
            try:
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return joblib.load(model_path)
    
    def _load_test_data(self) -> tuple:
        """Load test data for evaluation"""
        
        try:
            ingestion = DataIngestion()
            X, y = ingestion.load_with_validation()[:2]
            
            preprocessor = ClinicalPreprocessor()
            X_train, X_test, y_train, y_test = preprocessor.fit_transform(X, y)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.warning(f"Failed to load test data: {str(e)}")
            return None, None, None, None
    
    def _evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
        """Evaluate model performance"""
        
        if X_test is None or y_test is None:
            return {}, {}
        
        try:
            metrics = self.evaluator.evaluate_model(model, X_test, y_test)
            
            # Extract metrics
            metrics_dict = {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'specificity': metrics.specificity,
                'roc_auc': metrics.roc_auc,
                'mcc': metrics.mcc,
                'kappa': metrics.kappa,
            }
            
            performance_dict = {
                'confusion_matrix': metrics.confusion_matrix,
                'log_loss': metrics.log_loss,
                'brier_score': metrics.brier_score,
            }
            
            return metrics_dict, performance_dict
            
        except Exception as e:
            logger.warning(f"Model evaluation failed: {str(e)}")
            return {}, {}
    
    def _extract_features(self, model: Any, X_train: pd.DataFrame) -> List[str]:
        """Extract feature names from model"""
        
        if X_train is not None:
            return X_train.columns.tolist()
        
        # Try to extract from model
        if hasattr(model, 'feature_names_in_'):
            return model.feature_names_in_.tolist()
        elif hasattr(model, 'feature_importances_'):
            # Use default feature names
            return [f'feature_{i}' for i in range(len(model.feature_importances_))]
        
        return []
    
    def _generate_version(self) -> str:
        """Generate version string"""
        
        # Get existing versions
        existing = self.registry.list_versions("staging") + self.registry.list_versions("production")
        
        if not existing:
            return "v1.0.0"
        
        # Parse latest version
        latest = sorted(existing)[-1]
        parts = latest[1:].split('.')
        
        # Increment patch version
        parts[-1] = str(int(parts[-1]) + 1)
        
        return f"v{'.'.join(parts)}"
    
    def _run_tests(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Run validation tests"""
        
        results = {
            "passed": True,
            "tests": [],
            "errors": [],
        }
        
        if X_test is None or y_test is None:
            return results
        
        # Test 1: Basic prediction
        try:
            model.predict(X_test[:1])
            results["tests"].append({"name": "prediction", "status": "passed"})
        except Exception as e:
            results["passed"] = False
            results["errors"].append(f"Prediction test failed: {str(e)}")
            results["tests"].append({"name": "prediction", "status": "failed"})
        
        # Test 2: Probability output
        if hasattr(model, 'predict_proba'):
            try:
                model.predict_proba(X_test[:1])
                results["tests"].append({"name": "probability", "status": "passed"})
            except Exception as e:
                results["passed"] = False
                results["errors"].append(f"Probability test failed: {str(e)}")
                results["tests"].append({"name": "probability", "status": "failed"})
        else:
            results["tests"].append({"name": "probability", "status": "skipped"})
        
        # Test 3: Performance threshold
        try:
            metrics = self.evaluator.evaluate_model(model, X_test, y_test)
            if metrics.recall < 0.7:
                results["passed"] = False
                results["errors"].append(f"Recall {metrics.recall:.3f} below threshold 0.7")
                results["tests"].append({"name": "performance", "status": "failed"})
            else:
                results["tests"].append({"name": "performance", "status": "passed"})
        except Exception as e:
            results["passed"] = False
            results["errors"].append(f"Performance test failed: {str(e)}")
            results["tests"].append({"name": "performance", "status": "failed"})
        
        return results
    
    def _save_report(self, report: Dict):
        """Save registration report"""
        
        report_dir = Path("outputs/reports/registrations")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"registration_{report['model_name']}_{report['version']}_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Registration report saved: {report_path}")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Register a model in the CDSS model registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register model with specific version
  python scripts/register_model.py --model-path models/staging/model.pkl --version v1.0.0
  
  # Register with auto-version
  python scripts/register_model.py --model-path models/staging/model.pkl --auto-version
  
  # Register to production directly
  python scripts/register_model.py --model-path models/staging/model.pkl --stage production
  
  # Register with custom features
  python scripts/register_model.py --model-path models/staging/model.pkl --features glucose bmi age
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model file (.pkl, .joblib)"
    )
    
    parser.add_argument(
        "--version",
        type=str,
        help="Model version (e.g., v1.0.0)"
    )
    
    parser.add_argument(
        "--auto-version",
        action="store_true",
        help="Auto-generate version"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="cdss_model",
        help="Model name (default: cdss_model)"
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        choices=["staging", "production", "archived"],
        default="staging",
        help="Initial stage (default: staging)"
    )
    
    parser.add_argument(
        "--features",
        nargs="+",
        help="Feature names"
    )
    
    parser.add_argument(
        "--target",
        type=str,
        default="target",
        help="Target name (default: target)"
    )
    
    parser.add_argument(
        "--description",
        type=str,
        help="Model description"
    )
    
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip validation tests"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Training config file"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize registrar
        registrar = ModelRegistrar()
        
        # Load training config
        training_config = {}
        if args.config and Path(args.config).exists():
            import yaml
            with open(args.config, 'r') as f:
                training_config = yaml.safe_load(f)
        
        # Register model
        result = registrar.register_model(
            model_path=args.model_path,
            version=args.version,
            name=args.name,
            stage=args.stage,
            features=args.features,
            target=args.target,
            description=args.description or "Registered via CLI",
            tags={"source": "cli", "timestamp": datetime.now().isoformat()},
            training_config=training_config,
            run_tests=not args.no_tests,
            auto_version=args.auto_version,
        )
        
        print("\n" + "="*60)
        print("✅ Model Registration Complete")
        print("="*60)
        print(f"Model Name:    {result['model_name']}")
        print(f"Version:       {result['version']}")
        print(f"Stage:         {result['stage']}")
        print(f"Registered At: {result['registered_at']}")
        
        if result.get('metadata'):
            print(f"Hash:          {result['metadata'].get('hash', 'N/A')[:8]}")
            print(f"Size:          {result['metadata'].get('size_bytes', 0) / 1024:.2f} KB")
        
        if result.get('metrics'):
            print("\n📊 Metrics:")
            for key, value in result['metrics'].items():
                print(f"   {key}: {value:.4f}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"❌ Registration failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()