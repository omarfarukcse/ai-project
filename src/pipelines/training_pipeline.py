# src/pipelines/training_pipeline.py
"""
Enterprise Training Pipeline with Governance and Automation

Features:
- End-to-end automated training
- MLflow experiment tracking
- Model versioning with registry
- Automated validation with golden tests
- Performance benchmarking
- Model promotion criteria
- Bias detection and mitigation
- Resource optimization
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from contextlib import contextmanager

import mlflow
import mlflow.sklearn
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from src.components.data_ingestion import DataIngestion
from src.components.preprocessing import ClinicalPreprocessor
from src.components.model_training import ClinicalModelTrainer
from src.components.model_evaluation import ModelEvaluator
from src.components.model_calibration import ModelCalibrator
from src.components.model_registry import ModelRegistry
from src.components.explainability import ClinicalSHAPExplainer
from src.components.reporting import ClinicalReporter
from src.monitoring.drift_detection import DriftDetector
from src.monitoring.bias_monitor import BiasMonitor
from src.validation.schema_validation import DataValidator
from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


@dataclass
class TrainingContext:
    """Training context with all metadata"""
    run_id: str
    start_time: float
    experiment_name: str
    dataset_version: str
    model_version: str
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    status: str = "running"
    errors: List[str] = field(default_factory=list)


class TrainingPipeline:
    """
    Complete training pipeline with governance and automation
    
    Pipeline Steps:
    1. Data Ingestion & Validation
    2. Data Preprocessing
    3. Feature Engineering
    4. Model Training (Multiple models)
    5. Model Evaluation
    6. Model Calibration
    7. Model Selection
    8. Explainability Generation
    9. Model Registry
    10. Report Generation
    11. Bias Auditing
    12. Golden Test Validation
    """
    
    def __init__(
        self,
        experiment_name: str = "cdss_training",
        config_path: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.config = config_manager
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.context = TrainingContext(
            run_id=self.run_id,
            start_time=time.time(),
            experiment_name=experiment_name,
            dataset_version="latest",
            model_version="v1.0.0",
        )
        
        # Components
        self.data_ingestion = None
        self.preprocessor = None
        self.trainer = None
        self.evaluator = None
        self.calibrator = None
        self.registry = None
        self.explainer = None
        self.reporter = None
        self.drift_detector = None
        self.bias_monitor = None
        self.validator = None
        
        # Results
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.evaluation_results = {}
        self.validation_results = {}
        
        # MLflow setup
        self._setup_mlflow()
        
        logger.info(f"🏗️ Training Pipeline initialized: {self.run_id}")
    
    def _setup_mlflow(self):
        """Configure MLflow tracking"""
        mlflow.set_experiment(self.experiment_name)
        
        # Set tracking URI from config
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        logger.info(f"📊 MLflow tracking URI: {tracking_uri}")
    
    @contextmanager
    def _mlflow_run(self):
        """Context manager for MLflow run"""
        with mlflow.start_run(run_name=self.run_id) as run:
            self.context.run_id = run.info.run_id
            logger.info(f"📊 MLflow run: {self.context.run_id}")
            yield run
    
    def _log_artifacts(self, artifacts: Dict[str, str]):
        """Log artifacts to MLflow"""
        for name, path in artifacts.items():
            if Path(path).exists():
                mlflow.log_artifact(path, artifact_path=name)
                logger.info(f"📎 Logged artifact: {name} -> {path}")
    
    def _log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow"""
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, value, step=step)
    
    def _log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        for name, value in params.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(name, value)
            elif isinstance(value, (list, tuple)):
                mlflow.log_param(name, str(value))
    
    # ============================================================================
    # 🚀 Pipeline Steps
    # ============================================================================
    
    def step1_ingest_data(self) -> 'TrainingPipeline':
        """
        Step 1: Data Ingestion with Validation
        
        - Load data from source
        - Validate schema
        - Check data quality
        - Version tracking
        """
        logger.info("📊 Step 1: Data Ingestion")
        
        with self._mlflow_run():
            try:
                # Initialize data ingestion
                self.data_ingestion = DataIngestion(
                    dataset_type=self.config.get('data.dataset_type', 'diabetes')
                )
                
                # Load and validate
                self.X, self.y, self.validation_report = self.data_ingestion.load_with_validation()
                
                # Log validation report
                self._log_params({
                    "dataset_type": self.config.get('data.dataset_type'),
                    "n_samples": len(self.X),
                    "n_features": len(self.X.columns),
                    "class_balance": self.y.value_counts().to_dict(),
                    "missing_values": self.X.isnull().sum().sum(),
                })
                
                # Save validation report
                report_path = f"outputs/reports/validation_report_{self.run_id}.json"
                Path(report_path).parent.mkdir(parents=True, exist_ok=True)
                with open(report_path, 'w') as f:
                    json.dump(self.validation_report, f, indent=2)
                self._log_artifacts({"validation_report": report_path})
                
                logger.info("✅ Data ingestion complete")
                
            except Exception as e:
                logger.error(f"❌ Data ingestion failed: {str(e)}")
                self.context.errors.append(f"Data ingestion: {str(e)}")
                raise
        
        return self
    
    def step2_preprocess_data(self) -> 'TrainingPipeline':
        """
        Step 2: Data Preprocessing
        
        - Missing value imputation
        - Feature scaling
        - Class balancing (SMOTE)
        - Train/Test split
        - Feature engineering
        """
        logger.info("🔧 Step 2: Data Preprocessing")
        
        with self._mlflow_run():
            try:
                # Initialize preprocessor
                self.preprocessor = ClinicalPreprocessor(
                    config=self.config.get('preprocessing', {})
                )
                
                # Run preprocessing
                self.X_train, self.X_test, self.y_train, self.y_test = \
                    self.preprocessor.fit_transform(self.X, self.y)
                
                # Log preprocessing details
                self._log_params({
                    "train_size": len(self.X_train),
                    "test_size": len(self.X_test),
                    "train_class_dist": self.y_train.value_counts().to_dict(),
                    "test_class_dist": self.y_test.value_counts().to_dict(),
                    "imputation_strategy": self.config.get('preprocessing.imputation_strategy', 'median'),
                    "scaling_method": self.config.get('preprocessing.scaling_method', 'standard'),
                    "balancing_method": self.config.get('preprocessing.balancing_method', 'smote'),
                })
                
                # Save preprocessed data
                data_path = f"data/processed/train_test_{self.run_id}.parquet"
                Path(data_path).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({
                    'X_train': self.X_train.to_dict(),
                    'X_test': self.X_test.to_dict(),
                    'y_train': self.y_train.to_dict(),
                    'y_test': self.y_test.to_dict(),
                }).to_parquet(data_path)
                self._log_artifacts({"preprocessed_data": data_path})
                
                logger.info("✅ Preprocessing complete")
                logger.info(f"   Train: {len(self.X_train)} samples")
                logger.info(f"   Test: {len(self.X_test)} samples")
                
            except Exception as e:
                logger.error(f"❌ Preprocessing failed: {str(e)}")
                self.context.errors.append(f"Preprocessing: {str(e)}")
                raise
        
        return self
    
    def step3_train_models(self) -> 'TrainingPipeline':
        """
        Step 3: Model Training
        
        - Train multiple models
        - Hyperparameter optimization
        - Cross-validation
        - Model versioning
        """
        logger.info("🤖 Step 3: Model Training")
        
        with self._mlflow_run():
            try:
                # Initialize trainer
                self.trainer = ClinicalModelTrainer(
                    model_names=self.config.get('models.models', ['logistic_regression', 'random_forest', 'xgboost']),
                    search_type=self.config.get('models.search_type', 'grid'),
                    cv_folds=self.config.get('models.cv_folds', 5),
                )
                self.trainer.create_models()
                
                # Train models
                self.trained_models = self.trainer.train_with_optimization(
                    self.X_train, self.y_train
                )
                
                # Log each model's parameters and metrics
                for model_name, model_data in self.trained_models.items():
                    self._log_params({
                        f"{model_name}_best_params": model_data['best_params'],
                        f"{model_name}_cv_score": model_data['cv_recall_mean'],
                    })
                
                logger.info("✅ Model training complete")
                logger.info(f"   Models trained: {list(self.trained_models.keys())}")
                
            except Exception as e:
                logger.error(f"❌ Model training failed: {str(e)}")
                self.context.errors.append(f"Training: {str(e)}")
                raise
        
        return self
    
    def step4_evaluate_models(self) -> 'TrainingPipeline':
        """
        Step 4: Model Evaluation
        
        - Comprehensive metrics
        - Confusion matrices
        - ROC/PR curves
        - Clinical metrics
        - Model comparison
        """
        logger.info("📊 Step 4: Model Evaluation")
        
        with self._mlflow_run():
            try:
                # Initialize evaluator
                self.evaluator = ModelEvaluator()
                
                # Evaluate all models
                self.evaluation_results = self.evaluator.evaluate_models(
                    self.trained_models,
                    self.X_test,
                    self.y_test
                )
                
                # Select best model (based on recall)
                self.best_model, self.best_model_name = self.trainer.select_best_model(
                    self.trained_models,
                    self.X_test,
                    self.y_test,
                    metric='recall'  # Clinical priority
                )
                
                # Log metrics
                for model_name, metrics in self.evaluation_results.items():
                    self._log_metrics({
                        f"{model_name}_accuracy": metrics['accuracy'],
                        f"{model_name}_precision": metrics['precision'],
                        f"{model_name}_recall": metrics['recall'],
                        f"{model_name}_specificity": metrics['specificity'],
                        f"{model_name}_f1": metrics['f1_score'],
                        f"{model_name}_roc_auc": metrics['roc_auc'],
                    })
                
                # Save evaluation results
                eval_path = f"outputs/reports/evaluation_{self.run_id}.json"
                with open(eval_path, 'w') as f:
                    json.dump(self.evaluation_results, f, indent=2)
                self._log_artifacts({"evaluation_report": eval_path})
                
                logger.info("✅ Model evaluation complete")
                logger.info(f"🏆 Best model: {self.best_model_name}")
                logger.info(f"   Recall: {self.evaluation_results[self.best_model_name]['recall']:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Model evaluation failed: {str(e)}")
                self.context.errors.append(f"Evaluation: {str(e)}")
                raise
        
        return self
    
    def step5_calibrate_model(self) -> 'TrainingPipeline':
        """
        Step 5: Model Calibration
        
        - Platt scaling
        - Isotonic regression
        - Calibration curves
        """
        logger.info("🎯 Step 5: Model Calibration")
        
        with self._mlflow_run():
            try:
                # Initialize calibrator
                self.calibrator = ModelCalibrator(
                    method=self.config.get('models.calibration_method', 'platt')
                )
                
                # Calibrate best model
                self.calibrated_model = self.calibrator.calibrate(
                    self.best_model,
                    self.X_train,
                    self.y_train,
                    self.X_test,
                    self.y_test
                )
                
                # Log calibration metrics
                calibration_metrics = self.calibrator.get_metrics()
                self._log_metrics(calibration_metrics)
                
                # Save calibration plot
                cal_plot = "outputs/figures/calibration_curve.png"
                self.calibrator.plot_calibration_curve(save_path=cal_plot)
                self._log_artifacts({"calibration_curve": cal_plot})
                
                logger.info("✅ Model calibration complete")
                
            except Exception as e:
                logger.error(f"❌ Model calibration failed: {str(e)}")
                self.context.errors.append(f"Calibration: {str(e)}")
                raise
        
        return self
    
    def step6_generate_explanations(self) -> 'TrainingPipeline':
        """
        Step 6: Generate SHAP Explanations
        
        - Global explanations
        - Local explanations
        - Feature importance
        - SHAP plots
        """
        logger.info("🧠 Step 6: Generating Explanations")
        
        with self._mlflow_run():
            try:
                # Initialize explainer
                self.explainer = ClinicalSHAPExplainer(
                    self.calibrated_model or self.best_model,
                    self.X_train
                )
                self.explainer.initialize_explainer()
                
                # Global explanations
                global_explanation = self.explainer.generate_global_explanations()
                
                # Feature importance
                feature_importance = self.trainer.get_feature_importance(
                    self.calibrated_model or self.best_model,
                    self.X_train.columns
                )
                
                # SHAP plots
                shap_plots = [
                    "shap_summary.png",
                    "shap_waterfall.png",
                    "shap_force.png"
                ]
                
                for plot_name in shap_plots:
                    plot_path = f"outputs/figures/{plot_name}"
                    self._log_artifacts({plot_name: plot_path})
                
                # Save explanations
                explain_path = f"outputs/reports/explanations_{self.run_id}.json"
                with open(explain_path, 'w') as f:
                    json.dump({
                        "global_explanation": global_explanation,
                        "feature_importance": feature_importance.to_dict('records'),
                    }, f, indent=2)
                self._log_artifacts({"explanations": explain_path})
                
                logger.info("✅ Explanations generated")
                logger.info(f"   Top features: {global_explanation['top_features'][:5]}")
                
            except Exception as e:
                logger.error(f"❌ Explanation generation failed: {str(e)}")
                self.context.errors.append(f"Explainability: {str(e)}")
                raise
        
        return self
    
    def step7_audit_bias(self) -> 'TrainingPipeline':
        """
        Step 7: Bias Auditing
        
        - Demographic parity
        - Equal opportunity
        - Disparate impact
        - Fairness metrics
        """
        logger.info("⚖️ Step 7: Bias Auditing")
        
        with self._mlflow_run():
            try:
                # Initialize bias monitor
                self.bias_monitor = BiasMonitor()
                
                # Audit bias
                bias_report = self.bias_monitor.audit(
                    self.calibrated_model or self.best_model,
                    self.X_test,
                    self.y_test,
                    protected_attributes=['age', 'sex']
                )
                
                # Log bias metrics
                for metric_name, value in bias_report.items():
                    if isinstance(value, (int, float)):
                        self._log_metrics({f"bias_{metric_name}": value})
                
                # Save bias report
                bias_path = f"outputs/reports/bias_report_{self.run_id}.json"
                with open(bias_path, 'w') as f:
                    json.dump(bias_report, f, indent=2)
                self._log_artifacts({"bias_report": bias_path})
                
                logger.info("✅ Bias audit complete")
                
            except Exception as e:
                logger.error(f"❌ Bias audit failed: {str(e)}")
                self.context.errors.append(f"Bias audit: {str(e)}")
                raise
        
        return self
    
    def step8_validate_golden_tests(self) -> 'TrainingPipeline':
        """
        Step 8: Golden Test Validation
        
        - Known patient cases
        - Edge cases
        - Performance thresholds
        """
        logger.info("🥇 Step 8: Golden Test Validation")
        
        with self._mlflow_run():
            try:
                # Initialize validator
                self.validator = DataValidator()
                
                # Load golden tests
                golden_tests = self.validator.load_golden_tests()
                
                # Validate model
                validation_results = self.validator.validate_model(
                    self.calibrated_model or self.best_model,
                    golden_tests
                )
                
                # Log validation results
                for test_name, result in validation_results.items():
                    self._log_metrics({
                        f"golden_test_{test_name}": result['score']
                    })
                
                # Check if any test failed
                failures = [k for k, v in validation_results.items() if v['status'] == 'failed']
                if failures:
                    logger.warning(f"⚠️ Golden test failures: {failures}")
                    self.context.errors.append(f"Golden tests failed: {failures}")
                
                self._log_artifacts({
                    "golden_test_results": f"outputs/reports/golden_tests_{self.run_id}.json"
                })
                
                logger.info("✅ Golden test validation complete")
                
            except Exception as e:
                logger.error(f"❌ Golden test validation failed: {str(e)}")
                self.context.errors.append(f"Golden tests: {str(e)}")
                raise
        
        return self
    
    def step9_register_model(self) -> 'TrainingPipeline':
        """
        Step 9: Model Registration
        
        - MLflow model registry
        - Version management
        - Model staging
        - Metadata tracking
        """
        logger.info("📦 Step 9: Model Registration")
        
        with self._mlflow_run():
            try:
                # Initialize registry
                self.registry = ModelRegistry()
                
                # Prepare model artifacts
                model_path = f"models/staging/{self.run_id}"
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Save model
                import joblib
                joblib.dump(
                    self.calibrated_model or self.best_model,
                    f"{model_path}/model.pkl"
                )
                
                # Save metadata
                metadata = {
                    "run_id": self.run_id,
                    "model_name": self.best_model_name,
                    "metrics": self.evaluation_results[self.best_model_name],
                    "features": self.X_train.columns.tolist(),
                    "training_date": datetime.now().isoformat(),
                    "dataset_version": self.context.dataset_version,
                    "config": self.config.config,
                }
                with open(f"{model_path}/metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Register model
                model_version = self.registry.register_model(
                    model_path=model_path,
                    model_name=f"cdss_{self.config.get('data.dataset_type', 'diabetes')}",
                    version=self.context.model_version,
                    metadata=metadata,
                    stage="staging"
                )
                
                self.context.model_version = model_version
                
                # Log model to MLflow
                mlflow.sklearn.log_model(
                    self.calibrated_model or self.best_model,
                    artifact_path="model",
                    registered_model_name=f"cdss_model_{self.run_id}",
                )
                
                logger.info(f"✅ Model registered: {model_version}")
                logger.info(f"   Stage: staging")
                
            except Exception as e:
                logger.error(f"❌ Model registration failed: {str(e)}")
                self.context.errors.append(f"Registry: {str(e)}")
                raise
        
        return self
    
    def step10_generate_reports(self) -> 'TrainingPipeline':
        """
        Step 10: Generate Reports
        
        - Training summary
        - Model card
        - Clinical report
        - Documentation
        """
        logger.info("📄 Step 10: Generating Reports")
        
        with self._mlflow_run():
            try:
                # Initialize reporter
                self.reporter = ClinicalReporter(
                    dataset_type=self.config.get('data.dataset_type', 'diabetes')
                )
                
                # Generate training report
                report = {
                    "run_id": self.run_id,
                    "timestamp": datetime.now().isoformat(),
                    "dataset": {
                        "type": self.config.get('data.dataset_type'),
                        "samples": len(self.X),
                        "features": len(self.X.columns),
                    },
                    "best_model": {
                        "name": self.best_model_name,
                        "version": self.context.model_version,
                        "metrics": self.evaluation_results[self.best_model_name],
                    },
                    "errors": self.context.errors,
                    "duration": time.time() - self.context.start_time,
                }
                
                # Save report
                report_path = f"outputs/reports/training_report_{self.run_id}.json"
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                self._log_artifacts({"training_report": report_path})
                
                # Generate model card
                model_card = self.reporter.generate_model_card(
                    model_name=self.best_model_name,
                    version=self.context.model_version,
                    metrics=self.evaluation_results[self.best_model_name],
                    features=self.X_train.columns.tolist(),
                )
                card_path = f"outputs/reports/model_card_{self.run_id}.md"
                with open(card_path, 'w') as f:
                    f.write(model_card)
                self._log_artifacts({"model_card": card_path})
                
                logger.info("✅ Reports generated")
                
            except Exception as e:
                logger.error(f"❌ Report generation failed: {str(e)}")
                self.context.errors.append(f"Reporting: {str(e)}")
                raise
        
        return self
    
    # ============================================================================
    # 🚀 Main Pipeline Execution
    # ============================================================================
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Returns:
            Dictionary with pipeline results
        """
        start_time = time.time()
        logger.info("="*60)
        logger.info("🚀 STARTING TRAINING PIPELINE")
        logger.info("="*60)
        
        try:
            # Execute all steps
            (self.step1_ingest_data()
             .step2_preprocess_data()
             .step3_train_models()
             .step4_evaluate_models()
             .step5_calibrate_model()
             .step6_generate_explanations()
             .step7_audit_bias()
             .step8_validate_golden_tests()
             .step9_register_model()
             .step10_generate_reports())
            
            # Update context
            self.context.status = "completed"
            self.context.metrics = self.evaluation_results.get(self.best_model_name, {})
            
            # Log final metrics
            self._log_metrics({
                "pipeline_duration": time.time() - start_time,
                "pipeline_status": 1,  # Success
            })
            
            logger.info("="*60)
            logger.info(f"✅ PIPELINE COMPLETED in {time.time() - start_time:.2f}s")
            logger.info(f"   Best Model: {self.best_model_name}")
            logger.info(f"   Recall: {self.context.metrics.get('recall', 0):.3f}")
            logger.info("="*60)
            
            return {
                "status": "success",
                "run_id": self.run_id,
                "best_model": self.best_model_name,
                "metrics": self.context.metrics,
                "model_version": self.context.model_version,
                "duration": time.time() - start_time,
                "errors": self.context.errors,
            }
            
        except Exception as e:
            self.context.status = "failed"
            logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
            
            # Log failure
            self._log_metrics({
                "pipeline_duration": time.time() - start_time,
                "pipeline_status": 0,  # Failure
            })
            
            return {
                "status": "failed",
                "run_id": self.run_id,
                "error": str(e),
                "duration": time.time() - start_time,
            }
    
    # ============================================================================
    # 🔧 Utility Methods
    # ============================================================================
    
    def get_context(self) -> TrainingContext:
        """Get training context"""
        return self.context
    
    def get_results(self) -> Dict[str, Any]:
        """Get pipeline results"""
        return {
            "best_model": self.best_model_name,
            "metrics": self.evaluation_results.get(self.best_model_name, {}),
            "model_version": self.context.model_version,
            "validation_results": self.validation_results,
        }
    
    def load_saved_models(self) -> Dict[str, Any]:
        """Load previously trained models"""
        import joblib
        models = {}
        
        model_dir = Path("models/staging")
        if model_dir.exists():
            for model_path in model_dir.glob("*/model.pkl"):
                try:
                    model_name = model_path.parent.name
                    models[model_name] = joblib.load(model_path)
                except Exception:
                    continue
        
        return models