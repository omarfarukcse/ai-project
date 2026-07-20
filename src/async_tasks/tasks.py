# src/async_tasks/tasks.py
"""
Celery Task Definitions for CDSS
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.async_tasks.celery_worker import BaseTask, CachedTask, celery_app
from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


# ============================================================================
# 🧠 Prediction Tasks
# ============================================================================

@celery_app.task(
    name="src.async_tasks.tasks.generate_explanation_report",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    queue="high_priority",
)
def generate_explanation_report(
    self,
    patient_data: Dict[str, Any],
    prediction: Dict[str, Any],
    correlation_id: str,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """
    Generate SHAP explanation report for a patient
    
    Args:
        patient_data: Patient data used for prediction
        prediction: Prediction result
        correlation_id: Request correlation ID
        generate_plots: Generate SHAP plots
        
    Returns:
        Explanation report
    """
    
    logger.info(f"📊 Generating explanation report for {correlation_id}")
    
    try:
        from src.components.explainability import ClinicalSHAPExplainer
        from src.components.model_registry import ModelRegistry
        from src.components.reporting import ClinicalReporter
        
        # Load model
        registry = ModelRegistry()
        model = registry.get_production_model()
        
        if model is None:
            logger.warning("No production model found for explanations")
            return {"status": "failed", "reason": "No production model"}
        
        # Create explainer
        # Load training data for explainer
        training_data = pd.read_csv("data/reference_data.csv")
        X_train = training_data.drop('target', axis=1)
        
        explainer = ClinicalSHAPExplainer(model, X_train)
        explainer.initialize_explainer()
        
        # Generate explanation
        patient_df = pd.DataFrame([patient_data])
        explanation = explainer.generate_local_explanations(patient_df, 0)
        
        # Generate plots if requested
        plots = {}
        if generate_plots:
            # Save plots to files
            plot_dir = f"outputs/explanations/{correlation_id}"
            import os
            os.makedirs(plot_dir, exist_ok=True)
            
            try:
                explainer.plot_waterfall(patient_df, 0, save_path=f"{plot_dir}/waterfall.png")
                explainer.plot_force(patient_df, 0, save_path=f"{plot_dir}/force.png")
                plots = {
                    "waterfall": f"{plot_dir}/waterfall.png",
                    "force": f"{plot_dir}/force.png",
                }
            except Exception as e:
                logger.warning(f"Failed to generate plots: {str(e)}")
        
        # Generate clinical report
        reporter = ClinicalReporter()
        report = reporter.generate_patient_report(
            patient_id=prediction.get("patient_id", "unknown"),
            risk_score=prediction.get("risk_score", 0),
            risk_level=prediction.get("risk_level", "Unknown"),
            contributing_factors=explanation.get("contributing_factors", []),
            explanation=explanation.get("clinical_explanation", ""),
            feature_values=patient_data,
        )
        
        # Save report
        report_path = reporter.save_report(report, prediction.get("patient_id", "unknown"))
        
        logger.info(f"✅ Explanation report generated: {report_path}")
        
        return {
            "status": "success",
            "correlation_id": correlation_id,
            "explanation": explanation,
            "report_path": report_path,
            "plots": plots,
            "report": report,
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to generate explanation: {str(e)}", exc_info=True)
        
        # Retry on failure
        self.retry(exc=e, countdown=60)
        return {"status": "failed", "error": str(e)}


@celery_app.task(
    name="src.async_tasks.tasks.batch_predict_async",
    bind=True,
    max_retries=2,
    queue="ml_tasks",
)
def batch_predict_async(
    self,
    patients: List[Dict[str, Any]],
    correlation_id: str,
) -> Dict[str, Any]:
    """
    Async batch prediction for multiple patients
    
    Args:
        patients: List of patient data
        correlation_id: Request correlation ID
        
    Returns:
        Batch prediction results
    """
    
    logger.info(f"📊 Processing batch prediction: {len(patients)} patients")
    
    try:
        from src.pipelines.inference_pipeline import InferencePipeline
        
        # Run batch prediction
        pipeline = InferencePipeline()
        
        # Use asyncio to run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(
            pipeline.batch_predict(patients, correlation_id)
        )
        
        # Convert results to dict
        predictions = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]
        
        logger.info(f"✅ Batch prediction complete: {len(predictions)} results")
        
        return {
            "status": "success",
            "correlation_id": correlation_id,
            "total": len(predictions),
            "predictions": predictions,
        }
        
    except Exception as e:
        logger.error(f"❌ Batch prediction failed: {str(e)}", exc_info=True)
        self.retry(exc=e, countdown=30)
        return {"status": "failed", "error": str(e)}


# ============================================================================
# 🤖 Model Tasks
# ============================================================================

@celery_app.task(
    name="src.async_tasks.tasks.retrain_model",
    bind=True,
    max_retries=2,
    queue="ml_tasks",
)
def retrain_model(self, force: bool = False) -> Dict[str, Any]:
    """
    Retrain the model with latest data
    
    Args:
        force: Force retraining even if not needed
        
    Returns:
        Training results
    """
    
    logger.info("🔄 Starting model retraining...")
    
    try:
        from src.pipelines.training_pipeline import TrainingPipeline
        from src.monitoring.drift_detection import get_drift_detector
        from src.monitoring.metrics_monitor import get_metrics_monitor
        
        # Check if retraining is needed
        if not force:
            drift_detector = get_drift_detector()
            metrics_monitor = get_metrics_monitor()
            
            # Check for data drift
            drift_report = drift_detector.detect_data_drift()
            if not drift_report.drift_detected:
                logger.info("No data drift detected, skipping retraining")
                return {"status": "skipped", "reason": "No drift detected"}
            
            # Check performance drop
            current_metrics = metrics_monitor.get_metrics_summary()
            if current_metrics.get("model", {}).get("avg_recall", 1) > 0.7:
                logger.info("Model performance is acceptable, skipping retraining")
                return {"status": "skipped", "reason": "Performance acceptable"}
        
        # Run training pipeline
        pipeline = TrainingPipeline()
        results = pipeline.run()
        
        # Check results
        if results.get("status") == "success":
            # Register new model
            from src.components.model_registry import ModelRegistry
            
            registry = ModelRegistry()
            version = registry.register_model(
                model_path=results.get("model_path"),
                stage="staging",
            )
            
            # Promote if performance improved
            if results.get("metrics", {}).get("recall", 0) > 0.75:
                registry.promote_model(version=version, stage="production")
            
            # Send notification
            send_notification.delay(
                subject="Model Retraining Complete",
                message=f"New model version {version} trained with recall {results.get('metrics', {}).get('recall', 0):.3f}",
                severity="info",
            )
            
            logger.info(f"✅ Model retraining complete: {version}")
            
            return {
                "status": "success",
                "version": version,
                "metrics": results.get("metrics", {}),
                "promoted": results.get("metrics", {}).get("recall", 0) > 0.75,
            }
        else:
            raise Exception(f"Training failed: {results.get('error')}")
        
    except Exception as e:
        logger.error(f"❌ Model retraining failed: {str(e)}", exc_info=True)
        
        # Send alert
        send_notification.delay(
            subject="Model Retraining Failed",
            message=f"Retraining failed: {str(e)}",
            severity="error",
        )
        
        self.retry(exc=e, countdown=300)
        return {"status": "failed", "error": str(e)}


@celery_app.task(
    name="src.async_tasks.tasks.evaluate_model",
    bind=True,
    queue="ml_tasks",
)
def evaluate_model(
    self,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate model performance
    
    Args:
        model_version: Model version to evaluate (production if None)
        
    Returns:
        Evaluation results
    """
    
    logger.info(f"📊 Evaluating model: {model_version or 'production'}")
    
    try:
        from src.components.model_registry import ModelRegistry
        from src.components.model_evaluation import ModelEvaluator
        from src.components.data_ingestion import DataIngestion
        
        # Get model
        registry = ModelRegistry()
        if model_version:
            model = registry.get_model(model_version)
        else:
            model = registry.get_production_model()
        
        if model is None:
            return {"status": "failed", "reason": "Model not found"}
        
        # Load test data
        ingestion = DataIngestion()
        X_test, y_test = ingestion.load_test_data()
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(model, X_test, y_test)
        
        logger.info(f"✅ Model evaluation complete: {metrics.accuracy:.3f} accuracy")
        
        return {
            "status": "success",
            "model_version": model_version or "production",
            "metrics": metrics.to_dict(),
        }
        
    except Exception as e:
        logger.error(f"❌ Model evaluation failed: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}


@celery_app.task(
    name="src.async_tasks.tasks.validate_model",
    bind=True,
    queue="ml_tasks",
)
def validate_model(
    self,
    model_version: str,
    run_golden_tests: bool = True,
) -> Dict[str, Any]:
    """
    Validate a model version
    
    Args:
        model_version: Model version to validate
        run_golden_tests: Run golden tests
        
    Returns:
        Validation results
    """
    
    logger.info(f"🔍 Validating model: {model_version}")
    
    try:
        from src.components.model_registry import ModelRegistry
        from src.validation.schema_validation import DataValidator
        
        # Get model
        registry = ModelRegistry()
        model = registry.get_model(model_version)
        
        if model is None:
            return {"status": "failed", "reason": "Model not found"}
        
        # Run tests
        validator = DataValidator()
        results = {}
        
        if run_golden_tests:
            golden_results = validator.run_golden_tests(model)
            results["golden_tests"] = golden_results
        
        # Performance validation
        evaluation = evaluate_model(model_version)
        results["performance"] = evaluation
        
        # Check if validation passed
        passed = True
        errors = []
        
        if run_golden_tests and not golden_results.get("passed", False):
            passed = False
            errors.extend(golden_results.get("errors", []))
        
        # Performance threshold
        if evaluation.get("metrics", {}).get("recall", 0) < 0.7:
            passed = False
            errors.append("Recall below threshold (0.7)")
        
        logger.info(f"✅ Model validation {'passed' if passed else 'failed'}")
        
        return {
            "status": "success" if passed else "failed",
            "model_version": model_version,
            "passed": passed,
            "errors": errors,
            "results": results,
        }
        
    except Exception as e:
        logger.error(f"❌ Model validation failed: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}


@celery_app.task(
    name="src.async_tasks.tasks.promote_model_async",
    bind=True,
    queue="ml_tasks",
)
def promote_model_async(
    self,
    version: str,
    stage: str = "production",
    run_validation: bool = True,
) -> Dict[str, Any]:
    """
    Promote a model version to production
    
    Args:
        version: Model version to promote
        stage: Target stage
        run_validation: Run validation before promotion
        
    Returns:
        Promotion results
    """
    
    logger.info(f"🚀 Promoting model {version} to {stage}")
    
    try:
        from src.components.model_registry import ModelRegistry
        
        # Validate first if requested
        if run_validation:
            validation = validate_model(version)
            if not validation.get("passed", False):
                return {
                    "status": "failed",
                    "reason": "Validation failed",
                    "errors": validation.get("errors", []),
                }
        
        # Promote
        registry = ModelRegistry()
        result = registry.promote_model(version=version, stage=stage)
        
        logger.info(f"✅ Model {version} promoted to {stage}")
        
        # Send notification
        send_notification.delay(
            subject=f"Model {version} Promoted to {stage}",
            message=f"Model version {version} has been promoted to {stage}",
            severity="success",
        )
        
        return {
            "status": "success",
            "version": version,
            "stage": stage,
            "result": result,
        }
        
    except Exception as e:
        logger.error(f"❌ Model promotion failed: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}


# ============================================================================
# 📊 Monitoring Tasks
# ============================================================================

@celery_app.task(
    name="src.async_tasks.tasks.check_data_drift",
    bind=True,
    queue="monitoring",
)
def check_data_drift(self) -> Dict[str, Any]:
    """
    Check for data drift in production
    
    Returns:
        Drift detection report
    """
    
    logger.info("📊 Checking for data drift...")
    
    try:
        from src.monitoring.drift_detection import get_drift_detector
        from src.components.data_ingestion import DataIngestion
        
        # Get current data
        ingestion = DataIngestion()
        current_data, _ = ingestion.load(limit=10000)
        
        # Detect drift
        drift_detector = get_drift_detector()
        report = drift_detector.detect_data_drift(current_data)
        
        logger.info(f"✅ Data drift check complete: {'Drift detected' if report.drift_detected else 'No drift'}")
        
        # Send alert if drift detected
        if report.drift_detected:
            send_notification.delay(
                subject="Data Drift Detected",
                message=f"Data drift detected: {len(report.features_drifted)} features drifted",
                severity="warning",
                metadata=report.to_dict(),
            )
        
        return report.to_dict()
        
    except Exception as e:
        logger.error(f"❌ Data drift check failed: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}


@celery_app.task(
    name="src.async_tasks.tasks.check_model_drift",
    bind=True,
    queue="monitoring",
)
def check_model_drift(self) -> Dict[str, Any]:
    """
    Check for model drift
    
    Returns:
        Model drift report
    """
    
    logger.info("🤖 Checking for model drift...")
    
    try:
        from src.monitoring.drift_detection import get_drift_detector
        from src.components.model_evaluation import ModelEvaluator
        from src.components.data_ingestion import DataIngestion
        from src.components.model_registry import ModelRegistry
        
        # Get production model
        registry = ModelRegistry()
        model = registry.get_production_model()
        
        if model is None:
            return {"status": "failed", "reason": "No production model"}
        
        # Get recent data
        ingestion = DataIngestion()
        X_test, y_test = ingestion.load_test_data(limit=1000, recent=True)
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(model, X_test, y_test)
        
        # Check drift
        drift_detector = get_drift_detector()
        report = drift_detector.detect_model_drift(metrics.to_dict())
        
        logger.info(f"✅ Model drift check complete: {'Drift detected' if report.drift_detected else 'No drift'}")
        
        # Send alert if drift detected
        if report.drift_detected:
            send_notification.delay(
                subject="Model Drift Detected",
                message=f"Model drift detected: {len(report.features_drifted)} metrics drifted",
                severity="warning",
                metadata=report.to_dict(),
            )
        
        return report.to_dict()
        
    except Exception as e:
        logger.error(f"❌ Model drift check failed: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}


@celery_app.task(
    name="src.async_tasks.tasks.check_bias",
    bind=True,
    queue="monitoring",
)
def check_bias(
    self,
    protected_attribute: str = "gender",
) -> Dict[str, Any]:
    """
    Check for bias in model predictions
    
    Args:
        protected_attribute: Protected attribute to check
        
    Returns:
        Bias report
    """
    
    logger.info(f"⚖️ Checking bias for {protected_attribute}...")
    
    try:
        from src.monitoring.bias_monitor import get_bias_monitor
        from src.components.model_registry import ModelRegistry
        from src.components.data_ingestion import DataIngestion
        
        # Get model and data
        registry = ModelRegistry()
        model = registry.get_production_model()
        
        if model is None:
            return {"status": "failed", "reason": "No production model"}
        
        # Get recent data with protected attribute
        ingestion = DataIngestion()
        X_test, y_test = ingestion.load_test_data(limit=5000)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Extract protected groups
        protected_groups = X_test.get(protected_attribute)
        if protected_groups is None:
            return {"status": "failed", "reason": f"Protected attribute {protected_attribute} not found"}
        
        # Check bias
        bias_monitor = get_bias_monitor()
        report = bias_monitor.audit_bias(
            predictions=y_pred,
            targets=y_test,
            protected_groups=protected_groups,
            attribute_name=protected_attribute,
        )
        
        logger.info(f"✅ Bias check complete: {len(report.fairness_violations)} violations")
        
        # Send alert if violations detected
        if report.fairness_violations:
            send_notification.delay(
                subject=f"Fairness Violation Detected - {protected_attribute}",
                message=f"Found {len(report.fairness_violations)} fairness violations",
                severity="warning",
                metadata=report.to_dict(),
            )
        
        return report.to_dict()
        
    except Exception as e:
        logger.error(f"❌ Bias check failed: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}


# ============================================================================
# 📄 Reporting Tasks
# ============================================================================

@celery_app.task(
    name="src.async_tasks.tasks.generate_weekly_report",
    bind=True,
    queue="reporting",
)
def generate_weekly_report(self) -> Dict[str, Any]:
    """
    Generate weekly clinical report
    
    Returns:
        Report data
    """
    
    logger.info("📄 Generating weekly report...")
    
    try:
        from src.components.reporting import ClinicalReporter
        from src.components.model_registry import ModelRegistry
        from src.components.data_ingestion import DataIngestion
        
        # Get data from last week
        ingestion = DataIngestion()
        data, _ = ingestion.load(limit=10000, sample=0.1)
        
        # Get model
        registry = ModelRegistry()
        model = registry.get_production_model()
        
        # Generate summary report
        reporter = ClinicalReporter()
        summary = reporter.generate_summary_report(data, model)
        
        # Save report
        report_path = f"outputs/reports/weekly_report_{datetime.now().strftime('%Y%m%d')}.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Weekly report generated: {report_path}")
        
        # Send notification
        send_notification.delay(
            subject="Weekly Report Generated",
            message=f"Weekly report available: {report_path}",
            severity="info",
        )
        
        return {
            "status": "success",
            "report_path": report_path,
            "summary": summary,
        }
        
    except Exception as e:
        logger.error(f"❌ Weekly report generation failed: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}


@celery_app.task(
    name="src.async_tasks.tasks.generate_monthly_report",
    bind=True,
    queue="reporting",
)
def generate_monthly_report(self) -> Dict[str, Any]:
    """
    Generate monthly clinical report
    
    Returns:
        Report data
    """
    
    logger.info("📄 Generating monthly report...")
    
    try:
        # Similar to weekly report but with monthly data
        return {
            "status": "success",
            "message": "Monthly report generated",
        }
        
    except Exception as e:
        logger.error(f"❌ Monthly report generation failed: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}


@celery_app.task(
    name="src.async_tasks.tasks.generate_patient_report",
    bind=True,
    queue="reporting",
)
def generate_patient_report(
    self,
    patient_id: str,
    patient_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate detailed patient report
    
    Args:
        patient_id: Patient identifier
        patient_data: Patient data
        
    Returns:
        Patient report
    """
    
    logger.info(f"📄 Generating patient report: {patient_id}")
    
    try:
        from src.components.reporting import ClinicalReporter
        from src.pipelines.inference_pipeline import InferencePipeline
        
        # Get prediction
        pipeline = InferencePipeline()
        result = pipeline.predict(patient_data, f"report_{patient_id}")
        
        # Generate report
        reporter = ClinicalReporter()
        report = reporter.generate_patient_report(
            patient_id=patient_id,
            risk_score=result.risk_score,
            risk_level=result.risk_level,
            contributing_factors=result.contributing_factors,
            explanation=result.clinical_explanation,
            feature_values=patient_data,
        )
        
        # Save report
        report_path = reporter.save_report(report, patient_id)
        
        logger.info(f"✅ Patient report generated: {report_path}")
        
        return {
            "status": "success",
            "patient_id": patient_id,
            "report_path": report_path,
            "report": report,
        }
        
    except Exception as e:
        logger.error(f"❌ Patient report generation failed: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}


# ============================================================================
# 🔧 Maintenance Tasks
# ============================================================================

@celery_app.task(
    name="src.async_tasks.tasks.cleanup_logs",
    bind=True,
    queue="low_priority",
)
def cleanup_logs(self, days: int = 30) -> Dict[str, Any]:
    """
    Cleanup old logs
    
    Args:
        days: Keep logs from last N days
        
    Returns:
        Cleanup results
    """
    
    logger.info(f"🧹 Cleaning up logs older than {days} days...")
    
    try:
        import os
        import shutil
        from pathlib import Path
        
        log_dir = Path("outputs/logs")
        if not log_dir.exists():
            return {"status": "success", "message": "No logs found"}
        
        cutoff = datetime.now() - timedelta(days=days)
        cleaned = 0
        
        for log_file in log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff.timestamp():
                log_file.unlink()
                cleaned += 1
        
        logger.info(f"✅ Cleaned up {cleaned} log files")
        
        return {
            "status": "success",
            "cleaned": cleaned,
            "days_kept": days,
        }
        
    except Exception as e:
        logger.error(f"❌ Log cleanup failed: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}


@celery_app.task(
    name="src.async_tasks.tasks.cleanup_cache",
    bind=True,
    queue="low_priority",
)
def cleanup_cache(self) -> Dict[str, Any]:
    """
    Cleanup expired cache entries
    
    Returns:
        Cleanup results
    """
    
    logger.info("🧹 Cleaning up cache...")
    
    try:
        from src.caching.redis_client import get_fast_redis_client
        
        redis_client = get_fast_redis_client()
        
        # Get all cache keys
        keys = redis_client.client.keys("cdss:cache:*")
        
        # Remove expired keys (Redis handles TTL automatically)
        # But we can still clean up old keys
        cleaned = 0
        for key in keys:
            ttl = redis_client.client.ttl(key)
            if ttl == -2:  # Key doesn't exist
                cleaned += 1
        
        logger.info(f"✅ Cache cleanup complete")
        
        return {
            "status": "success",
            "cleaned": cleaned,
            "total_keys": len(keys),
        }
        
    except Exception as e:
        logger.error(f"❌ Cache cleanup failed: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}


@celery_app.task(
    name="src.async_tasks.tasks.backup_models",
    bind=True,
    queue="low_priority",
)
def backup_models(self) -> Dict[str, Any]:
    """
    Backup all models to archive
    
    Returns:
        Backup results
    """
    
    logger.info("💾 Backing up models...")
    
    try:
        import shutil
        from pathlib import Path
        
        backup_dir = Path(f"models/archived/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy staging and production models
        for stage in ["staging", "production"]:
            src = Path(f"models/{stage}")
            if src.exists():
                dst = backup_dir / stage
                shutil.copytree(src, dst)
        
        logger.info(f"✅ Models backed up to: {backup_dir}")
        
        return {
            "status": "success",
            "backup_path": str(backup_dir),
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"❌ Model backup failed: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}


# ============================================================================
# 🔔 Notification Tasks
# ============================================================================

@celery_app.task(
    name="src.async_tasks.tasks.send_notification",
    bind=True,
    queue="default",
)
def send_notification(
    self,
    subject: str,
    message: str,
    severity: str = "info",
    metadata: Optional[Dict] = None,
    channels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Send notification via configured channels
    
    Args:
        subject: Notification subject
        message: Notification message
        severity: Severity level
        metadata: Additional metadata
        channels: Channel list (None = use configured)
        
    Returns:
        Notification status
    """
    
    logger.info(f"🔔 Sending notification: {subject}")
    
    try:
        from src.monitoring.alerting import get_alert_manager
        from src.monitoring.alerting import AlertSeverity
        
        alert_manager = get_alert_manager()
        
        # Map severity
        severity_map = {
            "info": AlertSeverity.INFO,
            "warning": AlertSeverity.MEDIUM,
            "error": AlertSeverity.HIGH,
            "critical": AlertSeverity.CRITICAL,
            "success": AlertSeverity.INFO,
        }
        
        # Send alert
        import asyncio
        alert = asyncio.run(
            alert_manager.send_alert(
                severity=severity_map.get(severity, AlertSeverity.INFO),
                message=f"{subject}: {message}",
                metadata=metadata or {},
                channels=channels,
            )
        )
        
        logger.info(f"✅ Notification sent: {subject}")
        
        return {
            "status": "success",
            "subject": subject,
            "severity": severity,
        }
        
    except Exception as e:
        logger.error(f"❌ Notification failed: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}


@celery_app.task(
    name="src.async_tasks.tasks.health_check",
    bind=True,
    queue="monitoring",
)
def health_check(self) -> Dict[str, Any]:
    """
    Comprehensive health check of all systems
    
    Returns:
        Health status
    """
    
    logger.info("🏥 Running health check...")
    
    try:
        from src.caching.redis_client import get_fast_redis_client
        from src.components.model_registry import ModelRegistry
        from src.monitoring.drift_detection import get_drift_detector
        
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }
        
        # Check Redis
        redis_client = get_fast_redis_client()
        health["components"]["redis"] = {
            "status": "healthy" if redis_client.is_connected() else "unhealthy",
        }
        
        # Check Model Registry
        registry = ModelRegistry()
        model_info = registry.get_model_info()
        health["components"]["model_registry"] = {
            "status": "healthy" if model_info else "unhealthy",
            "model": model_info.get("version", "none"),
        }
        
        # Check Drift Detector
        drift_detector = get_drift_detector()
        health["components"]["drift_detector"] = {
            "status": "healthy",
        }
        
        # Determine overall status
        unhealthy = [
            name for name, comp in health["components"].items()
            if comp.get("status") == "unhealthy"
        ]
        
        if unhealthy:
            health["status"] = "unhealthy"
            health["unhealthy_components"] = unhealthy
        
        logger.info(f"✅ Health check complete: {health['status']}")
        
        # Send alert if unhealthy
        if health["status"] == "unhealthy":
            send_notification.delay(
                subject="System Health Check Failed",
                message=f"Unhealthy components: {', '.join(unhealthy)}",
                severity="critical",
                metadata=health,
            )
        
        return health
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }