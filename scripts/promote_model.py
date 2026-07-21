# scripts/promote_model.py
#!/usr/bin/env python
"""
Model Promotion Script

This script promotes a model from staging to production with:
- Canary deployment support
- Performance validation
- Rollback capability
- Golden test validation
- Bias auditing
- Notification sending

Usage:
    python scripts/promote_model.py --version v1.0.0
    python scripts/promote_model.py --version v1.0.0 --canary --canary-percentage 0.1
    python scripts/promote_model.py --version v1.0.0 --no-validation
    python scripts/promote_model.py --rollback
    python scripts/promote_model.py --help
"""

import os
import sys
import json
import time
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.components.model_registry import ModelRegistry
from src.components.model_evaluation import ModelEvaluator
from src.components.data_ingestion import DataIngestion
from src.components.model_calibration import ModelCalibrator
from src.monitoring.bias_monitor import BiasMonitor
from src.monitoring.drift_detection import DriftDetector
from src.validation.schema_validation import GoldenTestSuite
from src.async_tasks.tasks import send_notification
from src.logger import get_logger
from src.config_manager import get_config_manager

logger = get_logger(__name__)


class ModelPromoter:
    """
    Model promotion with canary deployment and validation
    """
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.evaluator = ModelEvaluator()
        self.calibrator = ModelCalibrator()
        self.bias_monitor = BiasMonitor()
        self.drift_detector = DriftDetector()
        self.golden_suite = GoldenTestSuite()
        self.config = get_config_manager()
        
        # Promotion thresholds
        self.recall_threshold = 0.7
        self.bias_threshold = 0.8
        self.drift_threshold = 0.2
        
        # Canary settings
        self.canary_enabled = True
        self.canary_percentage = 0.05
        self.monitor_duration = 300  # 5 minutes
        self.rollback_on_failure = True
        
        logger.info("🚀 ModelPromoter initialized")
        logger.info(f"   Recall Threshold: {self.recall_threshold}")
        logger.info(f"   Bias Threshold: {self.bias_threshold}")
        logger.info(f"   Drift Threshold: {self.drift_threshold}")
    
    def promote_model(
        self,
        version: str,
        source_stage: str = "staging",
        target_stage: str = "production",
        run_validation: bool = True,
        run_golden_tests: bool = True,
        run_bias_audit: bool = True,
        canary_deployment: bool = False,
        canary_percentage: Optional[float] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Promote a model to production
        
        Args:
            version: Model version to promote
            source_stage: Source stage
            target_stage: Target stage
            run_validation: Run performance validation
            run_golden_tests: Run golden tests
            run_bias_audit: Run bias audit
            canary_deployment: Enable canary deployment
            canary_percentage: Canary traffic percentage
            force: Force promotion without validation
            
        Returns:
            Promotion results
        """
        
        logger.info(f"🚀 Promoting model {version} from {source_stage} to {target_stage}")
        
        # Get model metadata
        metadata = self.registry.get_model_metadata("cdss_model", version, source_stage)
        if not metadata:
            raise ValueError(f"Model {version} not found in {source_stage}")
        
        # Load model
        model = self.registry.load_model("cdss_model", version, source_stage)[0]
        
        # Run validations
        validation_results = {}
        
        if run_validation and not force:
            logger.info("🔍 Running validation checks...")
            validation_results = self._run_validation(model)
            
            if not validation_results.get("passed", False):
                error_msg = validation_results.get("errors", ["Validation failed"])
                logger.error(f"❌ Validation failed: {', '.join(error_msg)}")
                
                if not force:
                    raise ValueError(f"Validation failed: {', '.join(error_msg)}")
        
        if run_golden_tests and not force:
            logger.info("🥇 Running golden tests...")
            golden_results = self._run_golden_tests(model)
            validation_results["golden_tests"] = golden_results
            
            if not golden_results.get("passed", False):
                error_msg = golden_results.get("errors", ["Golden tests failed"])
                logger.error(f"❌ Golden tests failed: {', '.join(error_msg)}")
                
                if not force:
                    raise ValueError(f"Golden tests failed: {', '.join(error_msg)}")
        
        if run_bias_audit and not force:
            logger.info("⚖️ Running bias audit...")
            bias_results = self._run_bias_audit(model)
            validation_results["bias_audit"] = bias_results
            
            if not bias_results.get("passed", False):
                logger.warning(f"⚠️ Bias audit detected issues: {bias_results.get('violations', [])}")
                
                if not force:
                    raise ValueError(f"Bias audit failed: {bias_results.get('violations')}")
        
        # Canary deployment
        if canary_deployment and target_stage == "production":
            logger.info(f"🦜 Deploying canary with {canary_percentage or self.canary_percentage * 100}% traffic")
            canary_results = self._deploy_canary(
                model,
                version,
                canary_percentage or self.canary_percentage
            )
            validation_results["canary"] = canary_results
            
            if not canary_results.get("passed", False) and self.rollback_on_failure:
                logger.error("❌ Canary deployment failed, rolling back...")
                self._rollback_canary(version)
                raise ValueError("Canary deployment failed")
        
        # Promote model
        promoted_metadata = self.registry.promote_to_production(version)
        
        # Send notification
        self._send_notification(
            subject=f"Model {version} Promoted to Production",
            message=f"Model {version} successfully promoted to production",
            severity="success",
            metadata={
                "version": version,
                "validation_results": validation_results,
                "promoted_at": datetime.now().isoformat(),
            }
        )
        
        # Save promotion report
        report = {
            "status": "success",
            "version": version,
            "source_stage": source_stage,
            "target_stage": target_stage,
            "promoted_at": datetime.now().isoformat(),
            "validation_results": validation_results,
            "metadata": promoted_metadata.to_dict() if promoted_metadata else {},
        }
        
        self._save_report(report)
        
        logger.info(f"✅ Model {version} promoted to {target_stage}")
        
        return report
    
    def _run_validation(self, model: Any) -> Dict[str, Any]:
        """Run performance validation"""
        
        try:
            # Load test data
            ingestion = DataIngestion()
            X, y = ingestion.load_with_validation()[:2]
            
            from src.components.preprocessing import ClinicalPreprocessor
            preprocessor = ClinicalPreprocessor()
            X_train, X_test, y_train, y_test = preprocessor.fit_transform(X, y)
            
            # Evaluate
            metrics = self.evaluator.evaluate_model(model, X_test, y_test)
            
            # Check thresholds
            passed = True
            errors = []
            
            if metrics.recall < self.recall_threshold:
                passed = False
                errors.append(f"Recall {metrics.recall:.3f} < {self.recall_threshold}")
            
            # Check for severe performance issues
            if metrics.accuracy < 0.6:
                passed = False
                errors.append(f"Accuracy {metrics.accuracy:.3f} < 0.6")
            
            return {
                "passed": passed,
                "errors": errors,
                "metrics": {
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "specificity": metrics.specificity,
                    "f1_score": metrics.f1_score,
                    "roc_auc": metrics.roc_auc,
                },
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                "passed": False,
                "errors": [f"Validation error: {str(e)}"],
                "metrics": {},
            }
    
    def _run_golden_tests(self, model: Any) -> Dict[str, Any]:
        """Run golden tests"""
        
        try:
            # Define prediction function
            def predict_fn(data):
                df = pd.DataFrame([data])
                return model.predict_proba(df)[0][1]
            
            # Run tests
            results = self.golden_suite.run_tests(predict_fn)
            summary = self.golden_suite.get_summary()
            
            passed = summary.get("failed", 0) == 0
            errors = []
            
            if not passed:
                for result in results:
                    if not result.passed:
                        errors.append(f"{result.test_name}: {result.differences}")
            
            return {
                "passed": passed,
                "errors": errors,
                "summary": summary,
                "results": [r.to_dict() for r in results],
            }
            
        except Exception as e:
            logger.error(f"Golden tests failed: {str(e)}")
            return {
                "passed": False,
                "errors": [f"Golden tests error: {str(e)}"],
            }
    
    def _run_bias_audit(self, model: Any) -> Dict[str, Any]:
        """Run bias audit"""
        
        try:
            # Load test data with protected attributes
            ingestion = DataIngestion()
            X, y = ingestion.load_with_validation()[:2]
            
            # Get predictions
            y_pred = model.predict(X)
            
            # Audit bias for protected attributes
            results = {}
            passed = True
            violations = []
            
            protected_attributes = ['age', 'sex']
            for attr in protected_attributes:
                if attr in X.columns:
                    groups = X[attr]
                    report = self.bias_monitor.audit_bias(
                        predictions=y_pred,
                        targets=y,
                        protected_groups=groups,
                        attribute_name=attr,
                    )
                    
                    if report.fairness_violations:
                        passed = False
                        violations.extend(report.fairness_violations)
                    
                    results[attr] = report.to_dict()
            
            return {
                "passed": passed,
                "violations": violations,
                "results": results,
            }
            
        except Exception as e:
            logger.error(f"Bias audit failed: {str(e)}")
            return {
                "passed": False,
                "violations": [f"Bias audit error: {str(e)}"],
                "results": {},
            }
    
    def _deploy_canary(self, model: Any, version: str, percentage: float) -> Dict[str, Any]:
        """Deploy canary version with traffic splitting"""
        
        logger.info(f"🦜 Deploying canary for {version} with {percentage*100}% traffic")
        
        try:
            # In production, this would:
            # 1. Deploy model with canary label
            # 2. Configure traffic splitting in service mesh
            # 3. Monitor metrics
            # 4. Validate performance
            
            # Simulate canary deployment
            start_time = time.time()
            metrics_samples = []
            
            while time.time() - start_time < self.monitor_duration:
                # Simulate metrics collection
                # In production, this would query Prometheus
                metrics = self._simulate_canary_metrics()
                metrics_samples.append(metrics)
                
                # Check error rate
                if metrics["error_rate"] > 0.05:
                    logger.warning(f"⚠️ Canary error rate: {metrics['error_rate']:.2%}")
                    return {
                        "passed": False,
                        "reason": f"Error rate {metrics['error_rate']:.2%} > 5%",
                        "metrics": metrics_samples,
                    }
                
                # Check latency
                if metrics["latency_p95"] > 2.0:
                    logger.warning(f"⚠️ Canary latency: {metrics['latency_p95']:.2f}s")
                    return {
                        "passed": False,
                        "reason": f"Latency {metrics['latency_p95']:.2f}s > 2s",
                        "metrics": metrics_samples,
                    }
                
                time.sleep(15)
            
            # Canary successful
            return {
                "passed": True,
                "metrics": metrics_samples,
                "duration": self.monitor_duration,
                "percentage": percentage,
            }
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {str(e)}")
            return {
                "passed": False,
                "reason": f"Canary error: {str(e)}",
                "metrics": [],
            }
    
    def _simulate_canary_metrics(self) -> Dict[str, float]:
        """Simulate canary metrics (for demo)"""
        
        import random
        
        return {
            "error_rate": random.uniform(0.001, 0.02),
            "latency_p95": random.uniform(0.1, 0.5),
            "throughput": random.uniform(50, 200),
            "success_rate": random.uniform(0.98, 1.0),
        }
    
    def _rollback_canary(self, version: str):
        """Rollback canary deployment"""
        
        logger.warning(f"🔄 Rolling back canary for {version}")
        
        try:
            # In production, this would:
            # 1. Remove canary deployment
            # 2. Restore traffic to previous version
            # 3. Clean up canary resources
            
            # Rollback to previous production version
            self.registry.rollback()
            
            logger.info(f"✅ Canary rolled back successfully")
            
        except Exception as e:
            logger.error(f"Canary rollback failed: {str(e)}")
    
    def _send_notification(self, subject: str, message: str, severity: str, metadata: Dict):
        """Send notification"""
        
        try:
            # Use Celery task for async notification
            send_notification.delay(
                subject=subject,
                message=message,
                severity=severity,
                metadata=metadata,
            )
            logger.info(f"📧 Notification sent: {subject}")
        except Exception as e:
            logger.warning(f"Notification failed: {str(e)}")
    
    def _save_report(self, report: Dict):
        """Save promotion report"""
        
        report_dir = Path("outputs/reports/promotions")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"promotion_{report['version']}_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Promotion report saved: {report_path}")
    
    def rollback_model(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Rollback to previous version"""
        
        logger.info(f"🔄 Rolling back model (target: {version or 'previous'})")
        
        try:
            result = self.registry.rollback(version)
            
            # Send notification
            self._send_notification(
                subject=f"Model Rollback to {result['current_version']}",
                message=f"Rolled back from {result['previous_version']} to {result['current_version']}",
                severity="warning",
                metadata=result,
            )
            
            logger.info(f"✅ Rollback complete: {result['previous_version']} -> {result['current_version']}")
            
            return {
                "status": "success",
                "previous_version": result['previous_version'],
                "current_version": result['current_version'],
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
            }


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Promote a model from staging to production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Promote model to production
  python scripts/promote_model.py --version v1.0.0
  
  # Promote with canary deployment
  python scripts/promote_model.py --version v1.0.0 --canary --canary-percentage 0.1
  
  # Force promotion without validation
  python scripts/promote_model.py --version v1.0.0 --force
  
  # Skip specific validations
  python scripts/promote_model.py --version v1.0.0 --no-golden --no-bias
  
  # Rollback to previous version
  python scripts/promote_model.py --rollback
        """
    )
    
    parser.add_argument(
        "--version",
        type=str,
        help="Model version to promote"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        choices=["staging", "production", "archived"],
        default="staging",
        help="Source stage (default: staging)"
    )
    
    parser.add_argument(
        "--target",
        type=str,
        choices=["staging", "production"],
        default="production",
        help="Target stage (default: production)"
    )
    
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip performance validation"
    )
    
    parser.add_argument(
        "--no-golden",
        action="store_true",
        help="Skip golden tests"
    )
    
    parser.add_argument(
        "--no-bias",
        action="store_true",
        help="Skip bias audit"
    )
    
    parser.add_argument(
        "--canary",
        action="store_true",
        help="Enable canary deployment"
    )
    
    parser.add_argument(
        "--canary-percentage",
        type=float,
        default=0.05,
        help="Canary traffic percentage (default: 0.05)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force promotion without validation"
    )
    
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback to previous version"
    )
    
    parser.add_argument(
        "--rollback-version",
        type=str,
        help="Specific version to rollback to"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize promoter
        promoter = ModelPromoter()
        
        # Handle rollback
        if args.rollback:
            result = promoter.rollback_model(args.rollback_version)
            
            print("\n" + "="*60)
            print("✅ Rollback Complete")
            print("="*60)
            print(f"Previous Version: {result.get('previous_version', 'N/A')}")
            print(f"Current Version:  {result.get('current_version', 'N/A')}")
            print("="*60)
            return
        
        # Validate required arguments
        if not args.version:
            parser.error("--version is required for promotion")
        
        # Promote model
        result = promoter.promote_model(
            version=args.version,
            source_stage=args.source,
            target_stage=args.target,
            run_validation=not args.no_validation,
            run_golden_tests=not args.no_golden,
            run_bias_audit=not args.no_bias,
            canary_deployment=args.canary,
            canary_percentage=args.canary_percentage,
            force=args.force,
        )
        
        print("\n" + "="*60)
        print("✅ Model Promotion Complete")
        print("="*60)
        print(f"Version:       {result['version']}")
        print(f"From:          {result['source_stage']}")
        print(f"To:            {result['target_stage']}")
        print(f"Promoted At:   {result['promoted_at']}")
        
        if result.get('validation_results'):
            validation = result['validation_results']
            print("\n📊 Validation Results:")
            if 'passed' in validation:
                print(f"   Performance: {'✅ PASSED' if validation.get('passed') else '❌ FAILED'}")
            if 'golden_tests' in validation:
                print(f"   Golden Tests: {'✅ PASSED' if validation['golden_tests'].get('passed') else '❌ FAILED'}")
            if 'bias_audit' in validation:
                print(f"   Bias Audit: {'✅ PASSED' if validation['bias_audit'].get('passed') else '❌ FAILED'}")
            if 'canary' in validation:
                print(f"   Canary: {'✅ PASSED' if validation['canary'].get('passed') else '❌ FAILED'}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"❌ Promotion failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()