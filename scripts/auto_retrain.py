# scripts/auto_retrain.py
import sys
from pathlib import Path
import json
from datetime import datetime
import subprocess

from src.logger import get_logger
from src.monitoring.comprehensive_drift import ComprehensiveDriftDetector
from src.monitoring.ground_truth_pipeline import GroundTruthPipeline
from src.components.model_registry import ModelRegistry

logger = get_logger(__name__)

class AutoRetrainManager:
    """
    Automated retraining strategy with drift detection and validation
    """
    
    def __init__(self):
        self.drift_detector = ComprehensiveDriftDetector()
        self.ground_truth = GroundTruthPipeline()
        self.model_registry = ModelRegistry()
    
    def should_retrain(self) -> bool:
        """
        Determine if retraining is needed based on:
        1. Drift detection
        2. Performance degradation
        3. Scheduled retraining
        """
        # Check drift
        drift_report = self.drift_detector.get_drift_report()
        if drift_report.get('overall_risk') == 'HIGH':
            logger.warning("High drift detected - retraining triggered")
            return True
        
        # Check performance
        performance = self.ground_truth.get_performance_trend()
        if performance.get('recent_accuracy', 1.0) < 0.75:
            logger.warning("Performance degradation detected - retraining triggered")
            return True
        
        # Check scheduled retraining
        registry_info = self.model_registry.get_latest_version('diabetes')
        if registry_info:
            last_training = datetime.fromisoformat(registry_info.get('timestamp', '2000-01-01'))
            if (datetime.now() - last_training).days > 30:
                logger.info("Monthly scheduled retraining triggered")
                return True
        
        return False
    
    def run_retraining_pipeline(self) -> bool:
        """
        Execute the complete retraining pipeline
        """
        logger.info("🔄 Starting automated retraining pipeline")
        
        try:
            # 1. Prepare training data with ground truth
            logger.info("📊 Preparing training data")
            training_data = self.ground_truth.prepare_training_data()
            
            if training_data.empty:
                logger.warning("No training data available")
                return False
            
            # 2. Run training
            logger.info("🤖 Running training pipeline")
            result = subprocess.run([
                "python", "scripts/train.py",
                "--data", "data/ground_truth/retraining_data.parquet",
                "--calibrate", "isotonic",
                "--output", "models/trained/retrained_model.joblib"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Training failed: {result.stderr}")
                return False
            
            # 3. Validate model
            logger.info("✅ Validating model")
            validate_result = subprocess.run([
                "python", "scripts/validate.py",
                "--model", "models/trained/retrained_model.joblib",
                "--threshold", "0.75"
            ], capture_output=True, text=True)
            
            if validate_result.returncode != 0:
                logger.error(f"Validation failed: {validate_result.stderr}")
                return False
            
            # 4. Register model
            logger.info("📦 Registering model")
            register_result = subprocess.run([
                "python", "scripts/register_model.py",
                "--model", "models/trained/retrained_model.joblib",
                "--version", datetime.now().strftime("%Y%m%d_%H%M%S")
            ], capture_output=True, text=True)
            
            if register_result.returncode != 0:
                logger.error(f"Registration failed: {register_result.stderr}")
                return False
            
            # 5. Promote to staging
            logger.info("🚀 Promoting to staging")
            promote_result = subprocess.run([
                "python", "scripts/promote_model.py",
                "--model", "diabetes",
                "--version", datetime.now().strftime("%Y%m%d_%H%M%S"),
                "--stage", "staging"
            ], capture_output=True, text=True)
            
            if promote_result.returncode != 0:
                logger.error(f"Promotion to staging failed: {promote_result.stderr}")
                return False
            
            # 6. Trigger canary deployment
            logger.info("🦅 Triggering canary deployment")
            deploy_result = subprocess.run([
                "python", "scripts/deploy.py",
                "--model", "diabetes",
                "--version", datetime.now().strftime("%Y%m%d_%H%M%S"),
                "--canary", "5"
            ], capture_output=True, text=True)
            
            if deploy_result.returncode != 0:
                logger.error(f"Canary deployment failed: {deploy_result.stderr}")
                return False
            
            logger.info("✅ Retraining pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Retraining pipeline failed: {e}")
            return False
    
    def monitor_and_retrain(self):
        """
        Continuous monitoring and retraining loop
        """
        logger.info("🔄 Starting continuous monitoring and retraining loop")
        
        while True:
            try:
                if self.should_retrain():
                    logger.info("⚠️ Retraining conditions met - starting retraining")
                    success = self.run_retraining_pipeline()
                    
                    if success:
                        logger.info("✅ Retraining successful")
                    else:
                        logger.error("❌ Retraining failed - manual intervention required")
                
                # Wait before next check
                import time
                time.sleep(3600)  # Check every hour
                
            except KeyboardInterrupt:
                logger.info("👋 Stopping monitoring loop")
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(3600)  # Wait and retry

def main():
    manager = AutoRetrainManager()
    
    # Check if should retrain and execute if needed
    if manager.should_retrain():
        print("🔄 Retraining conditions met - executing retraining pipeline")
        success = manager.run_retraining_pipeline()
        sys.exit(0 if success else 1)
    else:
        print("✅ No retraining needed")
        sys.exit(0)

if __name__ == '__main__':
    main()