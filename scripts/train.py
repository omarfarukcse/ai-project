# scripts/train.py
#!/usr/bin/env python
"""
Model Training Script
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.training_pipeline import TrainingPipeline
from src.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train CDSS model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="diabetes",
        help="Dataset type (diabetes, heart_disease)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="cdss_training",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if not needed"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest", "xgboost"],
        help="Models to train"
    )
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting training pipeline for {args.dataset}")
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        experiment_name=args.experiment,
        config_path="config/training_config.yaml"
    )
    
    # Configure models
    pipeline.trainer.model_names = args.models
    
    # Run training
    results = pipeline.run()
    
    logger.info(f"✅ Training complete: {results['status']}")
    logger.info(f"   Best Model: {results.get('best_model')}")
    logger.info(f"   Recall: {results.get('metrics', {}).get('recall', 0):.3f}")
    
    return results


if __name__ == "__main__":
    main()