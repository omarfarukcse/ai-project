# scripts/validate.py
#!/usr/bin/env python
"""
Model Validation Script
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.components.model_registry import ModelRegistry
from src.components.model_evaluation import ModelEvaluator
from src.components.data_ingestion import DataIngestion
from src.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description="Validate CDSS model")
    parser.add_argument(
        "--version",
        type=str,
        help="Model version to validate (production if not specified)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="production",
        choices=["staging", "production", "archived"],
        help="Model stage"
    )
    parser.add_argument(
        "--golden",
        action="store_true",
        help="Run golden tests"
    )
    
    args = parser.parse_args()
    
    logger.info("🔍 Starting model validation")
    
    # Get model
    registry = ModelRegistry()
    
    if args.version:
        model, metadata = registry.load_model("cdss_model", args.version)
    else:
        model, metadata = registry.load_model("cdss_model", stage=args.stage)
    
    logger.info(f"📊 Validating: {metadata.version} ({metadata.stage})")
    
    # Load test data
    ingestion = DataIngestion()
    X_test, y_test = ingestion.load_test_data()
    
    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(model, X_test, y_test)
    
    logger.info(f"📈 Metrics:")
    logger.info(f"   Accuracy: {metrics.accuracy:.3f}")
    logger.info(f"   Recall: {metrics.recall:.3f}")
    logger.info(f"   Precision: {metrics.precision:.3f}")
    logger.info(f"   F1: {metrics.f1_score:.3f}")
    
    # Check performance threshold
    if metrics.recall < 0.7:
        logger.error("❌ Validation failed: Recall < 0.7")
        sys.exit(1)
    
    # Run golden tests
    if args.golden:
        from src.validation.schema_validation import GoldenTestSuite
        
        suite = GoldenTestSuite()
        results = suite.run_tests(lambda x: model.predict_proba(pd.DataFrame([x]))[0][1])
        
        if results.get("failed", 0) > 0:
            logger.error(f"❌ Golden tests failed: {results['failed']} failures")
            sys.exit(1)
        else:
            logger.info(f"✅ Golden tests passed: {results['passed']}/{results['total_tests']}")
    
    logger.info("✅ Validation passed")
    
    return {
        "version": metadata.version,
        "metrics": metrics.to_dict(),
        "passed": True
    }


if __name__ == "__main__":
    main()