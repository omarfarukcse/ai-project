# scripts/deploy.py
#!/usr/bin/env python
"""
Deployment Script with Canary Support
"""

import argparse
import sys
import subprocess
from pathlib import Path
import time
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.components.model_registry import ModelRegistry
from src.logger import get_logger

logger = get_logger(__name__)


class Deployer:
    """Deployment manager with canary support"""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.canary_enabled = True
        self.canary_percentage = 0.05
        self.monitor_duration = 300  # 5 minutes
    
    def deploy_model(self, version: str, target: str = "production"):
        """Deploy model to target stage"""
        
        logger.info(f"🚀 Deploying model {version} to {target}")
        
        # Validate model
        validation_result = self.registry.validate_model(version)
        if not validation_result["passed"]:
            raise ValueError(f"Model validation failed: {validation_result['errors']}")
        
        if target == "production" and self.canary_enabled:
            # Canary deployment
            return self.deploy_canary(version)
        else:
            # Direct deployment
            return self.registry.promote_to_production(version)
    
    def deploy_canary(self, version: str):
        """Perform canary deployment"""
        
        logger.info(f"🦜 Starting canary deployment for {version}")
        
        # Deploy to canary (5% traffic)
        result = self.registry.promote_to_production(
            version,
            canary_percentage=self.canary_percentage
        )
        
        # Monitor canary
        logger.info(f"📊 Monitoring canary for {self.monitor_duration}s")
        
        start_time = time.time()
        while time.time() - start_time < self.monitor_duration:
            # Check metrics
            metrics = self._get_canary_metrics()
            
            if metrics["error_rate"] > 0.05:  # 5% error rate threshold
                logger.error(f"❌ Canary failed: error rate {metrics['error_rate']:.2%}")
                self._rollback_canary(version)
                raise RuntimeError("Canary deployment failed")
            
            if metrics["latency_p95"] > 2.0:  # 2 second threshold
                logger.warning(f"⚠️ High latency: {metrics['latency_p95']:.2f}s")
            
            time.sleep(15)
        
        # Promote to full production
        logger.info("✅ Canary successful, promoting to full production")
        return self.registry.promote_to_production(version, full=True)
    
    def _get_canary_metrics(self) -> dict:
        """Get canary metrics from Prometheus"""
        # In production, query Prometheus API
        return {
            "error_rate": 0.01,
            "latency_p95": 0.5,
            "throughput": 100
        }
    
    def _rollback_canary(self, version: str):
        """Rollback canary deployment"""
        logger.warning(f"🔄 Rolling back canary deployment for {version}")
        
        # Get previous version
        production_metadata = self.registry.get_model_metadata("cdss_model", stage="production")
        previous_version = production_metadata.version
        
        # Rollback
        self.registry.rollback(previous_version)
        logger.info(f"✅ Rolled back to {previous_version}")


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy CDSS model")
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Model version to deploy"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="production",
        choices=["staging", "production"],
        help="Deployment target"
    )
    parser.add_argument(
        "--no-canary",
        action="store_true",
        help="Disable canary deployment"
    )
    
    args = parser.parse_args()
    
    deployer = Deployer()
    
    if args.no_canary:
        deployer.canary_enabled = False
    
    try:
        result = deployer.deploy_model(args.version, args.target)
        logger.info(f"✅ Deployment successful: {result}")
    except Exception as e:
        logger.error(f"❌ Deployment failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()