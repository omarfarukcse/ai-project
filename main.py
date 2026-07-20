#!/usr/bin/env python3
"""
Clinical Decision Support System - Main Entry Point
FAANG-Level Production Implementation
"""
import argparse
import sys
from pathlib import Path

from src.logger import get_logger
from src.config_manager import config_manager
from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.inference_pipeline import InferencePipeline
from src.utils.file_utils import FileUtils

logger = get_logger(__name__)

def main():
    """Main entry point for CDSS application"""
    parser = argparse.ArgumentParser(description="Clinical Decision Support System")
    parser.add_argument('--mode', choices=['train', 'predict', 'explain', 'serve'], 
                       default='serve', help='Operation mode')
    parser.add_argument('--config', type=str, default='config/main_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--model', type=str, help='Model path for inference')
    parser.add_argument('--patient', type=str, help='Patient data file for prediction')
    parser.add_argument('--patient_id', type=str, help='Patient ID for single prediction')
    
    args = parser.parse_args()
    
    # Load configuration
    config = config_manager.config
    
    if args.mode == 'train':
        logger.info("🔄 Starting training mode")
        pipeline = TrainingPipeline(config)
        results = pipeline.run()
        logger.info("✅ Training completed successfully")
        
    elif args.mode == 'serve':
        logger.info("🚀 Starting API server")
        import uvicorn
        from src.api.app import app
        
        api_config = config_manager.get_validated('api')
        uvicorn.run(
            app,
            host=api_config.host,
            port=api_config.port,
            workers=api_config.workers,
            reload=False,
            log_level="info"
        )
        
    elif args.mode == 'predict':
        logger.info("🔍 Starting prediction mode")
        pipeline = InferencePipeline(config, args.model)
        
        if args.patient:
            import pandas as pd
            patient_data = pd.read_csv(args.patient)
            results = pipeline.predict_batch(patient_data)
            print("\n📊 Prediction Results:")
            print(results)
        else:
            logger.error("Please provide patient data for prediction")
    
    elif args.mode == 'explain':
        logger.info("🔍 Starting explanation mode")
        pipeline = InferencePipeline(config, args.model)
        
        if args.patient:
            import pandas as pd
            patient_data = pd.read_csv(args.patient)
            if len(patient_data) > 0:
                result = pipeline.explain_prediction(patient_data.iloc[0].to_dict())
                print("\n📋 Prediction Explanation:")
                print(result)
        else:
            logger.error("Please provide patient data for explanation")

if __name__ == "__main__":
    main()