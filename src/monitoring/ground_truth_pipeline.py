# src/monitoring/ground_truth_pipeline.py
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

from src.logger import get_logger
from src.utils.file_utils import FileUtils
from src.feature_store.feature_registry import FeatureRegistry

logger = get_logger(__name__)

class GroundTruthPipeline:
    """
    Continuous learning feedback loop
    Captures ground truth outcomes and feeds back to training
    """
    
    def __init__(self, storage_path: str = "data/ground_truth"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.feature_registry = FeatureRegistry()
        self.ground_truth_history = []
        self._load_history()
    
    def _load_history(self):
        """Load historical ground truth data"""
        history_file = self.storage_path / "ground_truth_history.json"
        if history_file.exists():
            self.ground_truth_history = FileUtils.load_json(history_file)
            logger.info(f"Loaded {len(self.ground_truth_history)} ground truth records")
    
    def record_outcome(self, prediction_id: str, patient: Dict,
                      predicted: Dict, actual_outcome: int,
                      outcome_date: str = None):
        """
        Record actual clinical outcome for a prediction
        """
        record = {
            'prediction_id': prediction_id,
            'patient': patient,
            'predicted': predicted,
            'actual_outcome': actual_outcome,
            'prediction_timestamp': predicted.get('timestamp', datetime.now().isoformat()),
            'outcome_timestamp': outcome_date or datetime.now().isoformat(),
            'ground_truth_available': True,
            'accuracy': 1 if predicted['prediction'] == actual_outcome else 0
        }
        
        self.ground_truth_history.append(record)
        self._save_history()
        
        logger.info(f"Ground truth recorded for prediction {prediction_id}: "
                   f"Actual outcome = {actual_outcome}, Correct = {record['accuracy']}")
        
        # Update feature store
        self._update_feature_store(record)
        
        return record
    
    def _save_history(self):
        """Save ground truth history"""
        history_file = self.storage_path / "ground_truth_history.json"
        FileUtils.save_json(self.ground_truth_history, history_file)
    
    def _update_feature_store(self, record: Dict):
        """Update feature store with ground truth"""
        feature_store_path = self.storage_path / "feature_store"
        feature_store_path.mkdir(parents=True, exist_ok=True)
        
        # Append to feature store
        feature_file = feature_store_path / "ground_truth_features.parquet"
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'prediction_id': record['prediction_id'],
            'actual_outcome': record['actual_outcome'],
            'accuracy': record['accuracy'],
            'prediction_timestamp': record['prediction_timestamp'],
            'outcome_timestamp': record['outcome_timestamp']
        }])
        
        if feature_file.exists():
            existing = pd.read_parquet(feature_file)
            df = pd.concat([existing, df], ignore_index=True)
        
        df.to_parquet(feature_file, index=False)
    
    def prepare_training_data(self) -> pd.DataFrame:
        """
        Prepare data for model retraining with ground truth
        """
        if not self.ground_truth_history:
            logger.warning("No ground truth data available for retraining")
            return pd.DataFrame()
        
        records = [r for r in self.ground_truth_history if r.get('ground_truth_available', False)]
        
        if not records:
            return pd.DataFrame()
        
        # Combine with original features
        training_data = []
        for record in records:
            patient = record['patient']
            patient['actual_outcome'] = record['actual_outcome']
            patient['prediction_id'] = record['prediction_id']
            patient['accuracy'] = record['accuracy']
            training_data.append(patient)
        
        df = pd.DataFrame(training_data)
        
        # Save for retraining
        training_file = self.storage_path / "retraining_data.parquet"
        df.to_parquet(training_file, index=False)
        
        logger.info(f"Prepared {len(df)} samples for retraining")
        return df
    
    def get_performance_trend(self) -> Dict[str, Any]:
        """
        Analyze performance trend over time
        """
        if not self.ground_truth_history:
            return {'message': 'No ground truth data available'}
        
        df = pd.DataFrame(self.ground_truth_history)
        df = df[df['ground_truth_available'] == True]
        
        if df.empty:
            return {'message': 'No ground truth outcomes recorded'}
        
        # Calculate metrics over time
        df['outcome_date'] = pd.to_datetime(df['outcome_timestamp'])
        df['month'] = df['outcome_date'].dt.to_period('M')
        
        monthly_performance = df.groupby('month')['accuracy'].agg(['mean', 'count'])
        
        return {
            'total_records': len(df),
            'overall_accuracy': df['accuracy'].mean(),
            'monthly_performance': monthly_performance.to_dict(),
            'recent_accuracy': df.tail(100)['accuracy'].mean(),
            'trend': 'improving' if monthly_performance['mean'].iloc[-1] > monthly_performance['mean'].iloc[0] else 'declining'
        }
    
    def get_retraining_trigger(self) -> bool:
        """
        Determine if retraining should be triggered
        """
        if not self.ground_truth_history:
            return False
        
        # Check if we have enough recent data
        recent = [r for r in self.ground_truth_history[-100:] 
                 if r.get('ground_truth_available', False)]
        
        if len(recent) < 50:
            return False
        
        # Check if performance has dropped
        performance = self.get_performance_trend()
        if performance.get('recent_accuracy', 1.0) < 0.75:
            logger.warning("Performance drop detected - retraining recommended")
            return True
        
        # Check if it's been a while since last retraining
        if self.feature_registry.get('last_retraining_date'):
            last_retraining = datetime.fromisoformat(
                self.feature_registry.get('last_retraining_date')
            )
            if (datetime.now() - last_retraining).days > 30:
                logger.info("Monthly retraining triggered")
                return True
        
        return False