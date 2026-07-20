# experiments/mlflow/__init__.py
"""
MLflow Experiment Tracking Configuration
"""

import mlflow
from mlflow.tracking import MlflowClient
import os
from pathlib import Path

def setup_mlflow():
    """Setup MLflow tracking"""
    
    # Set tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./experiments/mlflow")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create experiment if not exists
    experiment_name = "cdss_healthcare"
    client = MlflowClient()
    
    try:
        experiment_id = client.create_experiment(experiment_name)
    except:
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    return mlflow, client, experiment_id

__all__ = ["setup_mlflow"]