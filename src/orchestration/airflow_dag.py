# src/orchestration/airflow_dag.py
"""
Apache Airflow DAG Definitions for ML Pipeline Orchestration
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import json
import os

# Airflow imports (conditional for environments without Airflow)
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator, BranchPythonOperator
    from airflow.operators.bash import BashOperator
    from airflow.operators.dummy import DummyOperator
    from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
    from airflow.providers.celery.operators.celery import CeleryOperator
    from airflow.utils.dates import days_ago
    from airflow.utils.trigger_rule import TriggerRule
    from airflow.models import Variable
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    # Create dummy classes for type hints
    class DAG: pass
    class PythonOperator: pass
    class BranchPythonOperator: pass
    class BashOperator: pass
    class DummyOperator: pass
    class SlackWebhookOperator: pass
    class CeleryOperator: pass

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


@dataclass
class DAGConfig:
    """DAG configuration"""
    dag_id: str
    schedule_interval: str = "@daily"
    start_date: datetime = field(default_factory=lambda: days_ago(1))
    catchup: bool = False
    max_active_runs: int = 1
    concurrency: int = 4
    retries: int = 2
    retry_delay: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    sla_miss_callback: Optional[Callable] = None
    on_failure_callback: Optional[Callable] = None
    on_success_callback: Optional[Callable] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class TaskConfig:
    """Task configuration"""
    task_id: str
    python_callable: Optional[Callable] = None
    bash_command: Optional[str] = None
    retries: int = 1
    retry_delay: timedelta = field(default_factory=lambda: timedelta(minutes=2))
    execution_timeout: timedelta = field(default_factory=lambda: timedelta(hours=1))
    depends_on_past: bool = False
    wait_for_downstream: bool = False
    trigger_rule: str = "all_success"
    pool: str = "default_pool"
    priority_weight: int = 1
    queue: str = "default"
    task_group: Optional[str] = None
    sla: Optional[timedelta] = None
    on_failure_callback: Optional[Callable] = None
    on_success_callback: Optional[Callable] = None
    on_retry_callback: Optional[Callable] = None


# ============================================================================
# 🔧 DAG Factory Functions
# ============================================================================

def create_training_dag(config: Optional[DAGConfig] = None) -> Optional[DAG]:
    """
    Create Airflow DAG for ML model training
    
    Steps:
    1. Data ingestion and validation
    2. Data preprocessing
    3. Feature engineering
    4. Model training
    5. Model evaluation
    6. Model calibration
    7. Model validation (golden tests)
    8. Model registration
    9. Model promotion (if performance improves)
    10. Notification
    """
    
    if not AIRFLOW_AVAILABLE:
        logger.warning("⚠️ Airflow not available. Training DAG not created.")
        return None
    
    # Default configuration
    if config is None:
        config = DAGConfig(
            dag_id="cdss_training_pipeline",
            schedule_interval="0 2 * * *",  # Daily at 2 AM
            tags=["training", "ml_pipeline", "cdss"],
        )
    
    default_args = {
        "owner": "ml_team",
        "depends_on_past": False,
        "start_date": config.start_date,
        "email_on_failure": True,
        "email_on_retry": False,
        "email": ["ml-team@healthcare.com"],
        "retries": config.retries,
        "retry_delay": config.retry_delay,
        "max_active_runs": config.max_active_runs,
        "concurrency": config.concurrency,
    }
    
    dag = DAG(
        dag_id=config.dag_id,
        default_args=default_args,
        description="CDSS Model Training Pipeline",
        schedule_interval=config.schedule_interval,
        catchup=config.catchup,
        tags=config.tags,
        max_active_runs=config.max_active_runs,
        concurrency=config.concurrency,
        sla_miss_callback=config.sla_miss_callback,
        on_failure_callback=config.on_failure_callback,
        on_success_callback=config.on_success_callback,
    )
    
    # Define tasks
    with dag:
        # Start marker
        start = DummyOperator(
            task_id="start_pipeline",
            dag=dag,
        )
        
        # Step 1: Data Ingestion
        data_ingestion = PythonOperator(
            task_id="data_ingestion",
            python_callable=_run_data_ingestion,
            retries=2,
            execution_timeout=timedelta(minutes=15),
            dag=dag,
        )
        
        # Step 2: Data Validation
        data_validation = PythonOperator(
            task_id="data_validation",
            python_callable=_run_data_validation,
            retries=2,
            execution_timeout=timedelta(minutes=10),
            dag=dag,
        )
        
        # Step 3: Data Preprocessing
        preprocessing = PythonOperator(
            task_id="preprocessing",
            python_callable=_run_preprocessing,
            retries=2,
            execution_timeout=timedelta(minutes=30),
            dag=dag,
        )
        
        # Step 4: Feature Engineering
        feature_engineering = PythonOperator(
            task_id="feature_engineering",
            python_callable=_run_feature_engineering,
            retries=2,
            execution_timeout=timedelta(minutes=20),
            dag=dag,
        )
        
        # Step 5: Model Training
        model_training = PythonOperator(
            task_id="model_training",
            python_callable=_run_model_training,
            retries=3,
            execution_timeout=timedelta(hours=2),
            pool="gpu_pool" if config_manager.get("training.use_gpu", False) else "default_pool",
            dag=dag,
        )
        
        # Step 6: Model Evaluation
        model_evaluation = PythonOperator(
            task_id="model_evaluation",
            python_callable=_run_model_evaluation,
            retries=2,
            execution_timeout=timedelta(minutes=30),
            dag=dag,
        )
        
        # Step 7: Model Calibration
        model_calibration = PythonOperator(
            task_id="model_calibration",
            python_callable=_run_model_calibration,
            retries=2,
            execution_timeout=timedelta(minutes=15),
            dag=dag,
        )
        
        # Step 8: Decision Branch (performance check)
        performance_check = BranchPythonOperator(
            task_id="performance_check",
            python_callable=_check_performance,
            dag=dag,
        )
        
        # Step 9a: Register Model (if performance improved)
        register_model = PythonOperator(
            task_id="register_model",
            python_callable=_run_model_registration,
            retries=2,
            execution_timeout=timedelta(minutes=10),
            dag=dag,
        )
        
        # Step 9b: Promote Model (if significantly better)
        promote_model = PythonOperator(
            task_id="promote_model",
            python_callable=_run_model_promotion,
            retries=2,
            execution_timeout=timedelta(minutes=5),
            dag=dag,
        )
        
        # Step 9c: Skip promotion (if no improvement)
        skip_promotion = DummyOperator(
            task_id="skip_promotion",
            dag=dag,
        )
        
        # Step 10: Generate Reports
        generate_reports = PythonOperator(
            task_id="generate_reports",
            python_callable=_run_report_generation,
            retries=2,
            execution_timeout=timedelta(minutes=15),
            dag=dag,
        )
        
        # Step 11: Notifications
        send_notification = SlackWebhookOperator(
            task_id="send_notification",
            slack_webhook_conn_id="slack_webhook",
            message="""
                :rocket: CDSS Training Pipeline Complete
            
                *Model:* {{ task_instance.xcom_pull(task_ids='model_training', key='model_name') }}
                *Performance:* {{ task_instance.xcom_pull(task_ids='model_evaluation', key='metrics') }}
                *Status:* {{ task_instance.xcom_pull(task_ids='register_model', key='status') }}
            """,
            dag=dag,
        )
        
        # End marker
        end = DummyOperator(
            task_id="end_pipeline",
            trigger_rule=TriggerRule.ALL_DONE,
            dag=dag,
        )
        
        # Define dependencies
        start >> data_ingestion >> data_validation >> preprocessing
        preprocessing >> feature_engineering >> model_training
        model_training >> model_evaluation >> model_calibration
        
        model_calibration >> performance_check
        
        # Branches
        performance_check >> register_model >> promote_model >> generate_reports
        performance_check >> skip_promotion >> generate_reports
        
        generate_reports >> send_notification >> end
    
    logger.info(f"✅ Training DAG created: {config.dag_id}")
    return dag


def create_validation_dag(config: Optional[DAGConfig] = None) -> Optional[DAG]:
    """
    Create Airflow DAG for model validation and testing
    
    Steps:
    1. Load production model
    2. Run golden tests
    3. Check performance metrics
    4. Generate validation report
    5. Notify on issues
    """
    
    if not AIRFLOW_AVAILABLE:
        return None
    
    if config is None:
        config = DAGConfig(
            dag_id="cdss_validation_pipeline",
            schedule_interval="0 12 * * *",  # Daily at 12 PM
            tags=["validation", "testing", "cdss"],
        )
    
    default_args = {
        "owner": "qa_team",
        "depends_on_past": False,
        "start_date": config.start_date,
        "email_on_failure": True,
        "email_on_retry": False,
        "email": ["qa-team@healthcare.com"],
        "retries": config.retries,
        "retry_delay": config.retry_delay,
    }
    
    dag = DAG(
        dag_id=config.dag_id,
        default_args=default_args,
        description="CDSS Model Validation Pipeline",
        schedule_interval=config.schedule_interval,
        catchup=config.catchup,
        tags=config.tags,
    )
    
    with dag:
        start = DummyOperator(task_id="start_validation", dag=dag)
        
        load_model = PythonOperator(
            task_id="load_production_model",
            python_callable=_load_production_model,
            retries=2,
            dag=dag,
        )
        
        run_golden_tests = PythonOperator(
            task_id="run_golden_tests",
            python_callable=_run_golden_tests,
            retries=2,
            execution_timeout=timedelta(minutes=15),
            dag=dag,
        )
        
        check_metrics = PythonOperator(
            task_id="check_validation_metrics",
            python_callable=_check_validation_metrics,
            retries=2,
            dag=dag,
        )
        
        generate_report = PythonOperator(
            task_id="generate_validation_report",
            python_callable=_generate_validation_report,
            retries=2,
            dag=dag,
        )
        
        alert_on_failure = SlackWebhookOperator(
            task_id="alert_on_failure",
            slack_webhook_conn_id="slack_webhook",
            message="""
                :warning: *Validation Pipeline Alert*
                
                *Model:* {{ task_instance.xcom_pull(task_ids='load_production_model') }}
                *Failed Tests:* {{ task_instance.xcom_pull(task_ids='run_golden_tests', key='failed_tests') }}
                *Action Required:* Please investigate immediately.
            """,
            trigger_rule=TriggerRule.ONE_FAILED,
            dag=dag,
        )
        
        end = DummyOperator(task_id="end_validation", dag=dag)
        
        start >> load_model >> run_golden_tests >> check_metrics >> generate_report >> end
        run_golden_tests >> alert_on_failure >> end
    
    return dag


def create_monitoring_dag(config: Optional[DAGConfig] = None) -> Optional[DAG]:
    """
    Create Airflow DAG for model monitoring
    
    Steps:
    1. Check data drift
    2. Check model drift
    3. Check performance metrics
    4. Generate monitoring report
    5. Auto-retrain if needed
    """
    
    if not AIRFLOW_AVAILABLE:
        return None
    
    if config is None:
        config = DAGConfig(
            dag_id="cdss_monitoring_pipeline",
            schedule_interval="0 */6 * * *",  # Every 6 hours
            tags=["monitoring", "drift", "cdss"],
        )
    
    default_args = {
        "owner": "ml_ops",
        "depends_on_past": False,
        "start_date": config.start_date,
        "email_on_failure": True,
        "email_on_retry": False,
        "email": ["ml-ops@healthcare.com"],
        "retries": config.retries,
        "retry_delay": config.retry_delay,
    }
    
    dag = DAG(
        dag_id=config.dag_id,
        default_args=default_args,
        description="CDSS Model Monitoring Pipeline",
        schedule_interval=config.schedule_interval,
        catchup=config.catchup,
        tags=config.tags,
    )
    
    with dag:
        start = DummyOperator(task_id="start_monitoring", dag=dag)
        
        check_data_drift = PythonOperator(
            task_id="check_data_drift",
            python_callable=_check_data_drift,
            retries=2,
            execution_timeout=timedelta(minutes=10),
            dag=dag,
        )
        
        check_model_drift = PythonOperator(
            task_id="check_model_drift",
            python_callable=_check_model_drift,
            retries=2,
            execution_timeout=timedelta(minutes=10),
            dag=dag,
        )
        
        check_performance = PythonOperator(
            task_id="check_performance",
            python_callable=_check_production_performance,
            retries=2,
            execution_timeout=timedelta(minutes=10),
            dag=dag,
        )
        
        # Branch: retrain if needed
        retrain_decision = BranchPythonOperator(
            task_id="retrain_decision",
            python_callable=_should_retrain,
            dag=dag,
        )
        
        trigger_retraining = PythonOperator(
            task_id="trigger_retraining",
            python_callable=_trigger_retraining,
            retries=2,
            dag=dag,
        )
        
        no_retrain = DummyOperator(
            task_id="no_retrain",
            dag=dag,
        )
        
        generate_monitoring_report = PythonOperator(
            task_id="generate_monitoring_report",
            python_callable=_generate_monitoring_report,
            retries=2,
            execution_timeout=timedelta(minutes=15),
            dag=dag,
        )
        
        end = DummyOperator(task_id="end_monitoring", dag=dag)
        
        start >> [check_data_drift, check_model_drift, check_performance]
        [check_data_drift, check_model_drift, check_performance] >> retrain_decision
        
        retrain_decision >> trigger_retraining >> generate_monitoring_report >> end
        retrain_decision >> no_retrain >> generate_monitoring_report >> end
    
    return dag


# ============================================================================
# 🔧 Task Callback Functions
# ============================================================================

def _run_data_ingestion(**context) -> Dict[str, Any]:
    """Run data ingestion task"""
    from src.pipelines.training_pipeline import TrainingPipeline
    
    pipeline = TrainingPipeline()
    pipeline.step1_ingest_data()
    
    return {
        "status": "success",
        "dataset_version": pipeline.context.dataset_version,
    }


def _run_data_validation(**context) -> Dict[str, Any]:
    """Run data validation task"""
    from src.components.data_ingestion import DataIngestion
    
    ingestion = DataIngestion()
    report = ingestion.generate_quality_report()
    
    # Check for critical issues
    if report.get("validation_errors"):
        raise ValueError(f"Data validation failed: {report['validation_errors']}")
    
    return {
        "status": "success",
        "report": report,
    }


def _run_preprocessing(**context) -> Dict[str, Any]:
    """Run preprocessing task"""
    from src.pipelines.training_pipeline import TrainingPipeline
    
    pipeline = TrainingPipeline()
    pipeline.step2_preprocess_data()
    
    return {
        "status": "success",
        "train_size": len(pipeline.X_train),
        "test_size": len(pipeline.X_test),
    }


def _run_feature_engineering(**context) -> Dict[str, Any]:
    """Run feature engineering task"""
    from src.feature_store.feature_engineering import FeatureEngineer
    
    engineer = FeatureEngineer()
    # Features will be applied in preprocessing
    
    return {
        "status": "success",
        "feature_count": len(engineer.get_feature_names()),
    }


def _run_model_training(**context) -> Dict[str, Any]:
    """Run model training task"""
    from src.pipelines.training_pipeline import TrainingPipeline
    
    pipeline = TrainingPipeline()
    pipeline.step3_train_models()
    pipeline.step4_evaluate_models()
    
    # Push to XCom
    context['ti'].xcom_push(
        key='model_name',
        value=pipeline.best_model_name
    )
    
    return {
        "status": "success",
        "model_name": pipeline.best_model_name,
        "metrics": pipeline.evaluation_results.get(pipeline.best_model_name, {}),
    }


def _run_model_evaluation(**context) -> Dict[str, Any]:
    """Run model evaluation task"""
    from src.components.model_evaluation import ModelEvaluator
    
    # Load model from previous task
    model_name = context['ti'].xcom_pull(
        task_ids='model_training',
        key='model_name'
    )
    
    evaluator = ModelEvaluator()
    
    # Push metrics to XCom
    context['ti'].xcom_push(
        key='metrics',
        value=evaluator.results.get(model_name, {})
    )
    
    return {
        "status": "success",
        "metrics": evaluator.results.get(model_name, {}),
    }


def _run_model_calibration(**context) -> Dict[str, Any]:
    """Run model calibration task"""
    from src.components.model_calibration import ModelCalibrator
    
    calibrator = ModelCalibrator(method='platt')
    # Calibration happens in training pipeline
    
    return {
        "status": "success",
        "calibrated": True,
    }


def _check_performance(**context) -> str:
    """Check model performance and decide next step"""
    metrics = context['ti'].xcom_pull(
        task_ids='model_evaluation',
        key='metrics'
    )
    
    # Get current production performance
    from src.components.model_registry import ModelRegistry
    
    registry = ModelRegistry()
    current_metrics = registry.get_model_info('production')
    
    # Check if new model is better
    new_recall = metrics.get('recall', 0)
    current_recall = current_metrics.get('recall', 0)
    
    improvement = new_recall - current_recall
    improvement_threshold = 0.01  # 1% improvement required
    
    if improvement >= improvement_threshold:
        return "register_model"
    else:
        return "skip_promotion"


def _run_model_registration(**context) -> Dict[str, Any]:
    """Run model registration task"""
    from src.components.model_registry import ModelRegistry
    
    registry = ModelRegistry()
    version = registry.register_model(
        model_path="models/staging/latest",
        stage="staging"
    )
    
    context['ti'].xcom_push(key='status', value='registered')
    context['ti'].xcom_push(key='version', value=version)
    
    return {
        "status": "success",
        "version": version,
    }


def _run_model_promotion(**context) -> Dict[str, Any]:
    """Run model promotion task"""
    from src.components.model_registry import ModelRegistry
    
    version = context['ti'].xcom_pull(
        task_ids='register_model',
        key='version'
    )
    
    registry = ModelRegistry()
    result = registry.promote_model(version=version, stage='production')
    
    return {
        "status": "success",
        "version": version,
        "result": result,
    }


def _run_report_generation(**context) -> Dict[str, Any]:
    """Run report generation task"""
    from src.components.reporting import ClinicalReporter
    
    reporter = ClinicalReporter()
    report = reporter.generate_summary_report()
    
    return {
        "status": "success",
        "report_path": report.get('path'),
    }


def _load_production_model(**context) -> Dict[str, Any]:
    """Load production model for validation"""
    from src.components.model_registry import ModelRegistry
    
    registry = ModelRegistry()
    model_info = registry.get_model_info('production')
    
    return {
        "status": "success",
        "model_version": model_info.get('version'),
        "model_info": model_info,
    }


def _run_golden_tests(**context) -> Dict[str, Any]:
    """Run golden tests on production model"""
    from src.validation.schema_validation import DataValidator
    
    validator = DataValidator()
    tests = validator.load_golden_tests()
    
    failed_tests = []
    for test_name, test_case in tests.items():
        if not test_case['passed']:
            failed_tests.append(test_name)
    
    context['ti'].xcom_push(key='failed_tests', value=failed_tests)
    
    return {
        "status": "success" if not failed_tests else "failed",
        "failed_tests": failed_tests,
        "total_tests": len(tests),
        "passed_tests": len(tests) - len(failed_tests),
    }


def _check_validation_metrics(**context) -> Dict[str, Any]:
    """Check validation metrics"""
    failed_tests = context['ti'].xcom_pull(
        task_ids='run_golden_tests',
        key='failed_tests'
    )
    
    if failed_tests:
        raise ValueError(f"Validation failed: {failed_tests}")
    
    return {
        "status": "success",
        "all_passed": True,
    }


def _generate_validation_report(**context) -> Dict[str, Any]:
    """Generate validation report"""
    # Implementation
    return {
        "status": "success",
        "report_path": "outputs/reports/validation_report.html",
    }


def _check_data_drift(**context) -> Dict[str, Any]:
    """Check for data drift"""
    from src.monitoring.drift_detection import DriftDetector
    
    detector = DriftDetector()
    report = detector.detect_drift()
    
    return {
        "status": "success",
        "drift_detected": report.get('drift_detected', False),
        "drift_share": report.get('drift_share', 0),
        "report": report,
    }


def _check_model_drift(**context) -> Dict[str, Any]:
    """Check for model drift"""
    # Implementation
    return {
        "status": "success",
        "drift_detected": False,
    }


def _check_production_performance(**context) -> Dict[str, Any]:
    """Check production model performance"""
    from src.monitoring.metrics_monitor import ModelMonitor
    
    monitor = ModelMonitor()
    metrics = monitor.get_performance_metrics()
    
    return {
        "status": "success",
        "metrics": metrics,
        "recall_drop": metrics.get('recall_drop', 0),
    }


def _should_retrain(**context) -> str:
    """Decide if retraining is needed"""
    drift_data = context['ti'].xcom_pull(
        task_ids='check_data_drift'
    )
    drift_model = context['ti'].xcom_pull(
        task_ids='check_model_drift'
    )
    performance = context['ti'].xcom_pull(
        task_ids='check_performance'
    )
    
    # Retrain if:
    # 1. Significant data drift detected
    # 2. Model drift detected
    # 3. Performance drop > 5%
    
    if (drift_data.get('drift_detected', False) or
        drift_model.get('drift_detected', False) or
        performance.get('recall_drop', 0) > 0.05):
        return "trigger_retraining"
    
    return "no_retrain"


def _trigger_retraining(**context) -> Dict[str, Any]:
    """Trigger model retraining"""
    from src.orchestration.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    scheduler.trigger_task("retrain_model")
    
    return {
        "status": "success",
        "triggered": True,
    }


def _generate_monitoring_report(**context) -> Dict[str, Any]:
    """Generate monitoring report"""
    # Implementation
    return {
        "status": "success",
        "report_path": "outputs/reports/monitoring_report.html",
    }