# src/utils/__init__.py
"""
Utilities Package - Enterprise Utility Functions

This package provides production-grade utilities with:
- File Operations: Async file I/O with compression and encryption
- Model Utilities: Model serialization, versioning, and management
- Metrics Utilities: Performance metrics, calculations, and statistics
- Circuit Breaker: Fault tolerance and resilience patterns
- Retry Logic: Exponential backoff and retry strategies
- Rate Limiting: Distributed rate limiting utilities
- Data Compression: Efficient data serialization and compression
- Secure Storage: Encrypted file operations

Architecture:
    file_utils.py      → File operations with compression/encryption
    model_utils.py     → Model utilities and management
    metrics_utils.py   → Metrics calculations and statistics
    circuit_breaker.py → Circuit breaker pattern for resilience

Version: 3.0.0
"""

from src.utils.file_utils import (
    FileManager,
    AsyncFileManager,
    compress_data,
    decompress_data,
    encrypt_file,
    decrypt_file,
    read_json_async,
    write_json_async,
    read_parquet_async,
    write_parquet_async,
    get_file_hash,
    safe_filename,
    ensure_directory,
    clean_old_files,
    get_file_utils,
)

from src.utils.model_utils import (
    ModelManager,
    ModelSerializer,
    ModelVersion,
    ModelMetadata,
    save_model,
    load_model,
    get_model_version,
    compare_models,
    get_model_size,
    get_model_signature,
    get_model_utils,
)

from src.utils.metrics_utils import (
    MetricsCalculator,
    ClinicalMetrics,
    PerformanceMetrics,
    calculate_accuracy,
    calculate_precision,
    calculate_recall,
    calculate_f1,
    calculate_roc_auc,
    calculate_confusion_matrix,
    calculate_calibration_score,
    calculate_brier_score,
    calculate_ece,
    get_metrics_calculator,
)

from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreakerOpenException,
    get_circuit_breaker,
    circuit_breaker,
    CircuitBreakerMetrics,
    CircuitBreakerRegistry,
)

__version__ = "3.0.0"
__all__ = [
    # File Utils
    "FileManager",
    "AsyncFileManager",
    "compress_data",
    "decompress_data",
    "encrypt_file",
    "decrypt_file",
    "read_json_async",
    "write_json_async",
    "read_parquet_async",
    "write_parquet_async",
    "get_file_hash",
    "safe_filename",
    "ensure_directory",
    "clean_old_files",
    "get_file_utils",
    
    # Model Utils
    "ModelManager",
    "ModelSerializer",
    "ModelVersion",
    "ModelMetadata",
    "save_model",
    "load_model",
    "get_model_version",
    "compare_models",
    "get_model_size",
    "get_model_signature",
    "get_model_utils",
    
    # Metrics Utils
    "MetricsCalculator",
    "ClinicalMetrics",
    "PerformanceMetrics",
    "calculate_accuracy",
    "calculate_precision",
    "calculate_recall",
    "calculate_f1",
    "calculate_roc_auc",
    "calculate_confusion_matrix",
    "calculate_calibration_score",
    "calculate_brier_score",
    "calculate_ece",
    "get_metrics_calculator",
    
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenException",
    "get_circuit_breaker",
    "circuit_breaker",
    "CircuitBreakerMetrics",
    "CircuitBreakerRegistry",
]

import logging
logger = logging.getLogger(__name__)
logger.info(f"🚀 Utilities Package v{__version__} initialized")