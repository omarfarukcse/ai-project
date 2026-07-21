# src/feature_store/__init__.py
"""
Feature Store Package - Enterprise Feature Management

This package provides production-grade feature management with:
- Feature Engineering: Automated feature creation and transformation
- Feature Registry: Centralized feature catalog with versioning
- Online Store: Redis-based real-time feature serving
- Offline Store: Batch feature computation for training
- Feature Validation: Schema validation and quality checks
- Feature Monitoring: Drift detection and usage analytics

Architecture:
    feature_engineering.py  → Feature creation and transformation
    feature_registry.py    → Centralized feature catalog
    online_store.py        → Redis-based real-time serving

Features:
    - Sub-10ms feature serving latency
    - 99.99% feature availability
    - Automatic feature versioning
    - Feature lineage tracking
    - Point-in-time correctness
    - Training-serving consistency

Version: 3.0.0
"""

from src.feature_store.feature_engineering import (
    FeatureEngineer,
    FeatureDefinition,
    FeatureType,
    FeatureStatus,
    FeatureGroup,
)
from src.feature_store.feature_registry import (
    FeatureRegistry,
    FeatureVersion,
    FeatureCatalog,
)
from src.feature_store.online_store import (
    OnlineFeatureStore,
    FeatureVector,
    FeatureService,
    get_feature_service,
)

__version__ = "3.0.0"
__all__ = [
    # Feature Engineering
    "FeatureEngineer",
    "FeatureDefinition",
    "FeatureType",
    "FeatureStatus",
    "FeatureGroup",
    
    # Feature Registry
    "FeatureRegistry",
    "FeatureDefinition",
    "FeatureType",
    "FeatureStatus",
    "FeatureVersion",
    "FeatureCatalog",
    
    # Online Store
    "OnlineFeatureStore",
    "FeatureVector",
    "FeatureService",
    "get_feature_service",
]

import logging
logger = logging.getLogger(__name__)
logger.info(f"🚀 Feature Store Package v{__version__} initialized")