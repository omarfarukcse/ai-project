# src/api/dependencies.py
"""
Enterprise-Grade Dependency Injection
- Lazy loading
- Caching
- Security checks
- Performance optimization
"""

from functools import lru_cache
from typing import Optional, Any, Dict
from fastapi import Request, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import pandas as pd
import numpy as np

from src.components.model_registry import ModelRegistry
from src.caching.redis_client import get_fast_redis_client
from src.monitoring.drift_detection import DriftDetector
from src.feature_store.online_store import FeatureStore
from src.security.auth import AuthManager
from src.security.audit import AuditLogger
from src.utils.circuit_breaker import get_circuit_breaker
from src.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# 🔐 Security Dependencies
# ============================================================================

security = HTTPBearer()
auth_manager = AuthManager()


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    Get current authenticated user
    
    **Security:**
    - JWT token validation
    - Token expiration check
    - User permissions
    """
    
    token = credentials.credentials
    
    try:
        user = await auth_manager.validate_token(token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check token expiry
        if user.get("expires_at") < pd.Timestamp.now():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user
        
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_permission(permission: str):
    """Dependency factory for permission checks"""
    async def check_permission(
        current_user: Dict = Depends(get_current_user),
    ) -> bool:
        if permission not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required",
            )
        return True
    return check_permission


# ============================================================================
# 🧠 Model Dependencies
# ============================================================================

@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    """Get model registry instance (cached)"""
    return ModelRegistry()


async def get_model(
    request: Request,
    model_registry: ModelRegistry = Depends(get_model_registry),
) -> Any:
    """
    Get the production model
    
    **Features:**
    - Lazy loading
    - Cache invalidation on version change
    - Fallback to staging if production unavailable
    """
    
    # Check if model is in request state (per request cache)
    if hasattr(request.state, "model"):
        return request.state.model
    
    try:
        # Get from registry
        model = await model_registry.get_production_model()
        request.state.model = model
        return model
        
    except Exception as e:
        logger.error(f"Failed to load production model: {str(e)}")
        
        # Try staging model as fallback
        try:
            model = await model_registry.get_staging_model()
            logger.warning("Using staging model as fallback")
            request.state.model = model
            return model
        except Exception as e2:
            logger.error(f"Failed to load staging model: {str(e2)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model unavailable",
            )


# ============================================================================
# 💾 Storage Dependencies
# ============================================================================

@lru_cache(maxsize=1)
def get_redis_client() -> FastRedisClient:
    """Get Redis client (cached)"""
    return get_fast_redis_client()


@lru_cache(maxsize=1)
def get_feature_store() -> FeatureStore:
    """Get feature store (cached)"""
    return FeatureStore()


# ============================================================================
# 📊 Monitoring Dependencies
# ============================================================================

@lru_cache(maxsize=1)
def get_drift_detector() -> DriftDetector:
    """Get drift detector (cached)"""
    return DriftDetector()


@lru_cache(maxsize=1)
def get_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker (cached)"""
    return get_circuit_breaker()


@lru_cache(maxsize=1)
def get_audit_logger() -> AuditLogger:
    """Get audit logger (cached)"""
    return AuditLogger()


# ============================================================================
# 🔧 Utility Dependencies
# ============================================================================

async def get_pagination(
    page: int = 1,
    page_size: int = 50,
) -> tuple:
    """Get pagination parameters"""
    page = max(1, page)
    page_size = min(1000, max(1, page_size))
    offset = (page - 1) * page_size
    return page, page_size, offset


async def get_sorting(
    sort_by: Optional[str] = None,
    sort_order: str = "asc",
) -> tuple:
    """Get sorting parameters"""
    sort_order = sort_order.lower()
    if sort_order not in ["asc", "desc"]:
        sort_order = "asc"
    return sort_by, sort_order


# ============================================================================
# 📊 Metrics Dependencies
# ============================================================================

from prometheus_client import Counter, Histogram

# Prediction metrics
PREDICTION_COUNTER = Counter(
    "cdss_predictions_total",
    "Total predictions",
    ["model", "risk_level", "drift"]
)

PREDICTION_LATENCY = Histogram(
    "cdss_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model", "batch"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

FEATURE_COUNTER = Counter(
    "cdss_features_used_total",
    "Feature usage count",
    ["feature"]
)


async def record_prediction_metrics(
    model_version: str,
    risk_level: str,
    drift_detected: bool,
    latency: float,
    batch_size: int = 1,
):
    """Record prediction metrics"""
    PREDICTION_COUNTER.labels(
        model=model_version,
        risk_level=risk_level,
        drift=str(drift_detected),
    ).inc()
    
    PREDICTION_LATENCY.labels(
        model=model_version,
        batch="yes" if batch_size > 1 else "no",
    ).observe(latency)


# ============================================================================
# 🚀 Exports
# ============================================================================

__all__ = [
    "get_model",
    "get_redis_client",
    "get_drift_detector",
    "get_circuit_breaker",
    "get_feature_store",
    "get_current_user",
    "require_permission",
    "get_audit_logger",
    "get_model_registry",
    "get_pagination",
    "get_sorting",
    "record_prediction_metrics",
]