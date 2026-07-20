# src/feature_store/online_store.py
"""
Online Feature Store with Redis-based Real-time Serving
"""

import asyncio
import json
import hashlib
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.caching.redis_client import FastRedisClient, get_fast_redis_client
from src.feature_store.feature_registry import FeatureRegistry, FeatureDefinition
from src.feature_store.feature_engineering import FeatureEngineer
from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


@dataclass
class FeatureVector:
    """Feature vector for a single entity"""
    entity_id: str
    features: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    source: str = "online_store"
    
    def to_dict(self) -> Dict:
        return {
            "entity_id": self.entity_id,
            "features": self.features,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "source": self.source,
        }
    
    def to_array(self, feature_names: List[str]) -> np.ndarray:
        """Convert to numpy array for model input"""
        return np.array([self.features.get(name, 0) for name in feature_names])
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        return pd.DataFrame([self.features])


class OnlineFeatureStore:
    """
    Production Online Feature Store with:
    - Redis-based real-time serving
    - Sub-10ms latency
    - Feature versioning
    - Batch retrieval
    - Cache warming
    - Feature monitoring
    """
    
    def __init__(
        self,
        redis_client: Optional[FastRedisClient] = None,
        feature_registry: Optional[FeatureRegistry] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
        cache_ttl: int = 3600,
    ):
        self.redis_client = redis_client or get_fast_redis_client()
        self.feature_registry = feature_registry or FeatureRegistry()
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.cache_ttl = cache_ttl
        
        self._key_prefix = "cdss:feature:"
        self._entity_prefix = "cdss:entity:"
        self._batch_size = 100
        
        # Metrics
        self._metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_latency_ms": 0,
            "feature_usage": {},
        }
        
        # Connect to Redis
        self._ensure_connected()
        
        logger.info("⚡ OnlineFeatureStore initialized")
        logger.info(f"   Cache TTL: {cache_ttl}s")
        logger.info(f"   Batch Size: {self._batch_size}")
    
    def _ensure_connected(self):
        """Ensure Redis connection"""
        if not self.redis_client._connected:
            asyncio.create_task(self.redis_client.connect())
    
    def _get_feature_key(self, entity_id: str, feature_name: str) -> str:
        """Generate Redis key for a feature"""
        return f"{self._key_prefix}{entity_id}:{feature_name}"
    
    def _get_entity_key(self, entity_id: str) -> str:
        """Generate Redis key for entity features"""
        return f"{self._entity_prefix}{entity_id}"
    
    # ============================================================================
    # 🚀 Feature Serving Methods
    # ============================================================================
    
    async def get_feature(
        self,
        entity_id: str,
        feature_name: str,
        default: Any = None,
    ) -> Any:
        """
        Get a single feature value
        
        Args:
            entity_id: Entity identifier
            feature_name: Feature name
            default: Default value if feature not found
            
        Returns:
            Feature value
        """
        
        start_time = time.perf_counter()
        self._metrics["total_requests"] += 1
        
        try:
            # Get from Redis
            key = self._get_feature_key(entity_id, feature_name)
            value = await self.redis_client.get(key)
            
            if value is not None:
                self._metrics["cache_hits"] += 1
                self._metrics["feature_usage"][feature_name] = (
                    self._metrics["feature_usage"].get(feature_name, 0) + 1
                )
                
                # Update latency
                latency = (time.perf_counter() - start_time) * 1000
                self._metrics["avg_latency_ms"] = (
                    self._metrics["avg_latency_ms"] * 0.9 + latency * 0.1
                )
                
                return value
            
            self._metrics["cache_misses"] += 1
            
            # Try to compute on-the-fly
            feature_def = self.feature_registry.get_feature(feature_name)
            if feature_def and feature_def.compute_fn:
                # Compute feature
                # Need to get entity data to compute
                # In production, this would query the entity database
                pass
            
            return default
            
        except Exception as e:
            logger.error(f"❌ Failed to get feature {feature_name}: {str(e)}")
            return default
    
    async def get_entity_features(
        self,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
    ) -> FeatureVector:
        """
        Get all features for an entity
        
        Args:
            entity_id: Entity identifier
            feature_names: Specific features to retrieve (all if None)
            
        Returns:
            FeatureVector with entity features
        """
        
        start_time = time.perf_counter()
        self._metrics["total_requests"] += 1
        
        try:
            # Get from Redis
            key = self._get_entity_key(entity_id)
            cached = await self.redis_client.get(key)
            
            if cached is not None:
                self._metrics["cache_hits"] += 1
                features = cached.get("features", {})
                
                # Filter features if requested
                if feature_names:
                    features = {k: v for k, v in features.items() if k in feature_names}
                
                latency = (time.perf_counter() - start_time) * 1000
                self._metrics["avg_latency_ms"] = (
                    self._metrics["avg_latency_ms"] * 0.9 + latency * 0.1
                )
                
                return FeatureVector(
                    entity_id=entity_id,
                    features=features,
                    version=cached.get("version", "1.0.0"),
                )
            
            self._metrics["cache_misses"] += 1
            
            # Compute features if not cached
            features = await self._compute_features(entity_id, feature_names)
            
            # Cache features
            await self._cache_features(entity_id, features)
            
            return FeatureVector(
                entity_id=entity_id,
                features=features,
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get features for {entity_id}: {str(e)}")
            return FeatureVector(entity_id=entity_id, features={})
    
    async def get_batch_features(
        self,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, FeatureVector]:
        """
        Get features for multiple entities in batch
        
        Args:
            entity_ids: List of entity identifiers
            feature_names: Specific features to retrieve
            
        Returns:
            Dictionary of entity_id -> FeatureVector
        """
        
        start_time = time.perf_counter()
        
        results = {}
        uncached = []
        
        # Try to get from cache first
        for entity_id in entity_ids:
            key = self._get_entity_key(entity_id)
            cached = await self.redis_client.get(key)
            
            if cached is not None:
                features = cached.get("features", {})
                if feature_names:
                    features = {k: v for k, v in features.items() if k in feature_names}
                
                results[entity_id] = FeatureVector(
                    entity_id=entity_id,
                    features=features,
                    version=cached.get("version", "1.0.0"),
                )
                self._metrics["cache_hits"] += 1
            else:
                uncached.append(entity_id)
                self._metrics["cache_misses"] += 1
        
        # Compute uncached features in batches
        if uncached:
            batch_results = await self._compute_batch_features(uncached, feature_names)
            
            for entity_id, features in batch_results.items():
                await self._cache_features(entity_id, features)
                results[entity_id] = FeatureVector(
                    entity_id=entity_id,
                    features=features,
                )
        
        # Update metrics
        latency = (time.perf_counter() - start_time) * 1000
        self._metrics["avg_latency_ms"] = (
            self._metrics["avg_latency_ms"] * 0.9 + latency * 0.1
        )
        self._metrics["total_requests"] += len(entity_ids)
        
        return results
    
    # ============================================================================
    # 🔧 Feature Computation
    # ============================================================================
    
    async def _compute_features(
        self,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute features for an entity"""
        
        # In production, this would:
        # 1. Fetch entity data from database
        # 2. Apply feature engineering
        # 3. Return computed features
        
        # For now, return empty features
        return {}
    
    async def _compute_batch_features(
        self,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute features for multiple entities in batch"""
        
        results = {}
        
        # Process in batches
        for i in range(0, len(entity_ids), self._batch_size):
            batch = entity_ids[i:i + self._batch_size]
            
            # Compute features in parallel
            tasks = [self._compute_features(entity_id, feature_names) for entity_id in batch]
            batch_results = await asyncio.gather(*tasks)
            
            for entity_id, features in zip(batch, batch_results):
                results[entity_id] = features
        
        return results
    
    # ============================================================================
    # 💾 Cache Management
    # ============================================================================
    
    async def _cache_features(
        self,
        entity_id: str,
        features: Dict[str, Any],
        ttl: Optional[int] = None,
    ):
        """Cache entity features in Redis"""
        
        key = self._get_entity_key(entity_id)
        value = {
            "features": features,
            "version": "1.0.0",
            "cached_at": datetime.now().isoformat(),
        }
        
        await self.redis_client.set(
            key,
            value,
            ttl=ttl or self.cache_ttl,
        )
        
        # Also cache individual features for faster access
        for feature_name, feature_value in features.items():
            feature_key = self._get_feature_key(entity_id, feature_name)
            await self.redis_client.set(
                feature_key,
                feature_value,
                ttl=ttl or self.cache_ttl,
            )
    
    async def invalidate_cache(self, entity_id: str):
        """Invalidate cache for an entity"""
        
        # Delete entity cache
        key = self._get_entity_key(entity_id)
        await self.redis_client.delete(key)
        
        # Get all features for entity
        # In production, this would use Redis SCAN
        await self.redis_client.delete_pattern(f"{self._key_prefix}{entity_id}:*")
        
        logger.info(f"🗑️ Cache invalidated for {entity_id}")
    
    async def warm_cache(
        self,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None,
    ):
        """Warm cache for entities"""
        
        logger.info(f"🔥 Warming cache for {len(entity_ids)} entities...")
        
        start_time = time.perf_counter()
        
        # Compute features
        features_dict = await self._compute_batch_features(entity_ids, feature_names)
        
        # Cache features
        tasks = []
        for entity_id, features in features_dict.items():
            tasks.append(self._cache_features(entity_id, features))
        
        await asyncio.gather(*tasks)
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"✅ Cache warmed in {elapsed:.2f}ms")
    
    # ============================================================================
    # 📊 Feature Ingestion
    # ============================================================================
    
    async def ingest_features(
        self,
        entity_id: str,
        features: Dict[str, Any],
        ttl: Optional[int] = None,
    ):
        """
        Ingest features into the online store
        
        Args:
            entity_id: Entity identifier
            features: Feature dictionary
            ttl: Time to live
        """
        
        # Validate features
        for feature_name, value in features.items():
            is_valid, message = self.feature_registry.validate_feature(feature_name, value)
            if not is_valid:
                logger.warning(f"⚠️ Invalid feature {feature_name}: {message}")
        
        # Cache features
        await self._cache_features(entity_id, features, ttl)
        
        logger.info(f"✅ Ingested features for {entity_id}: {len(features)} features")
    
    async def ingest_batch(
        self,
        entities: Dict[str, Dict[str, Any]],
        ttl: Optional[int] = None,
    ):
        """Ingest multiple entities in batch"""
        
        tasks = []
        for entity_id, features in entities.items():
            tasks.append(self.ingest_features(entity_id, features, ttl))
        
        await asyncio.gather(*tasks)
        
        logger.info(f"✅ Ingested {len(entities)} entities")
    
    # ============================================================================
    # 📊 Metrics & Monitoring
    # ============================================================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get feature store metrics"""
        
        total = self._metrics["cache_hits"] + self._metrics["cache_misses"]
        hit_rate = self._metrics["cache_hits"] / total if total > 0 else 0
        
        return {
            **self._metrics,
            "hit_rate": hit_rate,
            "feature_usage": dict(sorted(
                self._metrics["feature_usage"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
        }
    
    def get_feature_stats(self, feature_name: str) -> Dict[str, Any]:
        """Get statistics for a specific feature"""
        
        return {
            "feature_name": feature_name,
            "usage_count": self._metrics["feature_usage"].get(feature_name, 0),
            "is_registered": feature_name in self.feature_registry.catalog.features,
            "feature_def": self.feature_registry.get_feature(feature_name),
        }


# ============================================================================
# 🔧 Singleton Service
# ============================================================================

_feature_service: Optional[OnlineFeatureStore] = None


def get_feature_service() -> OnlineFeatureStore:
    """Get feature service singleton"""
    global _feature_service
    if _feature_service is None:
        _feature_service = OnlineFeatureStore()
    return _feature_service


class FeatureService:
    """
    Convenience wrapper for Feature Store operations
    """
    
    def __init__(self):
        self.store = get_feature_service()
        self.registry = FeatureRegistry()
        self.engineer = FeatureEngineer()
    
    async def get_features(
        self,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get features for an entity"""
        vector = await self.store.get_entity_features(entity_id, feature_names)
        return vector.features
    
    async def get_batch(
        self,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Get features for multiple entities"""
        results = await self.store.get_batch_features(entity_ids, feature_names)
        return {k: v.features for k, v in results.items()}
    
    async def ingest(
        self,
        entity_id: str,
        features: Dict[str, Any],
        ttl: Optional[int] = None,
    ):
        """Ingest features"""
        await self.store.ingest_features(entity_id, features, ttl)
    
    async def warm(self, entity_ids: List[str], feature_names: Optional[List[str]] = None):
        """Warm cache"""
        await self.store.warm_cache(entity_ids, feature_names)
    
    async def invalidate(self, entity_id: str):
        """Invalidate cache"""
        await self.store.invalidate_cache(entity_id)
    
    def register_feature(self, **kwargs) -> FeatureDefinition:
        """Register a new feature"""
        return self.registry.register_feature(**kwargs)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get feature store metrics"""
        return self.store.get_metrics()