# src/caching/cache_manager.py
"""
Multi-Level Cache Manager with Advanced Strategies
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
import hashlib
import json
from collections import OrderedDict

from src.logger import get_logger
from src.caching.redis_client import FastRedisClient, get_fast_redis_client

logger = get_logger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"      # Least Recently Used
    LFU = "lfu"      # Least Frequently Used
    FIFO = "fifo"    # First In First Out
    TTL = "ttl"      # Time To Live
    AUTO = "auto"    # Adaptive strategy


class CacheLevel(Enum):
    """Cache levels"""
    L1 = "l1"        # Memory cache (fastest)
    L2 = "l2"        # Redis cache (distributed)
    L3 = "l3"        # Database (slowest)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[int] = None
    size: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access metadata"""
        self.accessed_at = time.time()
        self.access_count += 1


class L1Cache:
    """
    In-memory L1 cache with LRU eviction
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._metrics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0,
        }
        
        logger.info(f"📦 L1 Cache initialized: max_size={max_size}, ttl={default_ttl}s")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from L1 cache"""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._metrics["misses"] += 1
                return None
            
            if entry.is_expired():
                del self._cache[key]
                self._metrics["misses"] += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            entry.touch()
            
            self._metrics["hits"] += 1
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in L1 cache"""
        async with self._lock:
            # Evict if full
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Remove oldest (LRU)
                oldest = next(iter(self._cache))
                del self._cache[oldest]
                self._metrics["evictions"] += 1
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.default_ttl,
                size=len(str(value)) if value else 0,
            )
            
            self._cache[key] = entry
            self._cache.move_to_end(key)
            self._metrics["size"] = len(self._cache)
    
    async def delete(self, key: str):
        """Delete from L1 cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._metrics["size"] = len(self._cache)
    
    async def clear(self):
        """Clear L1 cache"""
        async with self._lock:
            self._cache.clear()
            self._metrics["size"] = 0
            self._metrics["evictions"] = 0
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        total = self._metrics["hits"] + self._metrics["misses"]
        return {
            **self._metrics,
            "hit_rate": self._metrics["hits"] / total if total > 0 else 0,
            "miss_rate": self._metrics["misses"] / total if total > 0 else 0,
            "size": len(self._cache),
            "max_size": self.max_size,
        }


class MultiLevelCache:
    """
    Multi-level cache with L1 (memory) and L2 (Redis)
    """
    
    def __init__(
        self,
        redis_client: Optional[FastRedisClient] = None,
        l1_max_size: int = 1000,
        l1_ttl: int = 300,
        l2_ttl: int = 3600,
        fallback_to_l2: bool = True,
    ):
        self.redis_client = redis_client or get_fast_redis_client()
        self.l1_cache = L1Cache(max_size=l1_max_size, default_ttl=l1_ttl)
        self.l2_ttl = l2_ttl
        self.fallback_to_l2 = fallback_to_l2
        
        self._metrics = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
        }
        
        logger.info("🏗️ MultiLevelCache initialized")
    
    async def get(self, key: str, skip_l1: bool = False) -> Optional[Any]:
        """Get value from cache (L1 -> L2)"""
        
        if not skip_l1:
            # Try L1
            value = await self.l1_cache.get(key)
            if value is not None:
                self._metrics["l1_hits"] += 1
                CACHE_METRICS["cache_hits"].labels(cache_level="l1").inc()
                return value
            self._metrics["l1_misses"] += 1
        
        # Try L2 (Redis)
        value = await self.redis_client.get(key)
        if value is not None:
            self._metrics["l2_hits"] += 1
            CACHE_METRICS["cache_hits"].labels(cache_level="l2").inc()
            
            # Populate L1
            await self.l1_cache.set(key, value)
            return value
        
        self._metrics["l2_misses"] += 1
        CACHE_METRICS["cache_misses"].labels(cache_level="l2").inc()
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        skip_l1: bool = False,
        skip_l2: bool = False,
    ):
        """Set value in cache"""
        
        if not skip_l1:
            await self.l1_cache.set(key, value, ttl=ttl or 300)
        
        if not skip_l2:
            await self.redis_client.set(key, value, ttl=ttl or self.l2_ttl)
    
    async def delete(self, key: str):
        """Delete from all cache levels"""
        await self.l1_cache.delete(key)
        await self.redis_client.delete(key)
    
    async def clear(self):
        """Clear all cache"""
        await self.l1_cache.clear()
        # Don't clear Redis globally
    
    async def get_or_set(
        self,
        key: str,
        func: Callable,
        ttl: Optional[int] = None,
        skip_l1: bool = False,
    ) -> Any:
        """Get from cache or compute"""
        
        # Try cache
        value = await self.get(key, skip_l1=skip_l1)
        if value is not None:
            return value
        
        # Compute
        value = await func()
        
        # Cache it
        if value is not None:
            await self.set(key, value, ttl=ttl)
        
        return value
    
    async def get_many(self, keys: List[str]) -> Dict[str, Optional[Any]]:
        """Get multiple values"""
        
        # Try L1 first
        results = {}
        remaining_keys = []
        
        for key in keys:
            value = await self.l1_cache.get(key)
            if value is not None:
                results[key] = value
                self._metrics["l1_hits"] += 1
            else:
                remaining_keys.append(key)
                self._metrics["l1_misses"] += 1
        
        # Try L2 for remaining
        if remaining_keys:
            l2_results = await self.redis_client.get_many(remaining_keys)
            for key, value in l2_results.items():
                if value is not None:
                    results[key] = value
                    self._metrics["l2_hits"] += 1
                    # Populate L1
                    await self.l1_cache.set(key, value)
                else:
                    results[key] = None
                    self._metrics["l2_misses"] += 1
        
        return results
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        l1_metrics = await self.l1_cache.get_metrics()
        
        total_hits = self._metrics["l1_hits"] + self._metrics["l2_hits"]
        total_misses = self._metrics["l1_misses"] + self._metrics["l2_misses"]
        total = total_hits + total_misses
        
        return {
            "l1": l1_metrics,
            "l2": {
                "hits": self._metrics["l2_hits"],
                "misses": self._metrics["l2_misses"],
            },
            "overall": {
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate": total_hits / total if total > 0 else 0,
            },
        }


class CacheManager:
    """
    Advanced Cache Manager with strategies and metrics
    """
    
    def __init__(
        self,
        redis_client: Optional[FastRedisClient] = None,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: int = 3600,
        max_size: int = 1000,
    ):
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.cache = MultiLevelCache(
            redis_client=redis_client,
            l1_max_size=max_size,
            l1_ttl=default_ttl // 12,  # 5 minutes
            l2_ttl=default_ttl,
        )
        
        self._key_prefix = "cdss:cache:"
        self._namespace = "default"
        
        logger.info(f"🗄️ CacheManager initialized: strategy={strategy.value}")
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key with namespace"""
        return f"{self._key_prefix}{self._namespace}:{key}"
    
    def _generate_hash_key(self, *args, **kwargs) -> str:
        """Generate hash key from arguments"""
        data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._get_cache_key(key)
        return await self.cache.get(cache_key)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ):
        """Set value in cache"""
        cache_key = self._get_cache_key(key)
        await self.cache.set(cache_key, value, ttl=ttl or self.default_ttl)
    
    async def delete(self, key: str):
        """Delete from cache"""
        cache_key = self._get_cache_key(key)
        await self.cache.delete(cache_key)
    
    async def delete_pattern(self, pattern: str):
        """Delete keys matching pattern"""
        # This would require Redis SCAN, implement if needed
        pass
    
    async def clear_namespace(self, namespace: Optional[str] = None):
        """Clear entire namespace"""
        if namespace:
            old_namespace = self._namespace
            self._namespace = namespace
        # In production, use Redis SCAN to delete keys
        # For now, just clear L1
        await self.cache.clear()
    
    async def get_or_compute(
        self,
        key: str,
        compute_func: Callable,
        ttl: Optional[int] = None,
        force_refresh: bool = False,
    ) -> Any:
        """Get from cache or compute with lock"""
        
        if not force_refresh:
            cached = await self.get(key)
            if cached is not None:
                return cached
        
        # Compute with distributed lock
        lock_key = f"lock:{key}"
        async with self.cache.redis_client.lock(lock_key, timeout=10):
            # Double-check after acquiring lock
            if not force_refresh:
                cached = await self.get(key)
                if cached is not None:
                    return cached
            
            # Compute value
            value = await compute_func()
            
            # Cache it
            if value is not None:
                await self.set(key, value, ttl=ttl)
            
            return value
    
    def set_namespace(self, namespace: str):
        """Set cache namespace"""
        self._namespace = namespace
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        return await self.cache.get_metrics()