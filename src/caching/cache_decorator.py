# src/caching/cache_decorator.py
"""
Cache Decorators for Automatic Caching
"""

import asyncio
import functools
import inspect
import time
from typing import Any, Callable, Optional, Union, List, Dict
from functools import wraps

from src.caching.cache_manager import CacheManager, get_fast_redis_client
from src.logger import get_logger

logger = get_logger(__name__)

# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create cache manager"""
    global _cache_manager
    if _cache_manager is None:
        redis_client = get_fast_redis_client()
        _cache_manager = CacheManager(redis_client=redis_client)
    return _cache_manager


def cached(
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
    skip_cache: Optional[Callable] = None,
    namespace: Optional[str] = None,
    cache_none: bool = False,
):
    """
    Decorator for automatic caching of function results
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        skip_cache: Function to determine if cache should be skipped
        namespace: Cache namespace
        cache_none: Whether to cache None results
    
    Example:
        @cached(ttl=300)
        async def predict_risk(patient_id: str):
            return await compute_risk(patient_id)
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Skip caching if requested
            if skip_cache and skip_cache(*args, **kwargs):
                return await func(*args, **kwargs)
            
            # Generate cache key
            cache_manager = get_cache_manager()
            
            if namespace:
                cache_manager.set_namespace(namespace)
            
            # Build key
            key_parts = [key_prefix or func.__name__]
            
            # Add function arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for name, value in bound_args.arguments.items():
                if name != 'self':
                    key_parts.append(f"{name}:{str(value)}")
            
            cache_key = ":".join(key_parts)
            
            # Try cache
            cached_value = await cache_manager.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_value
            
            # Compute
            result = await func(*args, **kwargs)
            
            # Cache result
            if result is not None or cache_none:
                await cache_manager.set(cache_key, result, ttl=ttl)
                logger.debug(f"Cached result: {cache_key}")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, run in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context
                return loop.create_task(async_wrapper(*args, **kwargs))
            else:
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        # Use async wrapper if function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def cache_invalidate(pattern: Optional[str] = None, key_prefix: Optional[str] = None):
    """
    Decorator to invalidate cache after function execution
    
    Args:
        pattern: Key pattern to invalidate
        key_prefix: Prefix of keys to invalidate
    
    Example:
        @cache_invalidate(key_prefix="predict")
        async def update_patient(patient_id: str):
            return await save_patient(patient_id)
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            cache_manager = get_cache_manager()
            
            if pattern:
                await cache_manager.delete_pattern(pattern)
            elif key_prefix:
                # Invalidate all keys with this prefix
                # In production, use Redis SCAN
                pass
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.create_task(async_wrapper(*args, **kwargs))
            else:
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def cache_warm(keys: Union[str, List[str]], namespace: Optional[str] = None):
    """
    Decorator to warm cache after function execution
    
    Example:
        @cache_warm(["patient_123", "patient_456"])
        async def get_patients():
            return await fetch_patients()
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            cache_manager = get_cache_manager()
            
            if namespace:
                cache_manager.set_namespace(namespace)
            
            # Cache the result under each key
            if isinstance(keys, list):
                for key in keys:
                    await cache_manager.set(key, result)
            else:
                await cache_manager.set(keys, result)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.create_task(async_wrapper(*args, **kwargs))
            else:
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def cache_metrics(func: Callable) -> Callable:
    """
    Decorator to track cache metrics for a function
    """
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        
        try:
            result = await func(*args, **kwargs)
            success = True
        except Exception as e:
            success = False
            raise
        
        finally:
            duration = (time.perf_counter() - start_time) * 1000
            
            # Record metrics
            from src.caching import CACHE_METRICS
            
            CACHE_METRICS["cache_latency"].labels(
                operation=func.__name__,
                cache_level="compute"
            ).observe(duration / 1000)
            
            if success:
                logger.debug(f"{func.__name__} completed in {duration:.2f}ms")
        
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.create_task(async_wrapper(*args, **kwargs))
        else:
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper