# src/security/rate_limiter.py
"""
Distributed Rate Limiting with Token Bucket Algorithm
"""

import time
import asyncio
import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
from functools import wraps

from src.logger import get_logger
from src.caching.redis_client import get_fast_redis_client

logger = get_logger(__name__)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    
    def __init__(self, limit: int, period: str, retry_after: int):
        self.limit = limit
        self.period = period
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded: {limit} per {period}, retry after {retry_after}s")


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    limit: int = 100
    period: str = "minute"  # second, minute, hour, day
    burst_multiplier: float = 1.5
    key_prefix: str = "rate_limit"
    enabled: bool = True
    
    def get_period_seconds(self) -> int:
        """Get period in seconds"""
        periods = {
            "second": 1,
            "minute": 60,
            "hour": 3600,
            "day": 86400,
        }
        return periods.get(self.period, 60)
    
    def get_burst_capacity(self) -> int:
        """Get burst capacity"""
        return int(self.limit * self.burst_multiplier)


class TokenBucket:
    """
    Token Bucket algorithm implementation
    
    A token bucket has:
    - Capacity: Maximum tokens (burst capacity)
    - Rate: Tokens added per second (refill rate)
    - Tokens: Current token count
    """
    
    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        initial_tokens: Optional[int] = None,
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from the bucket"""
        
        async with self._lock:
            # Refill tokens
            now = time.monotonic()
            elapsed = now - self.last_refill
            new_tokens = elapsed * self.refill_rate
            
            if new_tokens > 0:
                self.tokens = min(self.capacity, self.tokens + new_tokens)
                self.last_refill = now
            
            # Check if enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def get_remaining_tokens(self) -> int:
        """Get remaining tokens"""
        await self._refill()
        return math.floor(self.tokens)
    
    async def get_reset_time(self) -> float:
        """Get time until bucket refills"""
        await self._refill()
        if self.tokens >= self.capacity:
            return 0
        
        needed = self.capacity - self.tokens
        return needed / self.refill_rate
    
    async def _refill(self):
        """Refill tokens"""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            new_tokens = elapsed * self.refill_rate
            
            if new_tokens > 0:
                self.tokens = min(self.capacity, self.tokens + new_tokens)
                self.last_refill = now


class DistributedRateLimiter:
    """
    Distributed Rate Limiter using Redis
    
    Features:
    - Token bucket algorithm
    - Redis-based storage for distributed systems
    - Automatic cleanup
    - Burst handling
    - Per-key rate limiting (IP, user, endpoint)
    """
    
    def __init__(
        self,
        redis_client = None,
        default_config: Optional[RateLimitConfig] = None,
    ):
        self.redis_client = redis_client or get_fast_redis_client()
        self.default_config = default_config or RateLimitConfig()
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
        
        # Cleanup settings
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.monotonic()
        
        logger.info("⏱️ DistributedRateLimiter initialized")
        logger.info(f"   Default: {self.default_config.limit} per {self.default_config.period}")
    
    def _get_key(self, identifier: str, config: RateLimitConfig) -> str:
        """Generate Redis key for a bucket"""
        return f"{config.key_prefix}:{identifier}"
    
    async def is_allowed(
        self,
        identifier: str,
        config: Optional[RateLimitConfig] = None,
        tokens: int = 1,
    ) -> bool:
        """
        Check if a request is allowed
        
        Args:
            identifier: Unique identifier (IP, user ID, etc.)
            config: Rate limit configuration (uses default if None)
            tokens: Tokens to consume
            
        Returns:
            True if allowed, False otherwise
        """
        
        config = config or self.default_config
        
        if not config.enabled:
            return True
        
        # Periodic cleanup
        now = time.monotonic()
        if now - self._last_cleanup > self._cleanup_interval:
            asyncio.create_task(self._cleanup())
            self._last_cleanup = now
        
        key = self._get_key(identifier, config)
        
        # Get bucket from Redis
        bucket = await self._get_bucket(key, config)
        
        # Consume tokens
        allowed = await bucket.consume(tokens)
        
        return allowed
    
    async def _get_bucket(self, key: str, config: RateLimitConfig) -> TokenBucket:
        """Get or create a token bucket"""
        
        # Check memory cache first
        if key in self._buckets:
            return self._buckets[key]
        
        # Try to get from Redis
        async with self._lock:
            if key in self._buckets:
                return self._buckets[key]
            
            # Create new bucket
            capacity = config.get_burst_capacity()
            refill_rate = config.limit / config.get_period_seconds()
            
            # Try to get from Redis
            redis_key = f"bucket:{key}"
            data = await self.redis_client.get(redis_key)
            
            if data:
                try:
                    tokens = data.get("tokens", capacity)
                    last_refill = data.get("last_refill", time.monotonic())
                    bucket = TokenBucket(capacity, refill_rate, tokens)
                    bucket.last_refill = last_refill
                    self._buckets[key] = bucket
                    return bucket
                except:
                    pass
            
            # Create new bucket
            bucket = TokenBucket(capacity, refill_rate)
            self._buckets[key] = bucket
            
            # Save to Redis
            await self._save_bucket(key, bucket)
            
            return bucket
    
    async def _save_bucket(self, key: str, bucket: TokenBucket):
        """Save bucket state to Redis"""
        
        redis_key = f"bucket:{key}"
        data = {
            "tokens": bucket.tokens,
            "last_refill": bucket.last_refill,
        }
        await self.redis_client.set(redis_key, data, ttl=3600)
    
    async def _cleanup(self):
        """Clean up inactive buckets"""
        
        async with self._lock:
            now = time.monotonic()
            keys_to_remove = []
            
            for key, bucket in self._buckets.items():
                if now - bucket.last_refill > 3600:  # 1 hour inactivity
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._buckets[key]
        
        if keys_to_remove:
            logger.debug(f"🧹 Cleaned up {len(keys_to_remove)} inactive buckets")
    
    async def get_remaining(self, identifier: str, config: Optional[RateLimitConfig] = None) -> int:
        """Get remaining tokens for an identifier"""
        
        config = config or self.default_config
        key = self._get_key(identifier, config)
        
        bucket = await self._get_bucket(key, config)
        return await bucket.get_remaining_tokens()
    
    async def get_reset_time(self, identifier: str, config: Optional[RateLimitConfig] = None) -> float:
        """Get time until bucket resets"""
        
        config = config or self.default_config
        key = self._get_key(identifier, config)
        
        bucket = await self._get_bucket(key, config)
        return await bucket.get_reset_time()
    
    async def reset(self, identifier: str, config: Optional[RateLimitConfig] = None):
        """Reset rate limit for an identifier"""
        
        config = config or self.default_config
        key = self._get_key(identifier, config)
        
        async with self._lock:
            if key in self._buckets:
                del self._buckets[key]
            
            # Delete from Redis
            redis_key = f"bucket:{key}"
            await self.redis_client.delete(redis_key)
        
        logger.debug(f"🔄 Rate limit reset for {identifier}")


class RateLimiter:
    """
    Convenience wrapper for rate limiting with decorators
    """
    
    def __init__(self, distributed_limiter: Optional[DistributedRateLimiter] = None):
        self.limiter = distributed_limiter or DistributedRateLimiter()
        
    def limit(
        self,
        identifier: Optional[str] = None,
        limit: int = 100,
        period: str = "minute",
    ):
        """
        Decorator for rate limiting a function
        
        Args:
            identifier: Function identifier (defaults to function name)
            limit: Rate limit
            period: Period (second, minute, hour, day)
        """
        
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                config = RateLimitConfig(
                    limit=limit,
                    period=period,
                )
                
                func_id = identifier or func.__name__
                
                allowed = await self.limiter.is_allowed(func_id, config)
                
                if not allowed:
                    retry_after = await self.limiter.get_reset_time(func_id, config)
                    raise RateLimitExceeded(limit, period, int(retry_after) + 1)
                
                return await func(*args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Run async wrapper in event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return loop.create_task(async_wrapper(*args, **kwargs))
                else:
                    return loop.run_until_complete(async_wrapper(*args, **kwargs))
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator


# ============================================================================
# 🔧 FastAPI Middleware Integration
# ============================================================================

class RateLimitMiddleware:
    """
    FastAPI middleware for rate limiting
    """
    
    def __init__(
        self,
        app,
        limit: int = 100,
        period: str = "minute",
        identifier_extractor=None,
    ):
        self.app = app
        self.limiter = DistributedRateLimiter()
        self.default_config = RateLimitConfig(limit=limit, period=period)
        self.identifier_extractor = identifier_extractor or self._get_client_ip
        
        logger.info("⏱️ RateLimitMiddleware initialized")
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Extract identifier
        request = None
        # In FastAPI, we get request from scope
        # This is a simplified version
        
        identifier = await self.identifier_extractor(scope)
        
        # Check rate limit
        allowed = await self.limiter.is_allowed(identifier, self.default_config)
        
        if not allowed:
            # Rate limit exceeded
            retry_after = await self.limiter.get_reset_time(identifier, self.default_config)
            response = {
                "error": "Rate limit exceeded",
                "retry_after": int(retry_after) + 1,
                "limit": self.default_config.limit,
                "period": self.default_config.period,
            }
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"retry-after", str(int(retry_after) + 1).encode()],
                ],
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps(response).encode(),
            })
            return
        
        # Proceed to application
        await self.app(scope, receive, send)
    
    async def _get_client_ip(self, scope) -> str:
        """Extract client IP from request scope"""
        # In production, use forwarded headers
        return scope.get("client", ("unknown", 0))[0]


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_rate_limiter: Optional[DistributedRateLimiter] = None


def get_rate_limiter() -> DistributedRateLimiter:
    """Get rate limiter singleton"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = DistributedRateLimiter()
    return _rate_limiter