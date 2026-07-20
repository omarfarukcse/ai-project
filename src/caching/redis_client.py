# src/caching/redis_client.py
"""
Ultra-Fast Redis Client with Connection Pooling and Auto-Reconnection
"""

import asyncio
import orjson
from typing import Optional, Any, Dict, List, Union, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
import time
from prometheus_client import Counter, Histogram

try:
    import redis.asyncio as redis
    from redis.asyncio import ConnectionPool
    from redis.exceptions import ConnectionError, TimeoutError, RedisError
except ImportError:
    raise ImportError("redis-py is required for caching. Install with: pip install redis")

from src.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# 📊 Metrics
# ============================================================================

CACHE_METRICS = {
    "cache_hits": Counter(
        "cdss_cache_hits_total",
        "Total cache hits",
        ["cache_level"]
    ),
    "cache_misses": Counter(
        "cdss_cache_misses_total",
        "Total cache misses",
        ["cache_level"]
    ),
    "cache_latency": Histogram(
        "cdss_cache_latency_seconds",
        "Cache operation latency",
        ["operation", "cache_level"],
        buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
    ),
    "cache_errors": Counter(
        "cdss_cache_errors_total",
        "Total cache errors",
        ["operation", "error_type"]
    ),
}


@dataclass
class CacheConfig:
    """Redis cache configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 20
    connection_timeout: int = 1
    socket_timeout: int = 1
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    default_ttl: int = 3600  # 1 hour


class FastRedisClient:
    """
    Ultra-fast Redis client with:
    - Connection pooling
    - Pipeline support for batch operations
    - Zero-copy serialization with orjson
    - Automatic reconnection
    - Health checks
    - Circuit breaker integration
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 20,
        connection_timeout: int = 1,
        socket_timeout: int = 1,
        default_ttl: int = 3600,
    ):
        self.config = CacheConfig(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            connection_timeout=connection_timeout,
            socket_timeout=socket_timeout,
            default_ttl=default_ttl,
        )
        
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._connected: bool = False
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_latency_ms": 0,
        }
        
        logger.info(f"⚡ FastRedisClient initialized: {host}:{port}")
    
    async def connect(self) -> bool:
        """Connect to Redis with retry logic"""
        if self._connected:
            return True
        
        async with self._lock:
            if self._connected:
                return True
            
            try:
                self._pool = ConnectionPool(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password,
                    max_connections=self.config.max_connections,
                    connection_timeout=self.config.connection_timeout,
                    socket_timeout=self.config.socket_timeout,
                    retry_on_timeout=self.config.retry_on_timeout,
                    decode_responses=False,  # Keep as bytes for speed
                )
                
                self._client = redis.Redis(connection_pool=self._pool)
                
                # Test connection
                await self._client.ping()
                self._connected = True
                logger.info(f"✅ Redis connected: {self.config.host}:{self.config.port}")
                
                # Start health check
                self._start_health_check()
                
                return True
                
            except (ConnectionError, TimeoutError) as e:
                logger.error(f"❌ Redis connection failed: {str(e)}")
                self._connected = False
                return False
    
    async def close(self):
        """Close Redis connection"""
        if self._health_check_task:
            self._health_check_task.cancel()
        
        if self._client:
            await self._client.close()
            self._connected = False
            logger.info("Redis connection closed")
    
    def _start_health_check(self):
        """Start periodic health check"""
        async def health_check_loop():
            while True:
                await asyncio.sleep(self.config.health_check_interval)
                if self._connected:
                    try:
                        await self._client.ping()
                    except Exception:
                        self._connected = False
                        logger.warning("Redis health check failed, attempting reconnect...")
                        await self.connect()
        
        self._health_check_task = asyncio.create_task(health_check_loop())
    
    async def _ensure_connected(self) -> bool:
        """Ensure Redis connection is established"""
        if not self._connected:
            return await self.connect()
        return True
    
    async def _execute_operation(
        self,
        operation: Callable,
        operation_name: str,
        cache_level: str = "redis",
        *args,
        **kwargs,
    ) -> Any:
        """Execute Redis operation with metrics"""
        
        if not await self._ensure_connected():
            CACHE_METRICS["cache_errors"].labels(
                operation=operation_name,
                error_type="connection"
            ).inc()
            return None
        
        start_time = time.perf_counter()
        self._metrics["total_operations"] += 1
        
        try:
            result = await operation(*args, **kwargs)
            
            # Update metrics
            latency = (time.perf_counter() - start_time) * 1000
            self._metrics["successful_operations"] += 1
            self._metrics["avg_latency_ms"] = (
                self._metrics["avg_latency_ms"] * 0.9 + latency * 0.1
            )
            
            CACHE_METRICS["cache_latency"].labels(
                operation=operation_name,
                cache_level=cache_level
            ).observe(latency / 1000)
            
            return result
            
        except Exception as e:
            self._metrics["failed_operations"] += 1
            CACHE_METRICS["cache_errors"].labels(
                operation=operation_name,
                error_type=type(e).__name__
            ).inc()
            
            logger.error(f"Redis operation failed: {operation_name} - {str(e)}")
            
            # Attempt to reconnect on connection errors
            if isinstance(e, (ConnectionError, TimeoutError)):
                self._connected = False
                await self.connect()
            
            return None
    
    # ============================================================================
    # 🚀 Core Redis Operations
    # ============================================================================
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis (orjson deserialization)"""
        
        async def _get():
            value = await self._client.get(key.encode())
            if value:
                return orjson.loads(value)
            return None
        
        return await self._execute_operation(
            _get,
            "get",
            args=[key],
        )
    
    async def get_binary(self, key: str) -> Optional[bytes]:
        """Get raw bytes from Redis"""
        
        async def _get_binary():
            return await self._client.get(key.encode())
        
        return await self._execute_operation(
            _get_binary,
            "get_binary",
            args=[key],
        )
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set value in Redis (orjson serialization)"""
        
        async def _set():
            serialized = orjson.dumps(value)
            ttl_value = ttl or self.config.default_ttl
            
            if ttl_value:
                await self._client.setex(
                    key.encode(),
                    ttl_value,
                    serialized,
                    nx=nx,
                    xx=xx,
                )
            else:
                await self._client.set(
                    key.encode(),
                    serialized,
                    nx=nx,
                    xx=xx,
                )
            return True
        
        return await self._execute_operation(
            _set,
            "set",
            args=[key],
        )
    
    async def setex(self, key: str, ttl: int, value: Any) -> bool:
        """Set value with TTL"""
        return await self.set(key, value, ttl=ttl)
    
    async def delete(self, *keys: str) -> int:
        """Delete one or more keys"""
        
        async def _delete():
            if keys:
                return await self._client.delete(*[k.encode() for k in keys])
            return 0
        
        return await self._execute_operation(
            _delete,
            "delete",
            args=list(keys),
        )
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        
        async def _exists():
            return await self._client.exists(key.encode()) > 0
        
        return await self._execute_operation(
            _exists,
            "exists",
            args=[key],
        )
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set key TTL"""
        
        async def _expire():
            return await self._client.expire(key.encode(), ttl)
        
        return await self._execute_operation(
            _expire,
            "expire",
            args=[key],
        )
    
    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a counter"""
        
        async def _incr():
            return await self._client.incrby(key.encode(), amount)
        
        return await self._execute_operation(
            _incr,
            "incr",
            args=[key],
        )
    
    async def ttl(self, key: str) -> Optional[int]:
        """Get key TTL in seconds"""
        
        async def _ttl():
            return await self._client.ttl(key.encode())
        
        return await self._execute_operation(
            _ttl,
            "ttl",
            args=[key],
        )
    
    # ============================================================================
    # 🚀 Batch Operations
    # ============================================================================
    
    async def pipeline(self):
        """Create a pipeline for batch operations"""
        if not await self._ensure_connected():
            return None
        return self._client.pipeline()
    
    async def execute_pipeline(self, pipeline):
        """Execute pipeline"""
        try:
            return await pipeline.execute()
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            return []
    
    async def get_many(self, keys: List[str]) -> Dict[str, Optional[Any]]:
        """Get multiple values from Redis"""
        
        async def _get_many():
            if not keys:
                return {}
            
            # Use pipeline for batch get
            pipeline = self._client.pipeline()
            for key in keys:
                pipeline.get(key.encode())
            results = await pipeline.execute()
            
            return {
                key: orjson.loads(value) if value else None
                for key, value in zip(keys, results)
            }
        
        return await self._execute_operation(
            _get_many,
            "get_many",
            args=[keys],
        )
    
    async def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """Set multiple values with optional TTL"""
        
        async def _set_many():
            ttl_value = ttl or self.config.default_ttl
            
            pipeline = self._client.pipeline()
            for key, value in items.items():
                serialized = orjson.dumps(value)
                if ttl_value:
                    pipeline.setex(key.encode(), ttl_value, serialized)
                else:
                    pipeline.set(key.encode(), serialized)
            await pipeline.execute()
            return True
        
        return await self._execute_operation(
            _set_many,
            "set_many",
            args=[items],
        )
    
    # ============================================================================
    # 🔧 Advanced Operations
    # ============================================================================
    
    @asynccontextmanager
    async def lock(self, key: str, timeout: int = 10):
        """Distributed lock context manager"""
        from contextlib import asynccontextmanager
        
        lock_key = f"lock:{key}"
        
        async def _acquire():
            return await self._client.set(
                lock_key.encode(),
                "locked",
                nx=True,
                ex=timeout,
            )
        
        async def _release():
            await self._client.delete(lock_key.encode())
        
        try:
            acquired = await _acquire()
            if not acquired:
                raise RuntimeError(f"Could not acquire lock for key: {key}")
            yield
        finally:
            await _release()
    
    async def set_if_not_exists(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value only if key doesn't exist"""
        return await self.set(key, value, ttl=ttl, nx=True)
    
    async def get_or_set(
        self,
        key: str,
        func: Callable,
        ttl: Optional[int] = None,
    ) -> Optional[Any]:
        """Get value from cache or compute and cache"""
        
        # Try to get from cache
        cached = await self.get(key)
        if cached is not None:
            CACHE_METRICS["cache_hits"].labels(cache_level="redis").inc()
            return cached
        
        CACHE_METRICS["cache_misses"].labels(cache_level="redis").inc()
        
        # Compute value
        value = await func()
        
        # Cache it
        if value is not None:
            await self.set(key, value, ttl=ttl)
        
        return value
    
    # ============================================================================
    # 🔧 Utility Methods
    # ============================================================================
    
    async def ping(self) -> bool:
        """Ping Redis server"""
        try:
            if await self._ensure_connected():
                return await self._client.ping()
            return False
        except Exception:
            return False
    
    async def flush_db(self) -> bool:
        """Flush current database (use with caution)"""
        async def _flush():
            await self._client.flushdb()
            return True
        
        return await self._execute_operation(
            _flush,
            "flushdb",
        )
    
    async def info(self) -> Dict[str, Any]:
        """Get Redis info"""
        async def _info():
            return await self._client.info()
        
        return await self._execute_operation(
            _info,
            "info",
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics"""
        return {
            **self._metrics,
            "connected": self._connected,
            "pool_size": self._pool.size if self._pool else 0,
            "pool_available": self._pool.available_count if self._pool else 0,
        }
    
    def is_connected(self) -> bool:
        """Check if connected to Redis"""
        return self._connected


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_redis_client: Optional[FastRedisClient] = None


def get_fast_redis_client() -> FastRedisClient:
    """Get Redis client instance (singleton)"""
    global _redis_client
    if _redis_client is None:
        from src.config_manager import config_manager
        config = config_manager.get_redis_config()
        _redis_client = FastRedisClient(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            max_connections=config.max_connections,
            default_ttl=config.cache_ttl,
        )
    return _redis_client


# ============================================================================
# 🔧 Redis Connection Pool (Low-level)
# ============================================================================

class RedisConnectionPool:
    """
    Low-level Redis connection pool with advanced features
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._pool: Optional[ConnectionPool] = None
        
    async def initialize(self):
        """Initialize connection pool"""
        self._pool = ConnectionPool(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            max_connections=self.config.max_connections,
            connection_timeout=self.config.connection_timeout,
            socket_timeout=self.config.socket_timeout,
            retry_on_timeout=self.config.retry_on_timeout,
        )
        return self
    
    async def get_connection(self):
        """Get a connection from pool"""
        if not self._pool:
            await self.initialize()
        return redis.Redis(connection_pool=self._pool)
    
    async def close(self):
        """Close all connections"""
        if self._pool:
            await self._pool.disconnect()