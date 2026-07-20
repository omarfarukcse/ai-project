# src/utils/circuit_breaker.py
"""
Circuit Breaker Pattern for Fault Tolerance and Resilience
"""

import time
import threading
import asyncio
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from collections import deque
from functools import wraps
import logging

from src.logger import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 30
    half_open_max_calls: int = 3
    success_threshold: int = 2
    timeout_seconds: int = 10
    name: str = "default"


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    open_count: int = 0
    half_open_count: int = 0
    last_open_time: Optional[float] = None
    last_close_time: Optional[float] = None


class CircuitBreaker:
    """
    Circuit Breaker Pattern Implementation
    
    States:
    - CLOSED: Normal operation, requests go through
    - OPEN: Failing, requests are rejected
    - HALF_OPEN: Testing recovery, limited requests allowed
    
    Features:
    - Thread-safe
    - Async support
    - Automatic recovery
    - Metrics collection
    - Event callbacks
    """
    
    def __init__(
        self,
        name: str = "default",
        config: Optional[CircuitBreakerConfig] = None,
        on_open: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        on_half_open: Optional[Callable] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig(name=name)
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        self._half_open_calls = 0
        self._half_open_success_count = 0
        self._lock = threading.Lock()
        self._metrics = CircuitBreakerMetrics()
        
        # Event callbacks
        self._on_open = on_open
        self._on_close = on_close
        self._on_half_open = on_half_open
        
        # Async event callbacks
        self._async_on_open: Optional[Callable] = None
        self._async_on_close: Optional[Callable] = None
        self._async_on_half_open: Optional[Callable] = None
        
        # Async lock
        self._async_lock = asyncio.Lock()
        
        logger.info(f"🔌 Circuit Breaker '{name}' initialized")
        logger.info(f"   Failure Threshold: {self.config.failure_threshold}")
        logger.info(f"   Recovery Timeout: {self.config.recovery_timeout}s")
    
    # ============================================================================
    # 🚀 Core Methods
    # ============================================================================
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenException: When circuit is open
            Exception: Original exception from function
        """
        
        with self._lock:
            self._metrics.total_requests += 1
            
            # Check if circuit is open
            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    logger.info(f"🔄 Circuit '{self.name}' transitioning to HALF_OPEN")
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._half_open_success_count = 0
                    self._metrics.half_open_count += 1
                    
                    if self._on_half_open:
                        self._on_half_open()
                else:
                    self._metrics.rejected_requests += 1
                    raise CircuitBreakerOpenException(
                        f"Circuit '{self.name}' is open. Service unavailable."
                    )
            
            # Half-open state
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._metrics.rejected_requests += 1
                    raise CircuitBreakerOpenException(
                        f"Circuit '{self.name}' half-open limit reached"
                    )
                self._half_open_calls += 1
        
        # Execute function with timeout
        result = self._execute_with_timeout(func, *args, **kwargs)
        
        # Update state based on success/failure
        if result is not None:
            self._record_success()
            return result
        
        # If we get here, execution failed
        self._record_failure()
        return None
    
    def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function execution timed out ({self.config.timeout_seconds}s)")
        
        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.config.timeout_seconds)
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)
            return result
        except Exception as e:
            signal.alarm(0)
            # Re-raise the exception
            raise
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def _record_success(self):
        """Record a successful request"""
        with self._lock:
            self._metrics.successful_requests += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_success_count += 1
                if self._half_open_success_count >= self.config.success_threshold:
                    logger.info(f"✅ Circuit '{self.name}' recovered - transitioning to CLOSED")
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._metrics.last_close_time = time.time()
                    
                    if self._on_close:
                        self._on_close()
            else:
                self._failure_count = 0  # Reset failure count on success
    
    def _record_failure(self):
        """Record a failed request"""
        with self._lock:
            self._metrics.failed_requests += 1
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open means open again
                logger.warning(f"⚠️ Circuit '{self.name}' failure in half-open - transitioning to OPEN")
                self._state = CircuitState.OPEN
                self._metrics.open_count += 1
                self._metrics.last_open_time = time.time()
                self._failure_count = 0  # Reset for next recovery attempt
                
                if self._on_open:
                    self._on_open()
                
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning(
                        f"⚠️ Circuit '{self.name}' opened due to {self._failure_count} failures"
                    )
                    self._state = CircuitState.OPEN
                    self._metrics.open_count += 1
                    self._metrics.last_open_time = time.time()
                    
                    if self._on_open:
                        self._on_open()
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed for recovery attempt"""
        if self._state != CircuitState.OPEN:
            return False
        
        return time.time() - self._last_failure_time >= self.config.recovery_timeout
    
    # ============================================================================
    # 🔧 Async Methods
    # ============================================================================
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Async version of call"""
        
        async with self._async_lock:
            self._metrics.total_requests += 1
            
            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    logger.info(f"🔄 Circuit '{self.name}' transitioning to HALF_OPEN")
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._half_open_success_count = 0
                    self._metrics.half_open_count += 1
                    
                    if self._async_on_half_open:
                        await self._async_on_half_open()
                else:
                    self._metrics.rejected_requests += 1
                    raise CircuitBreakerOpenException(
                        f"Circuit '{self.name}' is open. Service unavailable."
                    )
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._metrics.rejected_requests += 1
                    raise CircuitBreakerOpenException(
                        f"Circuit '{self.name}' half-open limit reached"
                    )
                self._half_open_calls += 1
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout_seconds
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, func, *args, **kwargs),
                    timeout=self.config.timeout_seconds
                )
            
            self._record_success()
            return result
            
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
            self._record_failure()
            raise
    
    # ============================================================================
    # 🔧 Reset and Management
    # ============================================================================
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        with self._lock:
            logger.info(f"🔄 Manual reset of circuit '{self.name}'")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = 0
            self._half_open_calls = 0
            self._half_open_success_count = 0
    
    def get_state(self) -> str:
        """Get current circuit state"""
        return self._state.value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'total_requests': self._metrics.total_requests,
                'successful_requests': self._metrics.successful_requests,
                'failed_requests': self._metrics.failed_requests,
                'rejected_requests': self._metrics.rejected_requests,
                'open_count': self._metrics.open_count,
                'half_open_count': self._metrics.half_open_count,
                'last_open_time': self._metrics.last_open_time,
                'last_close_time': self._metrics.last_close_time,
                'success_rate': (
                    self._metrics.successful_requests / max(1, self._metrics.total_requests)
                ),
            }
    
    def set_callbacks(
        self,
        on_open: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        on_half_open: Optional[Callable] = None,
    ):
        """Set event callbacks"""
        self._on_open = on_open
        self._on_close = on_close
        self._on_half_open = on_half_open
    
    async def set_async_callbacks(
        self,
        on_open: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        on_half_open: Optional[Callable] = None,
    ):
        """Set async event callbacks"""
        self._async_on_open = on_open
        self._async_on_close = on_close
        self._async_on_half_open = on_half_open


# ============================================================================
# 🔧 Circuit Breaker Registry
# ============================================================================

class CircuitBreakerRegistry:
    """Registry for managing circuit breakers"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        
        logger.info("🔌 CircuitBreakerRegistry initialized")
    
    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name"""
        return self._breakers.get(name)
    
    def reset_all(self):
        """Reset all circuit breakers"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("🔄 All circuit breakers reset")
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all circuit breakers"""
        return {
            name: breaker.get_metrics()
            for name, breaker in self._breakers.items()
        }


# ============================================================================
# 🔧 Decorators
# ============================================================================

def circuit_breaker(
    name: str = "default",
    config: Optional[CircuitBreakerConfig] = None,
):
    """
    Decorator for applying circuit breaker to functions
    
    Example:
        @circuit_breaker(name="ml_model")
        def predict(data):
            return model.predict(data)
    """
    
    registry = CircuitBreakerRegistry()
    breaker = registry.get_or_create(name, config)
    
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call_async(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# ============================================================================
# 🔧 Singleton Instances
# ============================================================================

_breaker_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get circuit breaker registry singleton"""
    global _breaker_registry
    if _breaker_registry is None:
        _breaker_registry = CircuitBreakerRegistry()
    return _breaker_registry


def get_circuit_breaker(
    name: str = "default",
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker"""
    registry = get_circuit_breaker_registry()
    return registry.get_or_create(name, config)