"""Retry mechanisms and error recovery utilities."""

import asyncio
import functools
import random
import time
from typing import Any, Callable, Optional, Type, Union

from ..exceptions import DataQAError, ErrorRecovery, RetryableError
from ..logging_config import get_logger


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[tuple[Type[Exception], ...]] = None
    ):
        """Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
            retryable_exceptions: Tuple of exception types that should trigger retries
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (
            RetryableError,
            ConnectionError,
            TimeoutError,
        )
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = min(
            self.base_delay * (self.exponential_base ** (attempt - 1)),
            self.max_delay
        )
        
        if self.jitter:
            # Add up to 25% jitter
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount
        
        return delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry."""
        if attempt >= self.max_attempts:
            return False
        
        # Check if it's a retryable exception type
        if isinstance(exception, self.retryable_exceptions):
            return True
        
        # Use ErrorRecovery utility for additional logic
        return ErrorRecovery.should_retry(exception, attempt)


def retry_sync(
    config: Optional[RetryConfig] = None,
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    logger_name: Optional[str] = None
) -> Callable:
    """Decorator for synchronous function retry logic.
    
    Args:
        config: Retry configuration object
        max_attempts: Maximum retry attempts (used if config not provided)
        base_delay: Base delay between retries (used if config not provided)
        logger_name: Logger name for retry logging
    
    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig(max_attempts=max_attempts, base_delay=base_delay)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    if attempt > 1:
                        logger.info(f"Retry attempt {attempt}/{config.max_attempts} for {func.__name__}")
                    
                    result = func(*args, **kwargs)
                    
                    if attempt > 1:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt}")
                    
                    return result
                
                except Exception as e:
                    last_exception = e
                    
                    if not config.should_retry(e, attempt):
                        logger.error(f"Function {func.__name__} failed permanently: {e}")
                        raise
                    
                    if attempt < config.max_attempts:
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            f"Function {func.__name__} failed on attempt {attempt}, "
                            f"retrying in {delay:.2f}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"Function {func.__name__} failed after {attempt} attempts: {e}")
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    return decorator


def retry_async(
    config: Optional[RetryConfig] = None,
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    logger_name: Optional[str] = None
) -> Callable:
    """Decorator for asynchronous function retry logic.
    
    Args:
        config: Retry configuration object
        max_attempts: Maximum retry attempts (used if config not provided)
        base_delay: Base delay between retries (used if config not provided)
        logger_name: Logger name for retry logging
    
    Returns:
        Decorated async function with retry logic
    """
    if config is None:
        config = RetryConfig(max_attempts=max_attempts, base_delay=base_delay)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    if attempt > 1:
                        logger.info(f"Retry attempt {attempt}/{config.max_attempts} for {func.__name__}")
                    
                    result = await func(*args, **kwargs)
                    
                    if attempt > 1:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt}")
                    
                    return result
                
                except Exception as e:
                    last_exception = e
                    
                    if not config.should_retry(e, attempt):
                        logger.error(f"Function {func.__name__} failed permanently: {e}")
                        raise
                    
                    if attempt < config.max_attempts:
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            f"Function {func.__name__} failed on attempt {attempt}, "
                            f"retrying in {delay:.2f}s: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Function {func.__name__} failed after {attempt} attempts: {e}")
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
        self.logger = get_logger(__name__, component="circuit_breaker")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
    
    def _call_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker logic."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
                self.logger.info("Circuit breaker moving to half-open state")
            else:
                raise DataQAError(
                    "Circuit breaker is open",
                    user_message="Service is temporarily unavailable. Please try again later.",
                    error_code="CIRCUIT_BREAKER_OPEN"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful function execution."""
        if self.state == "half-open":
            self.state = "closed"
            self.failure_count = 0
            self.logger.info("Circuit breaker reset to closed state")
    
    def _on_failure(self):
        """Handle failed function execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


class GracefulDegradation:
    """Utilities for graceful degradation when services fail."""
    
    @staticmethod
    def with_fallback(
        primary_func: Callable,
        fallback_func: Callable,
        fallback_exceptions: tuple[Type[Exception], ...] = (Exception,),
        logger_name: Optional[str] = None
    ) -> Callable:
        """Execute primary function with fallback on failure.
        
        Args:
            primary_func: Primary function to execute
            fallback_func: Fallback function to execute on failure
            fallback_exceptions: Exception types that trigger fallback
            logger_name: Logger name for fallback logging
        
        Returns:
            Function that executes primary with fallback
        """
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or primary_func.__module__)
            
            try:
                return primary_func(*args, **kwargs)
            except fallback_exceptions as e:
                logger.warning(
                    f"Primary function {primary_func.__name__} failed, using fallback: {e}"
                )
                try:
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback function {fallback_func.__name__} also failed: {fallback_error}"
                    )
                    # Re-raise original error
                    raise e
        
        return wrapper
    
    @staticmethod
    async def with_fallback_async(
        primary_func: Callable,
        fallback_func: Callable,
        fallback_exceptions: tuple[Type[Exception], ...] = (Exception,),
        logger_name: Optional[str] = None
    ) -> Callable:
        """Async version of with_fallback."""
        async def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or primary_func.__module__)
            
            try:
                if asyncio.iscoroutinefunction(primary_func):
                    return await primary_func(*args, **kwargs)
                else:
                    return primary_func(*args, **kwargs)
            except fallback_exceptions as e:
                logger.warning(
                    f"Primary function {primary_func.__name__} failed, using fallback: {e}"
                )
                try:
                    if asyncio.iscoroutinefunction(fallback_func):
                        return await fallback_func(*args, **kwargs)
                    else:
                        return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback function {fallback_func.__name__} also failed: {fallback_error}"
                    )
                    # Re-raise original error
                    raise e
        
        return wrapper


# Convenience functions for common retry patterns
def retry_on_connection_error(max_attempts: int = 3, base_delay: float = 1.0):
    """Retry decorator specifically for connection errors."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        retryable_exceptions=(ConnectionError, TimeoutError, OSError)
    )
    return retry_sync(config)


def retry_on_rate_limit(max_attempts: int = 5, base_delay: float = 2.0):
    """Retry decorator specifically for rate limit errors."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=120.0,  # Longer max delay for rate limits
        retryable_exceptions=(RetryableError,)
    )
    return retry_async(config)