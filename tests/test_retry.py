"""Tests for retry mechanisms and error recovery utilities."""

import asyncio
import time
from unittest.mock import patch, MagicMock

import pytest

from src.dataqa.exceptions import RetryableError, LLMError
from src.dataqa.utils.retry import (
    RetryConfig,
    retry_sync,
    retry_async,
    CircuitBreaker,
    GracefulDegradation,
    retry_on_connection_error,
    retry_on_rate_limit
)


class TestRetryConfig:
    """Test retry configuration."""
    
    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert RetryableError in config.retryable_exceptions
    
    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=False,
            retryable_exceptions=(ValueError,)
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0
        assert config.jitter is False
        assert config.retryable_exceptions == (ValueError,)
    
    def test_calculate_delay(self):
        """Test delay calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        
        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 2.0
        assert config.calculate_delay(3) == 4.0
        assert config.calculate_delay(4) == 8.0
    
    def test_calculate_delay_with_max(self):
        """Test delay calculation with maximum."""
        config = RetryConfig(base_delay=10.0, max_delay=15.0, jitter=False)
        
        assert config.calculate_delay(1) == 10.0
        assert config.calculate_delay(2) == 15.0  # Capped at max_delay
        assert config.calculate_delay(3) == 15.0  # Still capped
    
    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(base_delay=1.0, jitter=True)
        
        delay1 = config.calculate_delay(1)
        delay2 = config.calculate_delay(1)
        
        # With jitter, delays should be different
        assert delay1 != delay2
        assert delay1 >= 1.0  # Should be at least base delay
        assert delay2 >= 1.0
    
    def test_should_retry(self):
        """Test retry decision logic."""
        config = RetryConfig(max_attempts=3, retryable_exceptions=(ValueError,))
        
        # Should retry retryable exceptions within max attempts
        assert config.should_retry(ValueError("test"), 1) is True
        assert config.should_retry(ValueError("test"), 2) is True
        assert config.should_retry(ValueError("test"), 3) is False  # At max
        
        # Should not retry non-retryable exceptions
        assert config.should_retry(TypeError("test"), 1) is False


class TestRetrySyncDecorator:
    """Test synchronous retry decorator."""
    
    def test_successful_function(self):
        """Test retry decorator with successful function."""
        call_count = 0
        
        @retry_sync(max_attempts=3)
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = test_function()
        
        assert result == "success"
        assert call_count == 1
    
    def test_function_succeeds_on_retry(self):
        """Test function that succeeds on second attempt."""
        call_count = 0
        
        @retry_sync(max_attempts=3, base_delay=0.01)  # Fast retry for testing
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RetryableError("Temporary failure")
            return "success"
        
        result = test_function()
        
        assert result == "success"
        assert call_count == 2
    
    def test_function_fails_permanently(self):
        """Test function that fails all retry attempts."""
        call_count = 0
        
        @retry_sync(max_attempts=3, base_delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise RetryableError("Permanent failure")
        
        with pytest.raises(RetryableError):
            test_function()
        
        assert call_count == 3
    
    def test_non_retryable_exception(self):
        """Test that non-retryable exceptions are not retried."""
        call_count = 0
        
        @retry_sync(max_attempts=3)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")
        
        with pytest.raises(ValueError):
            test_function()
        
        assert call_count == 1  # Should not retry


class TestRetryAsyncDecorator:
    """Test asynchronous retry decorator."""
    
    @pytest.mark.asyncio
    async def test_successful_async_function(self):
        """Test retry decorator with successful async function."""
        call_count = 0
        
        @retry_async(max_attempts=3)
        async def test_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await test_function()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_async_function_succeeds_on_retry(self):
        """Test async function that succeeds on second attempt."""
        call_count = 0
        
        @retry_async(max_attempts=3, base_delay=0.01)
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RetryableError("Temporary failure")
            return "success"
        
        result = await test_function()
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_async_function_fails_permanently(self):
        """Test async function that fails all retry attempts."""
        call_count = 0
        
        @retry_async(max_attempts=3, base_delay=0.01)
        async def test_function():
            nonlocal call_count
            call_count += 1
            raise RetryableError("Permanent failure")
        
        with pytest.raises(RetryableError):
            await test_function()
        
        assert call_count == 3


class TestCircuitBreaker:
    """Test circuit breaker implementation."""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        
        @breaker
        def test_function():
            return "success"
        
        # Should work normally in closed state
        result = test_function()
        assert result == "success"
        assert breaker.state == "closed"
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        
        @breaker
        def test_function():
            raise ValueError("Test failure")
        
        # First failure
        with pytest.raises(ValueError):
            test_function()
        assert breaker.state == "closed"
        
        # Second failure - should open circuit
        with pytest.raises(ValueError):
            test_function()
        assert breaker.state == "open"
        
        # Third call should be blocked by circuit breaker
        with pytest.raises(Exception) as exc_info:
            test_function()
        assert "Circuit breaker is open" in str(exc_info.value)
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        call_count = 0
        
        @breaker
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Initial failure")
            return "success"
        
        # Cause initial failure to open circuit
        with pytest.raises(ValueError):
            test_function()
        assert breaker.state == "open"
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Next call should succeed and close circuit
        result = test_function()
        assert result == "success"
        assert breaker.state == "closed"


class TestGracefulDegradation:
    """Test graceful degradation utilities."""
    
    def test_with_fallback_success(self):
        """Test fallback when primary function succeeds."""
        def primary():
            return "primary_result"
        
        def fallback():
            return "fallback_result"
        
        func = GracefulDegradation.with_fallback(primary, fallback)
        result = func()
        
        assert result == "primary_result"
    
    def test_with_fallback_failure(self):
        """Test fallback when primary function fails."""
        def primary():
            raise ValueError("Primary failed")
        
        def fallback():
            return "fallback_result"
        
        func = GracefulDegradation.with_fallback(
            primary, 
            fallback, 
            fallback_exceptions=(ValueError,)
        )
        result = func()
        
        assert result == "fallback_result"
    
    def test_with_fallback_both_fail(self):
        """Test when both primary and fallback fail."""
        def primary():
            raise ValueError("Primary failed")
        
        def fallback():
            raise RuntimeError("Fallback failed")
        
        func = GracefulDegradation.with_fallback(
            primary, 
            fallback, 
            fallback_exceptions=(ValueError,)
        )
        
        # Should re-raise original error
        with pytest.raises(ValueError):
            func()
    
    @pytest.mark.asyncio
    async def test_with_fallback_async(self):
        """Test async fallback functionality."""
        async def primary():
            raise ValueError("Primary failed")
        
        async def fallback():
            return "fallback_result"
        
        func = await GracefulDegradation.with_fallback_async(
            primary, 
            fallback, 
            fallback_exceptions=(ValueError,)
        )
        result = await func()
        
        assert result == "fallback_result"


class TestConvenienceDecorators:
    """Test convenience retry decorators."""
    
    def test_retry_on_connection_error(self):
        """Test connection error retry decorator."""
        call_count = 0
        
        @retry_on_connection_error(max_attempts=3, base_delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection failed")
            return "success"
        
        result = test_function()
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Test rate limit retry decorator."""
        call_count = 0
        
        @retry_on_rate_limit(max_attempts=3, base_delay=0.01)
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RetryableError("Rate limited")
            return "success"
        
        result = await test_function()
        
        assert result == "success"
        assert call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])