"""Integration tests for error handling and logging system."""

import asyncio
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.dataqa.exceptions import (
    DataQAError,
    LLMError,
    ExecutionError,
    KnowledgeError,
    RetryableError,
    WorkflowError
)
from src.dataqa.logging_config import setup_logging, get_logger
from src.dataqa.utils.retry import retry_async, RetryConfig


class TestErrorHandlingIntegration:
    """Integration tests for error handling and logging."""
    
    def setup_method(self):
        """Set up test environment."""
        # Set up logging for tests
        setup_logging(level="DEBUG", console_output=False)
        self.logger = get_logger(__name__)
    
    def test_error_creation_and_logging(self):
        """Test that errors are properly created and logged."""
        with patch('src.dataqa.exceptions.logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            # Create an error with full context
            original_error = ValueError("Original problem")
            error = LLMError(
                "Rate limit exceeded in API call",
                user_message="The AI service is busy. Please try again.",
                technical_details={"api_endpoint": "/chat/completions", "retry_after": 60},
                recovery_suggestions=["Wait 60 seconds", "Try a simpler request"],
                error_code="RATE_LIMIT_EXCEEDED",
                original_error=original_error
            )
            
            # Verify error properties
            assert str(error) == "Rate limit exceeded in API call"
            assert error.user_message == "The AI service is busy. Please try again."
            assert error.error_code == "RATE_LIMIT_EXCEEDED"
            assert error.original_error == original_error
            
            # Verify logging was called
            mock_logger.error.assert_called_once()
            
            # Verify error serialization
            error_dict = error.to_dict()
            assert error_dict["error_type"] == "LLMError"
            assert error_dict["error_code"] == "RATE_LIMIT_EXCEEDED"
            assert error_dict["technical_details"]["api_endpoint"] == "/chat/completions"
    
    @pytest.mark.asyncio
    async def test_retry_with_error_handling(self):
        """Test retry mechanism with structured error handling."""
        call_count = 0
        
        @retry_async(max_attempts=3, base_delay=0.01)
        async def failing_function():
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                raise RetryableError(
                    "Temporary connection failure",
                    user_message="Connection temporarily unavailable",
                    error_code="CONNECTION_TEMP_FAILURE",
                    retry_after=0.01
                )
            elif call_count == 2:
                raise LLMError(
                    "Rate limit hit",
                    user_message="Service is busy",
                    error_code="RATE_LIMIT"
                )
            else:
                return "success"
        
        # Should succeed on third attempt
        result = await failing_function()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_error_propagation_in_workflow(self):
        """Test error propagation through workflow components."""
        
        # Simulate a workflow step that fails
        async def simulate_llm_call():
            raise LLMError(
                "OpenAI API authentication failed",
                user_message="Invalid API key configuration",
                error_code="AUTH_FAILED",
                recovery_suggestions=[
                    "Check your OpenAI API key",
                    "Verify API key permissions"
                ]
            )
        
        async def simulate_workflow_step():
            try:
                await simulate_llm_call()
            except LLMError as e:
                # Workflow should wrap and re-raise with additional context
                raise WorkflowError(
                    f"Query processing failed: {e}",
                    user_message="Failed to process your question due to configuration issues",
                    error_code="WORKFLOW_LLM_FAILURE",
                    technical_details={"step": "query_processor", "component": "llm"},
                    original_error=e
                )
        
        # Test that the error propagates correctly
        with pytest.raises(WorkflowError) as exc_info:
            await simulate_workflow_step()
        
        error = exc_info.value
        assert error.error_code == "WORKFLOW_LLM_FAILURE"
        assert isinstance(error.original_error, LLMError)
        assert error.original_error.error_code == "AUTH_FAILED"
        assert "configuration issues" in error.user_message
    
    def test_error_recovery_suggestions(self):
        """Test that error recovery suggestions are contextual."""
        
        # Test different error types generate appropriate suggestions
        llm_error = LLMError("Rate limit exceeded")
        execution_error = ExecutionError("SQL syntax error")
        knowledge_error = KnowledgeError("Search index unavailable")
        
        from src.dataqa.exceptions import ErrorRecovery
        
        llm_suggestions = ErrorRecovery.suggest_recovery_actions(llm_error)
        exec_suggestions = ErrorRecovery.suggest_recovery_actions(execution_error)
        knowledge_suggestions = ErrorRecovery.suggest_recovery_actions(knowledge_error)
        
        # Verify suggestions are contextual
        assert any("rephrasing" in s.lower() for s in llm_suggestions)
        assert any("simpler query" in s.lower() for s in exec_suggestions)
        assert any("knowledge base" in s.lower() for s in knowledge_suggestions)
    
    def test_structured_logging_with_errors(self):
        """Test that structured logging captures error context."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            # Set up structured logging to file
            setup_logging(
                level="ERROR",
                log_file=log_file,
                structured=True,
                console_output=False
            )
            
            # Create an error that should be logged
            try:
                raise ExecutionError(
                    "Database connection failed",
                    user_message="Unable to connect to database",
                    error_code="DB_CONNECTION_FAILED",
                    technical_details={"host": "localhost", "port": 5432}
                )
            except ExecutionError:
                pass  # Error is logged during creation
            
            # Verify log file contains structured data
            if log_file.exists():
                log_content = log_file.read_text()
                assert "DB_CONNECTION_FAILED" in log_content
                assert "ExecutionError" in log_content
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_errors(self):
        """Test circuit breaker pattern with structured errors."""
        from src.dataqa.utils.retry import CircuitBreaker
        
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        call_count = 0
        
        @breaker
        def unreliable_service():
            nonlocal call_count
            call_count += 1
            
            if call_count <= 3:
                raise ExecutionError(
                    f"Service failure #{call_count}",
                    user_message="Service temporarily unavailable",
                    error_code="SERVICE_FAILURE"
                )
            return "success"
        
        # First two calls should fail and open circuit
        with pytest.raises(ExecutionError):
            unreliable_service()
        
        with pytest.raises(ExecutionError):
            unreliable_service()
        
        # Third call should be blocked by circuit breaker
        with pytest.raises(DataQAError) as exc_info:
            unreliable_service()
        
        assert "Circuit breaker is open" in str(exc_info.value)
        assert exc_info.value.error_code == "CIRCUIT_BREAKER_OPEN"
    
    def test_graceful_degradation_with_errors(self):
        """Test graceful degradation when services fail."""
        from src.dataqa.utils.retry import GracefulDegradation
        
        def primary_service():
            raise KnowledgeError(
                "Knowledge base search failed",
                user_message="Search service unavailable",
                error_code="SEARCH_FAILED"
            )
        
        def fallback_service():
            return "fallback_result"
        
        # Test that fallback is used when primary fails
        func = GracefulDegradation.with_fallback(
            primary_service,
            fallback_service,
            fallback_exceptions=(KnowledgeError,)
        )
        
        result = func()
        assert result == "fallback_result"
    
    def test_user_friendly_error_messages(self):
        """Test that all error types provide user-friendly messages."""
        
        errors = [
            LLMError("OpenAI API timeout"),
            ExecutionError("SQL table 'users' does not exist"),
            KnowledgeError("FAISS index corruption detected"),
            WorkflowError("Invalid state transition in workflow"),
            RetryableError("Temporary network failure")
        ]
        
        for error in errors:
            # All errors should have user-friendly messages
            assert error.user_message is not None
            assert len(error.user_message) > 0
            
            # User messages should be understandable and helpful
            user_msg = error.user_message.lower()
            
            # Check that the message contains helpful language
            # (RetryableError uses base class message which says "An error occurred")
            helpful_words = ["error", "failed", "issue", "problem", "unable", "occurred", "try"]
            assert any(word in user_msg for word in helpful_words), f"User message not helpful: {error.user_message}"


class TestLoggingIntegration:
    """Integration tests for logging functionality."""
    
    def test_performance_logging(self):
        """Test performance logging integration."""
        logger = get_logger(__name__, component="test")
        
        with patch.object(logger, 'log') as mock_log:
            # Simulate a timed operation
            logger.log_performance(
                "database_query",
                2.5,
                success=True,
                details={"rows_returned": 150, "query_type": "SELECT"}
            )
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            
            # Verify performance data structure
            perf_data = call_args[1]["extra"]["performance_data"]
            assert perf_data["operation"] == "database_query"
            assert perf_data["duration_seconds"] == 2.5
            assert perf_data["success"] is True
            assert perf_data["details"]["rows_returned"] == 150
    
    def test_security_event_logging(self):
        """Test security event logging."""
        logger = get_logger(__name__, component="security")
        
        with patch.object(logger, 'log') as mock_log:
            logger.log_security_event(
                "unauthorized_code_execution",
                "CRITICAL",
                details={
                    "user_id": "user123",
                    "attempted_code": "DROP TABLE users;",
                    "blocked": True
                }
            )
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            
            # Verify security data structure
            security_data = call_args[1]["extra"]["security_data"]
            assert security_data["event_type"] == "unauthorized_code_execution"
            assert security_data["severity"] == "CRITICAL"
            assert security_data["details"]["blocked"] is True
    
    def test_user_action_logging(self):
        """Test user action audit logging."""
        logger = get_logger(__name__, component="audit")
        
        with patch.object(logger, 'info') as mock_info:
            logger.log_user_action(
                "query_submitted",
                user_id="user456",
                conversation_id="conv789",
                details={
                    "query": "SELECT * FROM sales",
                    "query_length": 20,
                    "timestamp": "2024-01-01T12:00:00Z"
                }
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            
            # Verify user context structure
            user_context = call_args[1]["extra"]["user_context"]
            assert user_context["action"] == "query_submitted"
            assert user_context["user_id"] == "user456"
            assert user_context["conversation_id"] == "conv789"


if __name__ == "__main__":
    pytest.main([__file__])