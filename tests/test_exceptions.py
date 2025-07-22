"""Tests for custom exception classes and error handling."""

import pytest
from unittest.mock import patch, MagicMock

from src.dataqa.exceptions import (
    DataQAError,
    KnowledgeError,
    ExecutionError,
    LLMError,
    ConfigurationError,
    ValidationError,
    WorkflowError,
    SecurityError,
    RetryableError,
    ErrorRecovery
)


class TestDataQAError:
    """Test the base DataQA error class."""
    
    def test_basic_error_creation(self):
        """Test basic error creation with message."""
        error = DataQAError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.user_message == "An error occurred: Test error message"
        assert error.technical_details == {}
        assert error.recovery_suggestions == []
        assert error.error_code is None
        assert error.original_error is None
    
    def test_error_with_all_parameters(self):
        """Test error creation with all parameters."""
        original_error = ValueError("Original error")
        
        error = DataQAError(
            "Technical message",
            user_message="User-friendly message",
            technical_details={"key": "value"},
            recovery_suggestions=["Try again", "Check config"],
            error_code="TEST_ERROR",
            original_error=original_error
        )
        
        assert str(error) == "Technical message"
        assert error.user_message == "User-friendly message"
        assert error.technical_details == {"key": "value"}
        assert error.recovery_suggestions == ["Try again", "Check config"]
        assert error.error_code == "TEST_ERROR"
        assert error.original_error == original_error
    
    def test_error_to_dict(self):
        """Test error serialization to dictionary."""
        original_error = ValueError("Original error")
        
        error = DataQAError(
            "Technical message",
            user_message="User-friendly message",
            technical_details={"key": "value"},
            recovery_suggestions=["Try again"],
            error_code="TEST_ERROR",
            original_error=original_error
        )
        
        error_dict = error.to_dict()
        
        expected = {
            "error_type": "DataQAError",
            "error_code": "TEST_ERROR",
            "message": "Technical message",
            "user_message": "User-friendly message",
            "technical_details": {"key": "value"},
            "recovery_suggestions": ["Try again"],
            "original_error": "Original error"
        }
        
        assert error_dict == expected
    
    @patch('src.dataqa.exceptions.logging.getLogger')
    def test_error_logging(self, mock_get_logger):
        """Test that errors are logged with structured information."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        original_error = ValueError("Original error")
        
        error = DataQAError(
            "Technical message",
            user_message="User-friendly message",
            error_code="TEST_ERROR",
            original_error=original_error
        )
        
        # Verify logger was called
        mock_get_logger.assert_called_once()
        mock_logger.error.assert_called_once()
        
        # Check the log call arguments
        call_args = mock_logger.error.call_args
        assert "DataQA Error: DataQAError" in call_args[0][0]
        assert "error_data" in call_args[1]["extra"]
        assert call_args[1]["exc_info"] == original_error


class TestSpecificErrorTypes:
    """Test specific error type implementations."""
    
    def test_knowledge_error_user_messages(self):
        """Test KnowledgeError generates appropriate user messages."""
        # Connection error
        error = KnowledgeError("Connection failed to knowledge base")
        assert "connect to the knowledge base" in error.user_message
        
        # Search error
        error = KnowledgeError("Search operation failed")
        assert "search the knowledge base" in error.user_message
        
        # Ingestion error
        error = KnowledgeError("Failed to ingest documents")
        assert "add documents to the knowledge base" in error.user_message
        
        # Generic error
        error = KnowledgeError("Unknown knowledge error")
        assert "knowledge base" in error.user_message
    
    def test_execution_error_user_messages(self):
        """Test ExecutionError generates appropriate user messages."""
        # SQL error
        error = ExecutionError("SQL syntax error in query")
        assert "SQL query" in error.user_message
        
        # Python error
        error = ExecutionError("Python execution failed")
        assert "Python code" in error.user_message
        
        # Connection error
        error = ExecutionError("Database connection failed")
        assert "connect to the database" in error.user_message
        
        # Permission error
        error = ExecutionError("Permission denied for operation")
        assert "permission" in error.user_message
    
    def test_llm_error_user_messages(self):
        """Test LLMError generates appropriate user messages."""
        # Rate limit error
        error = LLMError("Rate limit exceeded")
        assert "service is currently busy" in error.user_message
        
        # Authentication error
        error = LLMError("Authentication failed")
        assert "authentication" in error.user_message
        
        # Timeout error
        error = LLMError("Request timeout")
        assert "took too long" in error.user_message
        
        # Connection error
        error = LLMError("Connection error")
        assert "connect to the AI service" in error.user_message
        
        # Quota error
        error = LLMError("Quota exceeded")
        assert "usage limit" in error.user_message
    
    def test_configuration_error_user_messages(self):
        """Test ConfigurationError generates appropriate user messages."""
        # File not found
        error = ConfigurationError("Configuration file not found")
        assert "file not found" in error.user_message
        
        # Invalid YAML
        error = ConfigurationError("Invalid YAML syntax")
        assert "format is invalid" in error.user_message
        
        # Missing configuration
        error = ConfigurationError("Missing required configuration")
        assert "configuration is missing" in error.user_message
        
        # Environment variables
        error = ConfigurationError("Environment variables not set")
        assert "environment variables" in error.user_message
    
    def test_validation_error_user_messages(self):
        """Test ValidationError generates appropriate user messages."""
        # Required field
        error = ValidationError("Required field missing")
        assert "Required information is missing" in error.user_message
        
        # Format error
        error = ValidationError("Invalid format")
        assert "format is incorrect" in error.user_message
        
        # Type error
        error = ValidationError("Invalid type")
        assert "type is not supported" in error.user_message


class TestRetryableError:
    """Test RetryableError functionality."""
    
    def test_retryable_error_creation(self):
        """Test RetryableError creation with retry parameters."""
        error = RetryableError(
            "Temporary failure",
            retry_after=5.0,
            max_retries=3
        )
        
        assert str(error) == "Temporary failure"
        assert error.retry_after == 5.0
        assert error.max_retries == 3
    
    def test_retryable_error_defaults(self):
        """Test RetryableError with default parameters."""
        error = RetryableError("Temporary failure")
        
        assert error.retry_after is None
        assert error.max_retries == 3


class TestErrorRecovery:
    """Test error recovery utilities."""
    
    def test_suggest_recovery_actions_llm_error(self):
        """Test recovery suggestions for LLM errors."""
        error = LLMError("Rate limit exceeded")
        suggestions = ErrorRecovery.suggest_recovery_actions(error)
        
        assert "Try rephrasing your question" in suggestions
        assert "Break complex requests into smaller parts" in suggestions
        assert "Wait a moment and try again" in suggestions
    
    def test_suggest_recovery_actions_execution_error(self):
        """Test recovery suggestions for execution errors."""
        error = ExecutionError("SQL syntax error")
        suggestions = ErrorRecovery.suggest_recovery_actions(error)
        
        assert "Check your data for any issues" in suggestions
        assert "Try a simpler query" in suggestions
        assert "Verify your database connection" in suggestions
    
    def test_suggest_recovery_actions_knowledge_error(self):
        """Test recovery suggestions for knowledge errors."""
        error = KnowledgeError("Search failed")
        suggestions = ErrorRecovery.suggest_recovery_actions(error)
        
        assert "Check if the knowledge base is properly configured" in suggestions
        assert "Try searching with different keywords" in suggestions
    
    def test_suggest_recovery_actions_configuration_error(self):
        """Test recovery suggestions for configuration errors."""
        error = ConfigurationError("Invalid config")
        suggestions = ErrorRecovery.suggest_recovery_actions(error)
        
        assert "Check your configuration file syntax" in suggestions
        assert "Verify all required environment variables are set" in suggestions
    
    def test_should_retry_retryable_error(self):
        """Test retry logic for RetryableError."""
        error = RetryableError("Temporary failure", max_retries=3)
        
        assert ErrorRecovery.should_retry(error, attempt=1) is True
        assert ErrorRecovery.should_retry(error, attempt=3) is True
        assert ErrorRecovery.should_retry(error, attempt=4) is False
    
    def test_should_retry_llm_timeout(self):
        """Test retry logic for LLM timeout errors."""
        error = LLMError("Connection timeout")
        
        assert ErrorRecovery.should_retry(error, attempt=1) is True
        assert ErrorRecovery.should_retry(error, attempt=3) is True
        assert ErrorRecovery.should_retry(error, attempt=4) is False
    
    def test_should_retry_non_retryable(self):
        """Test retry logic for non-retryable errors."""
        error = ValidationError("Invalid input")
        
        assert ErrorRecovery.should_retry(error, attempt=1) is False
    
    def test_get_retry_delay_retryable_error(self):
        """Test retry delay calculation for RetryableError."""
        error = RetryableError("Temporary failure", retry_after=10.0)
        
        delay = ErrorRecovery.get_retry_delay(error, attempt=1)
        assert delay == 10.0
    
    def test_get_retry_delay_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        error = LLMError("Rate limit")
        
        delay1 = ErrorRecovery.get_retry_delay(error, attempt=1)
        delay2 = ErrorRecovery.get_retry_delay(error, attempt=2)
        delay3 = ErrorRecovery.get_retry_delay(error, attempt=3)
        
        # Should increase exponentially (with jitter)
        assert delay1 < delay2 < delay3
        assert delay3 <= 60  # Should be capped at 60 seconds


class TestWorkflowError:
    """Test WorkflowError functionality."""
    
    def test_workflow_error_user_messages(self):
        """Test WorkflowError generates appropriate user messages."""
        # State error
        error = WorkflowError("Invalid state transition")
        assert "conversation state" in error.user_message
        
        # Step error
        error = WorkflowError("Workflow step failed")
        assert "workflow step failed" in error.user_message
        
        # Timeout error
        error = WorkflowError("Operation timeout")
        assert "took too long" in error.user_message


class TestSecurityError:
    """Test SecurityError functionality."""
    
    def test_security_error_user_messages(self):
        """Test SecurityError generates appropriate user messages."""
        # Unauthorized error
        error = SecurityError("Unauthorized access")
        assert "not authorized" in error.user_message
        
        # Unsafe operation
        error = SecurityError("Unsafe code detected")
        assert "not allowed for security reasons" in error.user_message
        
        # Blocked operation
        error = SecurityError("Operation blocked")
        assert "blocked by security policies" in error.user_message


if __name__ == "__main__":
    pytest.main([__file__])