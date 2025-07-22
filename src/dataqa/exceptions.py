"""Custom exception classes for DataQA components."""

import logging
from typing import Any, Dict, Optional


class DataQAError(Exception):
    """Base exception class for all DataQA errors.
    
    Provides structured error handling with user-friendly messages,
    technical details, and error recovery suggestions.
    """
    
    def __init__(
        self,
        message: str,
        *,
        user_message: Optional[str] = None,
        technical_details: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[list[str]] = None,
        error_code: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize DataQA error with structured information.
        
        Args:
            message: Technical error message for logging
            user_message: User-friendly error message
            technical_details: Additional technical context
            recovery_suggestions: List of suggested recovery actions
            error_code: Unique error code for categorization
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.user_message = user_message or self._generate_user_message(message)
        self.technical_details = technical_details or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.error_code = error_code
        self.original_error = original_error
        
        # Log the error with structured information
        self._log_error()
    
    def _generate_user_message(self, technical_message: str) -> str:
        """Generate a user-friendly message from technical details."""
        return f"An error occurred: {technical_message}"
    
    def _log_error(self) -> None:
        """Log the error with structured information."""
        logger = logging.getLogger(self.__class__.__module__)
        
        log_data = {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "technical_message": str(self),
            "user_message": self.user_message,
            "technical_details": self.technical_details,
            "recovery_suggestions": self.recovery_suggestions
        }
        
        if self.original_error:
            log_data["original_error"] = {
                "type": type(self.original_error).__name__,
                "message": str(self.original_error)
            }
        
        logger.error(
            f"DataQA Error: {self.__class__.__name__}",
            extra={"error_data": log_data},
            exc_info=self.original_error
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": str(self),
            "user_message": self.user_message,
            "technical_details": self.technical_details,
            "recovery_suggestions": self.recovery_suggestions,
            "original_error": str(self.original_error) if self.original_error else None
        }


class KnowledgeError(DataQAError):
    """Exception raised by knowledge primitive operations."""
    
    def _generate_user_message(self, technical_message: str) -> str:
        """Generate user-friendly message for knowledge errors."""
        if "connection" in technical_message.lower():
            return "Unable to connect to the knowledge base. Please check your configuration."
        elif "search" in technical_message.lower():
            return "Failed to search the knowledge base. The search service may be temporarily unavailable."
        elif "ingest" in technical_message.lower():
            return "Failed to add documents to the knowledge base. Please check the document format and try again."
        else:
            return "An issue occurred with the knowledge base. Please try again or contact support."


class ExecutionError(DataQAError):
    """Exception raised by executor primitive operations."""
    
    def _generate_user_message(self, technical_message: str) -> str:
        """Generate user-friendly message for execution errors."""
        if "sql" in technical_message.lower():
            return "There was an error executing the SQL query. Please check your data and query syntax."
        elif "python" in technical_message.lower():
            return "There was an error executing the Python code. The operation may not be supported."
        elif "connection" in technical_message.lower():
            return "Unable to connect to the database. Please check your connection settings."
        elif "permission" in technical_message.lower():
            return "You don't have permission to perform this operation. Please contact your administrator."
        else:
            return "An error occurred while executing your request. Please try a different approach."


class LLMError(DataQAError):
    """Exception raised by LLM interface operations."""
    
    def _generate_user_message(self, technical_message: str) -> str:
        """Generate user-friendly message for LLM errors."""
        if "rate limit" in technical_message.lower():
            return "The AI service is currently busy. Please wait a moment and try again."
        elif "authentication" in technical_message.lower():
            return "There's an issue with the AI service authentication. Please contact support."
        elif "timeout" in technical_message.lower():
            return "The AI service took too long to respond. Please try again with a simpler request."
        elif "connection" in technical_message.lower():
            return "Unable to connect to the AI service. Please check your internet connection."
        elif "quota" in technical_message.lower():
            return "You've reached your usage limit for the AI service. Please try again later."
        else:
            return "The AI service encountered an issue. Please try rephrasing your request."


class ConfigurationError(DataQAError):
    """Exception raised for configuration-related errors."""
    
    def _generate_user_message(self, technical_message: str) -> str:
        """Generate user-friendly message for configuration errors."""
        if "file not found" in technical_message.lower():
            return "Configuration file not found. Please check the file path and ensure it exists."
        elif "invalid yaml" in technical_message.lower():
            return "Configuration file format is invalid. Please check the YAML syntax."
        elif "missing" in technical_message.lower():
            return "Required configuration is missing. Please check your configuration file."
        elif "environment" in technical_message.lower():
            return "Required environment variables are not set. Please check your environment configuration."
        else:
            return "There's an issue with your configuration. Please review your settings."


class ValidationError(DataQAError):
    """Exception raised for data validation errors."""
    
    def _generate_user_message(self, technical_message: str) -> str:
        """Generate user-friendly message for validation errors."""
        if "required" in technical_message.lower():
            return "Required information is missing. Please provide all necessary details."
        elif "format" in technical_message.lower():
            return "The data format is incorrect. Please check your input and try again."
        elif "type" in technical_message.lower():
            return "The data type is not supported. Please use a different format."
        else:
            return "The provided data is not valid. Please check your input."


class WorkflowError(DataQAError):
    """Exception raised during workflow execution."""
    
    def _generate_user_message(self, technical_message: str) -> str:
        """Generate user-friendly message for workflow errors."""
        if "state" in technical_message.lower():
            return "There was an issue with the conversation state. Please start a new conversation."
        elif "step" in technical_message.lower():
            return "A workflow step failed. Please try your request again."
        elif "timeout" in technical_message.lower():
            return "The operation took too long to complete. Please try a simpler request."
        else:
            return "There was an issue processing your request. Please try again."


class SecurityError(DataQAError):
    """Exception raised for security-related issues."""
    
    def _generate_user_message(self, technical_message: str) -> str:
        """Generate user-friendly message for security errors."""
        if "unauthorized" in technical_message.lower():
            return "You are not authorized to perform this operation."
        elif "unsafe" in technical_message.lower():
            return "This operation is not allowed for security reasons."
        elif "blocked" in technical_message.lower():
            return "This request was blocked by security policies."
        else:
            return "This operation cannot be completed due to security restrictions."


class RetryableError(DataQAError):
    """Exception for errors that can be retried."""
    
    def __init__(self, *args, retry_after: Optional[float] = None, max_retries: int = 3, **kwargs):
        """Initialize retryable error.
        
        Args:
            retry_after: Seconds to wait before retry
            max_retries: Maximum number of retry attempts
        """
        super().__init__(*args, **kwargs)
        self.retry_after = retry_after
        self.max_retries = max_retries


# Error recovery utilities
class ErrorRecovery:
    """Utilities for error recovery and graceful degradation."""
    
    @staticmethod
    def suggest_recovery_actions(error: DataQAError) -> list[str]:
        """Suggest recovery actions based on error type and context."""
        suggestions = []
        
        if isinstance(error, LLMError):
            suggestions.extend([
                "Try rephrasing your question",
                "Break complex requests into smaller parts",
                "Wait a moment and try again"
            ])
        elif isinstance(error, ExecutionError):
            suggestions.extend([
                "Check your data for any issues",
                "Try a simpler query",
                "Verify your database connection"
            ])
        elif isinstance(error, KnowledgeError):
            suggestions.extend([
                "Check if the knowledge base is properly configured",
                "Try searching with different keywords",
                "Verify the knowledge base contains relevant information"
            ])
        elif isinstance(error, ConfigurationError):
            suggestions.extend([
                "Check your configuration file syntax",
                "Verify all required environment variables are set",
                "Review the configuration documentation"
            ])
        
        return suggestions
    
    @staticmethod
    def should_retry(error: Exception, attempt: int = 1) -> bool:
        """Determine if an error should be retried."""
        if isinstance(error, RetryableError):
            return attempt <= error.max_retries
        
        # Retry on specific error types
        if isinstance(error, (LLMError, KnowledgeError)):
            if any(keyword in str(error).lower() for keyword in ["timeout", "connection", "rate limit"]):
                return attempt <= 3
        
        return False
    
    @staticmethod
    def get_retry_delay(error: Exception, attempt: int = 1) -> float:
        """Get delay before retry attempt."""
        if isinstance(error, RetryableError) and error.retry_after:
            return error.retry_after
        
        # Exponential backoff with jitter
        import random
        base_delay = min(2 ** attempt, 60)  # Cap at 60 seconds
        jitter = random.uniform(0.1, 0.3) * base_delay
        return base_delay + jitter