"""Tests for logging configuration and structured logging."""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.dataqa.logging_config import (
    StructuredFormatter,
    DataQALoggerAdapter,
    setup_logging,
    get_logger,
    get_agent_logger,
    get_primitive_logger,
    get_workflow_logger,
    get_api_logger,
    LoggingContext
)


class TestStructuredFormatter:
    """Test the structured JSON formatter."""
    
    def test_basic_formatting(self):
        """Test basic log record formatting."""
        formatter = StructuredFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test_module"
        assert log_data["function"] == "test_function"
        assert log_data["line"] == 42
        assert "timestamp" in log_data
    
    def test_formatting_with_error_data(self):
        """Test formatting with error data."""
        formatter = StructuredFormatter()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.error_data = {"error_type": "TestError", "details": "test details"}
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["error_data"]["error_type"] == "TestError"
        assert log_data["error_data"]["details"] == "test details"
    
    def test_formatting_with_performance_data(self):
        """Test formatting with performance data."""
        formatter = StructuredFormatter()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Operation completed",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.performance_data = {"operation": "test_op", "duration": 1.5}
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["performance_data"]["operation"] == "test_op"
        assert log_data["performance_data"]["duration"] == 1.5
    
    def test_formatting_with_exception(self):
        """Test formatting with exception information."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=42,
            msg="Exception occurred",
            args=(),
            exc_info=exc_info
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test exception"
        assert "traceback" in log_data["exception"]


class TestDataQALoggerAdapter:
    """Test the DataQA logger adapter."""
    
    def test_adapter_initialization(self):
        """Test adapter initialization with context."""
        base_logger = logging.getLogger("test")
        adapter = DataQALoggerAdapter(base_logger, {"component": "test"})
        
        assert adapter.logger == base_logger
        assert adapter.extra["component"] == "test"
    
    def test_process_adds_context(self):
        """Test that process method adds context to log messages."""
        base_logger = logging.getLogger("test")
        adapter = DataQALoggerAdapter(base_logger, {"component": "test"})
        
        msg, kwargs = adapter.process("Test message", {})
        
        assert msg == "Test message"
        assert kwargs["extra"]["component"] == "test"
    
    def test_log_performance(self):
        """Test performance logging."""
        base_logger = logging.getLogger("test")
        adapter = DataQALoggerAdapter(base_logger)
        
        with patch.object(adapter, 'log') as mock_log:
            adapter.log_performance(
                "test_operation",
                1.5,
                success=True,
                details={"rows": 100}
            )
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            
            assert call_args[0][0] == logging.INFO
            assert "Performance: test_operation completed in 1.500s" in call_args[0][1]
            
            performance_data = call_args[1]["extra"]["performance_data"]
            assert performance_data["operation"] == "test_operation"
            assert performance_data["duration_seconds"] == 1.5
            assert performance_data["success"] is True
            assert performance_data["details"]["rows"] == 100
    
    def test_log_user_action(self):
        """Test user action logging."""
        base_logger = logging.getLogger("test")
        adapter = DataQALoggerAdapter(base_logger)
        
        with patch.object(adapter, 'info') as mock_info:
            adapter.log_user_action(
                "query_submitted",
                user_id="user123",
                conversation_id="conv456",
                details={"query_length": 50}
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            
            assert "User action: query_submitted" in call_args[0][0]
            
            user_context = call_args[1]["extra"]["user_context"]
            assert user_context["action"] == "query_submitted"
            assert user_context["user_id"] == "user123"
            assert user_context["conversation_id"] == "conv456"
            assert user_context["details"]["query_length"] == 50
    
    def test_log_security_event(self):
        """Test security event logging."""
        base_logger = logging.getLogger("test")
        adapter = DataQALoggerAdapter(base_logger)
        
        with patch.object(adapter, 'log') as mock_log:
            adapter.log_security_event(
                "unauthorized_access",
                "WARNING",
                details={"ip": "192.168.1.1"}
            )
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            
            assert call_args[0][0] == logging.WARNING
            assert "Security event: unauthorized_access" in call_args[0][1]
            
            security_data = call_args[1]["extra"]["security_data"]
            assert security_data["event_type"] == "unauthorized_access"
            assert security_data["severity"] == "WARNING"
            assert security_data["details"]["ip"] == "192.168.1.1"


class TestSetupLogging:
    """Test logging setup functionality."""
    
    def test_setup_logging_console_only(self):
        """Test logging setup with console output only."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_logging(level="DEBUG", console_output=True, rich_console=False)
            
            mock_basic_config.assert_called_once()
            call_args = mock_basic_config.call_args
            
            assert call_args[1]["level"] == logging.DEBUG
            assert len(call_args[1]["handlers"]) == 1
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            with patch('logging.basicConfig') as mock_basic_config:
                setup_logging(
                    level="INFO",
                    log_file=log_file,
                    structured=True,
                    console_output=False
                )
                
                mock_basic_config.assert_called_once()
                call_args = mock_basic_config.call_args
                
                assert call_args[1]["level"] == logging.INFO
                assert len(call_args[1]["handlers"]) == 1
    
    def test_setup_logging_structured(self):
        """Test structured logging setup."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_logging(structured=True, rich_console=False)
            
            mock_basic_config.assert_called_once()
            # Verify structured formatter is used
            handlers = mock_basic_config.call_args[1]["handlers"]
            assert len(handlers) == 1


class TestLoggerFactories:
    """Test logger factory functions."""
    
    def test_get_logger(self):
        """Test basic logger creation."""
        logger = get_logger("test.module", component="test")
        
        assert isinstance(logger, DataQALoggerAdapter)
        assert logger.extra["component"] == "test"
    
    def test_get_agent_logger(self):
        """Test agent logger creation."""
        logger = get_agent_logger("test_agent", "conv123")
        
        assert isinstance(logger, DataQALoggerAdapter)
        assert logger.extra["component"] == "agent"
        assert logger.extra["agent_name"] == "test_agent"
        assert logger.extra["conversation_id"] == "conv123"
    
    def test_get_primitive_logger(self):
        """Test primitive logger creation."""
        logger = get_primitive_logger("llm", "openai-gpt4")
        
        assert isinstance(logger, DataQALoggerAdapter)
        assert logger.extra["component"] == "primitive"
        assert logger.extra["primitive_type"] == "llm"
        assert logger.extra["primitive_name"] == "openai-gpt4"
    
    def test_get_workflow_logger(self):
        """Test workflow logger creation."""
        logger = get_workflow_logger("data_agent", "query_processor")
        
        assert isinstance(logger, DataQALoggerAdapter)
        assert logger.extra["component"] == "workflow"
        assert logger.extra["workflow_name"] == "data_agent"
        assert logger.extra["workflow_step"] == "query_processor"
    
    def test_get_api_logger(self):
        """Test API logger creation."""
        logger = get_api_logger("/api/query", "req123")
        
        assert isinstance(logger, DataQALoggerAdapter)
        assert logger.extra["component"] == "api"
        assert logger.extra["endpoint"] == "/api/query"
        assert logger.extra["request_id"] == "req123"


class TestLoggingContext:
    """Test logging context manager."""
    
    def test_logging_context(self):
        """Test logging context manager adds and removes context."""
        base_logger = logging.getLogger("test")
        adapter = DataQALoggerAdapter(base_logger, {"base": "value"})
        
        # Verify initial state
        assert adapter.extra == {"base": "value"}
        
        # Use context manager
        with LoggingContext(adapter, temp="context"):
            assert adapter.extra == {"base": "value", "temp": "context"}
        
        # Verify context is restored
        assert adapter.extra == {"base": "value"}
    
    def test_logging_context_with_exception(self):
        """Test logging context manager restores context even with exceptions."""
        base_logger = logging.getLogger("test")
        adapter = DataQALoggerAdapter(base_logger, {"base": "value"})
        
        try:
            with LoggingContext(adapter, temp="context"):
                assert adapter.extra == {"base": "value", "temp": "context"}
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Verify context is restored even after exception
        assert adapter.extra == {"base": "value"}


if __name__ == "__main__":
    pytest.main([__file__])