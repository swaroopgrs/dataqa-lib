"""
Tests for structured logging system.
"""

import pytest
import json
import threading
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from src.dataqa.orchestration.monitoring.logging import (
    StructuredLogger,
    CorrelationManager,
    LogContext,
    LogEntry,
    LogLevel
)


class TestCorrelationManager:
    """Test correlation ID management."""
    
    @pytest.fixture
    def manager(self):
        """Create a correlation manager for testing."""
        return CorrelationManager()
    
    def test_generate_correlation_id(self, manager):
        """Test correlation ID generation."""
        corr_id = manager.generate_correlation_id()
        assert corr_id is not None
        assert len(corr_id) > 0
        
        # Should generate unique IDs
        corr_id2 = manager.generate_correlation_id()
        assert corr_id != corr_id2
    
    def test_set_get_correlation_id(self, manager):
        """Test setting and getting correlation ID."""
        test_id = "test-correlation-123"
        manager.set_correlation_id(test_id)
        
        assert manager.get_correlation_id() == test_id
    
    def test_set_get_context(self, manager):
        """Test setting and getting log context."""
        context = LogContext(
            correlation_id="test-123",
            execution_id="exec-1",
            agent_id="agent-1"
        )
        
        manager.set_context(context)
        retrieved = manager.get_context()
        
        assert retrieved is not None
        assert retrieved.correlation_id == "test-123"
        assert retrieved.execution_id == "exec-1"
        assert retrieved.agent_id == "agent-1"
    
    def test_update_context(self, manager):
        """Test updating context fields."""
        # Set initial context
        context = LogContext(correlation_id="test-123")
        manager.set_context(context)
        
        # Update context
        manager.update_context(
            execution_id="exec-1",
            agent_id="agent-1",
            custom_field="custom_value"
        )
        
        updated = manager.get_context()
        assert updated.execution_id == "exec-1"
        assert updated.agent_id == "agent-1"
        assert updated.metadata["custom_field"] == "custom_value"
    
    def test_update_context_without_existing(self, manager):
        """Test updating context when none exists."""
        manager.update_context(
            execution_id="exec-1",
            agent_id="agent-1"
        )
        
        context = manager.get_context()
        assert context is not None
        assert context.correlation_id is not None
        assert context.execution_id == "exec-1"
        assert context.agent_id == "agent-1"
    
    def test_correlation_context_manager(self, manager):
        """Test correlation context manager."""
        with manager.correlation_context(
            correlation_id="test-123",
            execution_id="exec-1"
        ) as corr_id:
            assert corr_id == "test-123"
            
            context = manager.get_context()
            assert context.correlation_id == "test-123"
            assert context.execution_id == "exec-1"
        
        # Context should be cleared after exiting
        context = manager.get_context()
        assert context is None
    
    def test_correlation_context_manager_with_previous_context(self, manager):
        """Test correlation context manager preserves previous context."""
        # Set initial context
        initial_context = LogContext(correlation_id="initial-123")
        manager.set_context(initial_context)
        
        with manager.correlation_context(correlation_id="temp-456"):
            context = manager.get_context()
            assert context.correlation_id == "temp-456"
        
        # Should restore previous context
        context = manager.get_context()
        assert context.correlation_id == "initial-123"
    
    def test_thread_safety(self, manager):
        """Test thread safety of correlation manager."""
        results = {}
        
        def worker(thread_id):
            corr_id = f"thread-{thread_id}"
            manager.set_correlation_id(corr_id)
            time.sleep(0.01)  # Small delay
            results[thread_id] = manager.get_correlation_id()
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Each thread should have its own correlation ID
        for i in range(5):
            assert results[i] == f"thread-{i}"


class TestLogContext:
    """Test LogContext model."""
    
    def test_log_context_creation(self):
        """Test creating LogContext instance."""
        context = LogContext(
            correlation_id="test-123",
            execution_id="exec-1",
            agent_id="agent-1",
            workflow_id="workflow-1"
        )
        
        assert context.correlation_id == "test-123"
        assert context.execution_id == "exec-1"
        assert context.agent_id == "agent-1"
        assert context.workflow_id == "workflow-1"
        assert context.metadata == {}
    
    def test_log_context_with_metadata(self):
        """Test LogContext with metadata."""
        metadata = {"key1": "value1", "key2": 42}
        context = LogContext(
            correlation_id="test-123",
            metadata=metadata
        )
        
        assert context.metadata == metadata


class TestLogEntry:
    """Test LogEntry model."""
    
    def test_log_entry_creation(self):
        """Test creating LogEntry instance."""
        context = LogContext(correlation_id="test-123")
        entry = LogEntry(
            level=LogLevel.INFO,
            message="Test message",
            context=context
        )
        
        assert entry.level == LogLevel.INFO
        assert entry.message == "Test message"
        assert entry.context == context
        assert entry.timestamp is not None
        assert entry.exception is None
        assert entry.performance_data == {}
        assert entry.tags == []


class TestStructuredLogger:
    """Test structured logging functionality."""
    
    @pytest.fixture
    def logger(self):
        """Create a structured logger for testing."""
        correlation_manager = CorrelationManager()
        return StructuredLogger(
            logger_name="test.logger",
            correlation_manager=correlation_manager
        )
    
    @patch('src.dataqa.orchestration.monitoring.logging.get_logger')
    def test_logger_initialization(self, mock_get_logger, logger):
        """Test logger initialization."""
        mock_get_logger.assert_called_with("test.logger")
    
    def test_create_log_entry(self, logger):
        """Test creating log entries."""
        # Set context
        logger.correlation_manager.set_correlation_id("test-123")
        
        entry = logger._create_log_entry(
            LogLevel.INFO,
            "Test message",
            agent_id="agent-1",
            execution_id="exec-1"
        )
        
        assert entry.level == LogLevel.INFO
        assert entry.message == "Test message"
        assert entry.context.correlation_id == "test-123"
        assert entry.context.agent_id == "agent-1"
        assert entry.context.execution_id == "exec-1"
    
    def test_create_log_entry_with_exception(self, logger):
        """Test creating log entry with exception."""
        exception = ValueError("Test error")
        
        entry = logger._create_log_entry(
            LogLevel.ERROR,
            "Error occurred",
            exception=exception
        )
        
        assert entry.level == LogLevel.ERROR
        assert entry.exception == "Test error"
        assert entry.stack_trace is not None
    
    @patch('src.dataqa.orchestration.monitoring.logging.get_logger')
    def test_log_entry_output(self, mock_get_logger, logger):
        """Test log entry output format."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        logger._logger = mock_logger
        
        # Set context
        logger.correlation_manager.set_correlation_id("test-123")
        
        # Log info message
        logger.info("Test message", agent_id="agent-1")
        
        # Verify logger was called
        mock_logger.info.assert_called_once()
        
        # Parse the logged JSON
        logged_data = json.loads(mock_logger.info.call_args[0][0])
        
        assert logged_data["level"] == LogLevel.INFO
        assert logged_data["message"] == "Test message"
        assert logged_data["correlation_id"] == "test-123"
        assert logged_data["agent_id"] == "agent-1"
    
    @patch('src.dataqa.orchestration.monitoring.logging.get_logger')
    def test_debug_logging(self, mock_get_logger, logger):
        """Test debug logging."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        logger._logger = mock_logger
        
        logger.debug("Debug message", component="test")
        
        mock_logger.debug.assert_called_once()
        logged_data = json.loads(mock_logger.debug.call_args[0][0])
        assert logged_data["level"] == LogLevel.DEBUG
        assert logged_data["message"] == "Debug message"
        assert logged_data["component"] == "test"
    
    @patch('src.dataqa.orchestration.monitoring.logging.get_logger')
    def test_error_logging_with_exception(self, mock_get_logger, logger):
        """Test error logging with exception."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        logger._logger = mock_logger
        
        exception = ValueError("Test error")
        logger.error("Error occurred", exception=exception, component="test")
        
        mock_logger.error.assert_called_once()
        logged_data = json.loads(mock_logger.error.call_args[0][0])
        assert logged_data["level"] == LogLevel.ERROR
        assert logged_data["message"] == "Error occurred"
        assert logged_data["exception"] == "Test error"
        assert "stack_trace" in logged_data
    
    @patch('src.dataqa.orchestration.monitoring.logging.get_logger')
    def test_log_agent_action(self, mock_get_logger, logger):
        """Test logging agent actions."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        logger._logger = mock_logger
        
        logger.log_agent_action(
            agent_id="agent-1",
            action="analyze_data",
            details={"data_size": 1000, "algorithm": "tree"},
            execution_id="exec-1"
        )
        
        mock_logger.info.assert_called_once()
        logged_data = json.loads(mock_logger.info.call_args[0][0])
        assert logged_data["message"] == "Agent action: analyze_data"
        assert logged_data["agent_id"] == "agent-1"
        assert logged_data["execution_id"] == "exec-1"
        assert logged_data["operation"] == "analyze_data"
        assert "agent_action" in logged_data["tags"]
    
    @patch('src.dataqa.orchestration.monitoring.logging.get_logger')
    def test_log_performance(self, mock_get_logger, logger):
        """Test logging performance metrics."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        logger._logger = mock_logger
        
        logger.log_performance(
            operation="data_analysis",
            duration_ms=1500.5,
            additional_metrics={"rows_processed": 10000},
            agent_id="agent-1"
        )
        
        mock_logger.info.assert_called_once()
        logged_data = json.loads(mock_logger.info.call_args[0][0])
        assert "Performance: data_analysis completed in 1500.50ms" in logged_data["message"]
        assert logged_data["operation"] == "data_analysis"
        assert logged_data["performance_data"]["duration_ms"] == 1500.5
        assert logged_data["performance_data"]["rows_processed"] == 10000
        assert "performance" in logged_data["tags"]
    
    @patch('src.dataqa.orchestration.monitoring.logging.get_logger')
    def test_log_decision(self, mock_get_logger, logger):
        """Test logging agent decisions."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        logger._logger = mock_logger
        
        logger.log_decision(
            agent_id="agent-1",
            decision_type="algorithm_choice",
            chosen_option="tree_algorithm",
            reasoning="Best for this data size",
            confidence=0.85,
            execution_id="exec-1"
        )
        
        mock_logger.info.assert_called_once()
        logged_data = json.loads(mock_logger.info.call_args[0][0])
        assert "Agent decision: algorithm_choice -> tree_algorithm" in logged_data["message"]
        assert logged_data["agent_id"] == "agent-1"
        assert logged_data["operation"] == "algorithm_choice"
        assert logged_data["metadata"]["chosen_option"] == "tree_algorithm"
        assert logged_data["metadata"]["confidence"] == 0.85
        assert "decision" in logged_data["tags"]
    
    @patch('src.dataqa.orchestration.monitoring.logging.get_logger')
    def test_log_workflow_lifecycle(self, mock_get_logger, logger):
        """Test logging workflow start and end."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        logger._logger = mock_logger
        
        # Log workflow start
        logger.log_workflow_start(
            workflow_id="workflow-1",
            execution_id="exec-1",
            participating_agents=["agent-1", "agent-2"]
        )
        
        # Log workflow end
        logger.log_workflow_end(
            workflow_id="workflow-1",
            execution_id="exec-1",
            status="completed",
            duration_ms=5000.0
        )
        
        assert mock_logger.info.call_count == 2
        
        # Check start log
        start_log = json.loads(mock_logger.info.call_args_list[0][0][0])
        assert "Workflow started: workflow-1" in start_log["message"]
        assert start_log["workflow_id"] == "workflow-1"
        assert start_log["metadata"]["participating_agents"] == ["agent-1", "agent-2"]
        assert "workflow" in start_log["tags"]
        assert "start" in start_log["tags"]
        
        # Check end log
        end_log = json.loads(mock_logger.info.call_args_list[1][0][0])
        assert "Workflow completed: workflow-1 (completed)" in end_log["message"]
        assert end_log["performance_data"]["duration_ms"] == 5000.0
        assert end_log["metadata"]["status"] == "completed"
        assert "complete" in end_log["tags"]
    
    @patch('src.dataqa.orchestration.monitoring.logging.get_logger')
    def test_operation_context_manager(self, mock_get_logger, logger):
        """Test operation context manager."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        logger._logger = mock_logger
        
        with logger.operation_context("test_operation", component="test_component"):
            time.sleep(0.01)  # Small delay to ensure duration > 0
        
        # Should log start (debug) and performance (info)
        assert mock_logger.debug.call_count == 1
        assert mock_logger.info.call_count == 1
        
        # Check debug log (start)
        debug_log = json.loads(mock_logger.debug.call_args[0][0])
        assert "Starting operation: test_operation" in debug_log["message"]
        assert debug_log["operation"] == "test_operation"
        assert debug_log["component"] == "test_component"
        
        # Check info log (performance)
        info_log = json.loads(mock_logger.info.call_args[0][0])
        assert "Performance: test_operation completed" in info_log["message"]
        assert info_log["performance_data"]["duration_ms"] > 0
    
    @patch('src.dataqa.orchestration.monitoring.logging.get_logger')
    def test_operation_context_manager_with_exception(self, mock_get_logger, logger):
        """Test operation context manager with exception."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        logger._logger = mock_logger
        
        with pytest.raises(ValueError):
            with logger.operation_context("test_operation"):
                raise ValueError("Test error")
        
        # Should log start (debug) and error
        assert mock_logger.debug.call_count == 1
        assert mock_logger.error.call_count == 1
        
        # Check error log
        error_log = json.loads(mock_logger.error.call_args[0][0])
        assert "Error in test_operation: Test error" in error_log["message"]
        assert error_log["operation"] == "test_operation"
        assert error_log["exception"] == "Test error"