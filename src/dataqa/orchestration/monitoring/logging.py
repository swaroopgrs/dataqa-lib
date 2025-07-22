"""
Structured logging system with correlation IDs and diagnostic context.

Provides comprehensive logging capabilities for multi-agent orchestration
with proper correlation tracking and contextual information.
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from ...logging_config import get_logger


class LogLevel:
    """Log level constants."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogContext(BaseModel):
    """Context information for structured logging."""
    
    correlation_id: str = Field(description="Correlation ID for request tracing")
    execution_id: Optional[str] = Field(None, description="Execution identifier")
    agent_id: Optional[str] = Field(None, description="Agent identifier")
    workflow_id: Optional[str] = Field(None, description="Workflow identifier")
    step_id: Optional[str] = Field(None, description="Step identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    component: Optional[str] = Field(None, description="Component name")
    operation: Optional[str] = Field(None, description="Operation being performed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class LogEntry(BaseModel):
    """Structured log entry."""
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    level: str = Field(description="Log level")
    message: str = Field(description="Log message")
    context: LogContext = Field(description="Log context")
    exception: Optional[str] = Field(None, description="Exception information")
    stack_trace: Optional[str] = Field(None, description="Stack trace")
    performance_data: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    tags: List[str] = Field(default_factory=list, description="Log tags for filtering")


@dataclass
class CorrelationManager:
    """
    Manages correlation IDs for request tracing across agent workflows.
    
    Provides thread-local storage for correlation context and automatic
    propagation across agent boundaries.
    """
    
    _local: threading.local = field(default_factory=threading.local)
    
    def generate_correlation_id(self) -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set the correlation ID for the current thread."""
        self._local.correlation_id = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        """Get the correlation ID for the current thread."""
        return getattr(self._local, 'correlation_id', None)
    
    def set_context(self, context: LogContext) -> None:
        """Set the full log context for the current thread."""
        self._local.context = context
    
    def get_context(self) -> Optional[LogContext]:
        """Get the log context for the current thread."""
        return getattr(self._local, 'context', None)
    
    def update_context(self, **kwargs) -> None:
        """Update specific fields in the current context."""
        current_context = self.get_context()
        if current_context:
            # Update existing context
            for key, value in kwargs.items():
                if hasattr(current_context, key):
                    setattr(current_context, key, value)
                else:
                    current_context.metadata[key] = value
        else:
            # Create new context with correlation ID
            correlation_id = self.get_correlation_id() or self.generate_correlation_id()
            context = LogContext(correlation_id=correlation_id, **kwargs)
            self.set_context(context)
    
    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None, **context_kwargs):
        """Context manager for correlation tracking."""
        # Generate correlation ID if not provided
        if correlation_id is None:
            correlation_id = self.generate_correlation_id()
        
        # Save previous context
        previous_context = self.get_context()
        
        # Set new context
        new_context = LogContext(correlation_id=correlation_id, **context_kwargs)
        self.set_context(new_context)
        
        try:
            yield correlation_id
        finally:
            # Restore previous context
            if previous_context:
                self.set_context(previous_context)
            else:
                # Clear context
                if hasattr(self._local, 'context'):
                    delattr(self._local, 'context')


@dataclass
class StructuredLogger:
    """
    Structured logger with correlation ID support and diagnostic context.
    
    Provides comprehensive logging capabilities for multi-agent workflows
    with automatic context propagation and structured output.
    """
    
    logger_name: str = "dataqa.orchestration"
    correlation_manager: CorrelationManager = field(default_factory=CorrelationManager)
    _logger: logging.Logger = field(init=False)
    _log_handlers: List[logging.Handler] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize the logger."""
        self._logger = get_logger(self.logger_name)
    
    def _create_log_entry(self, level: str, message: str, 
                         exception: Optional[Exception] = None,
                         performance_data: Optional[Dict[str, Any]] = None,
                         tags: Optional[List[str]] = None,
                         **context_kwargs) -> LogEntry:
        """Create a structured log entry."""
        # Get current context or create new one
        context = self.correlation_manager.get_context()
        if not context:
            correlation_id = self.correlation_manager.generate_correlation_id()
            context = LogContext(correlation_id=correlation_id)
        
        # Update context with any provided kwargs
        for key, value in context_kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
            else:
                context.metadata[key] = value
        
        # Handle exception information
        exception_str = None
        stack_trace = None
        if exception:
            exception_str = str(exception)
            import traceback
            stack_trace = traceback.format_exc()
        
        return LogEntry(
            level=level,
            message=message,
            context=context,
            exception=exception_str,
            stack_trace=stack_trace,
            performance_data=performance_data or {},
            tags=tags or []
        )
    
    def _log_entry(self, entry: LogEntry) -> None:
        """Log a structured entry."""
        # Convert to JSON for structured logging
        log_data = {
            "timestamp": entry.timestamp.isoformat(),
            "level": entry.level,
            "message": entry.message,
            "correlation_id": entry.context.correlation_id,
            "execution_id": entry.context.execution_id,
            "agent_id": entry.context.agent_id,
            "workflow_id": entry.context.workflow_id,
            "step_id": entry.context.step_id,
            "user_id": entry.context.user_id,
            "session_id": entry.context.session_id,
            "component": entry.context.component,
            "operation": entry.context.operation,
            "metadata": entry.context.metadata,
            "performance_data": entry.performance_data,
            "tags": entry.tags
        }
        
        if entry.exception:
            log_data["exception"] = entry.exception
        
        if entry.stack_trace:
            log_data["stack_trace"] = entry.stack_trace
        
        # Log at appropriate level
        log_message = json.dumps(log_data, default=str)
        
        if entry.level == LogLevel.DEBUG:
            self._logger.debug(log_message)
        elif entry.level == LogLevel.INFO:
            self._logger.info(log_message)
        elif entry.level == LogLevel.WARNING:
            self._logger.warning(log_message)
        elif entry.level == LogLevel.ERROR:
            self._logger.error(log_message)
        elif entry.level == LogLevel.CRITICAL:
            self._logger.critical(log_message)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        entry = self._create_log_entry(LogLevel.DEBUG, message, **kwargs)
        self._log_entry(entry)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        entry = self._create_log_entry(LogLevel.INFO, message, **kwargs)
        self._log_entry(entry)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        entry = self._create_log_entry(LogLevel.WARNING, message, **kwargs)
        self._log_entry(entry)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log error message."""
        entry = self._create_log_entry(LogLevel.ERROR, message, exception=exception, **kwargs)
        self._log_entry(entry)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log critical message."""
        entry = self._create_log_entry(LogLevel.CRITICAL, message, exception=exception, **kwargs)
        self._log_entry(entry)
    
    def log_agent_action(self, agent_id: str, action: str, details: Dict[str, Any],
                        execution_id: Optional[str] = None, step_id: Optional[str] = None) -> None:
        """Log an agent action with structured context."""
        self.info(
            f"Agent action: {action}",
            agent_id=agent_id,
            execution_id=execution_id,
            step_id=step_id,
            operation=action,
            component="agent",
            tags=["agent_action"],
            **details
        )
    
    def log_performance(self, operation: str, duration_ms: float, 
                       additional_metrics: Optional[Dict[str, Any]] = None,
                       **context_kwargs) -> None:
        """Log performance metrics."""
        performance_data = {
            "duration_ms": duration_ms,
            **(additional_metrics or {})
        }
        
        self.info(
            f"Performance: {operation} completed in {duration_ms:.2f}ms",
            operation=operation,
            performance_data=performance_data,
            tags=["performance"],
            **context_kwargs
        )
    
    def log_decision(self, agent_id: str, decision_type: str, chosen_option: str,
                    reasoning: str, confidence: float, **context_kwargs) -> None:
        """Log agent decision points."""
        self.info(
            f"Agent decision: {decision_type} -> {chosen_option}",
            agent_id=agent_id,
            component="decision_engine",
            operation=decision_type,
            metadata={
                "chosen_option": chosen_option,
                "reasoning": reasoning,
                "confidence": confidence
            },
            tags=["decision", "agent_decision"],
            **context_kwargs
        )
    
    def log_escalation(self, agent_id: str, escalation_reason: str, 
                      escalation_target: str, **context_kwargs) -> None:
        """Log escalation events."""
        self.warning(
            f"Escalation: {escalation_reason}",
            agent_id=agent_id,
            component="escalation",
            operation="escalate",
            metadata={
                "escalation_reason": escalation_reason,
                "escalation_target": escalation_target
            },
            tags=["escalation"],
            **context_kwargs
        )
    
    def log_workflow_start(self, workflow_id: str, execution_id: str, 
                          participating_agents: List[str], **context_kwargs) -> None:
        """Log workflow start."""
        self.info(
            f"Workflow started: {workflow_id}",
            workflow_id=workflow_id,
            execution_id=execution_id,
            component="workflow",
            operation="start",
            metadata={
                "participating_agents": participating_agents
            },
            tags=["workflow", "start"],
            **context_kwargs
        )
    
    def log_workflow_end(self, workflow_id: str, execution_id: str, 
                        status: str, duration_ms: float, **context_kwargs) -> None:
        """Log workflow completion."""
        self.info(
            f"Workflow completed: {workflow_id} ({status})",
            workflow_id=workflow_id,
            execution_id=execution_id,
            component="workflow",
            operation="complete",
            performance_data={"duration_ms": duration_ms},
            metadata={"status": status},
            tags=["workflow", "complete"],
            **context_kwargs
        )
    
    def log_error_with_context(self, error: Exception, operation: str, 
                              **context_kwargs) -> None:
        """Log error with full context."""
        # Remove component from context_kwargs if it exists to avoid conflict
        context_kwargs.pop('component', None)
        
        self.error(
            f"Error in {operation}: {str(error)}",
            exception=error,
            operation=operation,
            component="error_handler",
            tags=["error"],
            **context_kwargs
        )
    
    @contextmanager
    def operation_context(self, operation: str, component: Optional[str] = None,
                         **context_kwargs):
        """Context manager for operation logging."""
        start_time = datetime.now(timezone.utc)
        
        # Update context
        self.correlation_manager.update_context(
            operation=operation,
            component=component,
            **context_kwargs
        )
        
        try:
            self.debug(f"Starting operation: {operation}", operation=operation, component=component)
            yield
            
            # Log successful completion
            end_time = datetime.now(timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000
            self.log_performance(operation, duration_ms, component=component)
            
        except Exception as e:
            # Log error
            end_time = datetime.now(timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            self.log_error_with_context(
                e, operation, 
                component=component,
                performance_data={"duration_ms": duration_ms}
            )
            raise


# Global instances for easy access
correlation_manager = CorrelationManager()
structured_logger = StructuredLogger(correlation_manager=correlation_manager)