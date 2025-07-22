"""Structured logging configuration for DataQA."""

import json
import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'error_data'):
            log_entry["error_data"] = record.error_data
        
        if hasattr(record, 'performance_data'):
            log_entry["performance_data"] = record.performance_data
        
        if hasattr(record, 'user_context'):
            log_entry["user_context"] = record.user_context
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        return json.dumps(log_entry, default=str)


class DataQALoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds DataQA-specific context."""
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        """Initialize adapter with context."""
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Process log message and add context."""
        # Add default context
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra'].update(self.extra)
        return msg, kwargs
    
    def log_performance(
        self,
        operation: str,
        duration: float,
        *,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log performance metrics."""
        performance_data = {
            "operation": operation,
            "duration_seconds": duration,
            "success": success,
            "details": details or {}
        }
        
        level = logging.INFO if success else logging.WARNING
        self.log(
            level,
            f"Performance: {operation} completed in {duration:.3f}s",
            extra={"performance_data": performance_data}
        )
    
    def log_user_action(
        self,
        action: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log user actions for audit trail."""
        user_context = {
            "action": action,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "details": details or {}
        }
        
        self.info(
            f"User action: {action}",
            extra={"user_context": user_context}
        )
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security-related events."""
        security_data = {
            "event_type": event_type,
            "severity": severity,
            "details": details or {}
        }
        
        level = getattr(logging, severity.upper(), logging.WARNING)
        self.log(
            level,
            f"Security event: {event_type}",
            extra={"security_data": security_data}
        )


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    structured: bool = False,
    console_output: bool = True,
    rich_console: bool = True
) -> None:
    """Set up comprehensive logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        structured: Use structured JSON logging
        console_output: Enable console output
        rich_console: Use Rich console formatting
    """
    
    # Convert level string to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create handlers
    handlers = []
    
    # Console handler
    if console_output:
        if rich_console:
            console = Console()
            console_handler = RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=False,
                show_time=True
            )
            console_handler.setFormatter(
                logging.Formatter("%(message)s", datefmt="[%X]")
            )
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            if structured:
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                )
        
        console_handler.setLevel(log_level)
        handlers.append(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        
        if structured:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
                )
            )
        
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )
    
    # Reduce noise from external libraries
    _configure_external_loggers()
    
    # Log configuration
    root_logger = logging.getLogger()
    root_logger.info(f"Logging configured: level={level}, structured={structured}, file={log_file}")


def _configure_external_loggers() -> None:
    """Configure external library loggers to reduce noise."""
    external_loggers = {
        "httpx": logging.WARNING,
        "openai": logging.WARNING,
        "sentence_transformers": logging.WARNING,
        "transformers": logging.WARNING,
        "torch": logging.WARNING,
        "faiss": logging.WARNING,
        "duckdb": logging.WARNING,
        "matplotlib": logging.WARNING,
        "PIL": logging.WARNING,
        "urllib3": logging.WARNING,
        "requests": logging.WARNING
    }
    
    for logger_name, level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str, **context) -> DataQALoggerAdapter:
    """Get a DataQA logger with optional context.
    
    Args:
        name: Logger name (usually __name__)
        **context: Additional context to include in all log messages
    
    Returns:
        Configured logger adapter
    """
    base_logger = logging.getLogger(name)
    return DataQALoggerAdapter(base_logger, context)


def log_function_call(func_name: str, args: tuple = (), kwargs: Optional[Dict] = None):
    """Decorator to log function calls with arguments."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            # Log function entry
            logger.debug(
                f"Entering {func_name}",
                extra={
                    "function_call": {
                        "function": func_name,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()) if kwargs else []
                    }
                }
            )
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func_name} successfully")
                return result
            except Exception as e:
                logger.error(
                    f"Exception in {func_name}: {e}",
                    extra={
                        "function_error": {
                            "function": func_name,
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    },
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


class LoggingContext:
    """Context manager for adding temporary logging context."""
    
    def __init__(self, logger: DataQALoggerAdapter, **context):
        """Initialize logging context.
        
        Args:
            logger: Logger to add context to
            **context: Context to add
        """
        self.logger = logger
        self.context = context
        self.original_extra = logger.extra.copy()
    
    def __enter__(self):
        """Enter context and add logging context."""
        self.logger.extra.update(self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original context."""
        self.logger.extra = self.original_extra


# Pre-configured loggers for common use cases
def get_agent_logger(agent_name: str, conversation_id: Optional[str] = None) -> DataQALoggerAdapter:
    """Get logger for agent operations."""
    context = {"component": "agent", "agent_name": agent_name}
    if conversation_id:
        context["conversation_id"] = conversation_id
    return get_logger("dataqa.agent", **context)


def get_primitive_logger(primitive_type: str, primitive_name: str) -> DataQALoggerAdapter:
    """Get logger for primitive operations."""
    return get_logger(
        f"dataqa.primitives.{primitive_type}",
        component="primitive",
        primitive_type=primitive_type,
        primitive_name=primitive_name
    )


def get_workflow_logger(workflow_name: str, step: Optional[str] = None) -> DataQALoggerAdapter:
    """Get logger for workflow operations."""
    context = {"component": "workflow", "workflow_name": workflow_name}
    if step:
        context["workflow_step"] = step
    return get_logger("dataqa.workflow", **context)


def get_api_logger(endpoint: str, request_id: Optional[str] = None) -> DataQALoggerAdapter:
    """Get logger for API operations."""
    context = {"component": "api", "endpoint": endpoint}
    if request_id:
        context["request_id"] = request_id
    return get_logger("dataqa.api", **context)