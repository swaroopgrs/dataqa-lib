"""
DataQA - A composable data agent framework.

This package provides tools for building intelligent data agents that can
interact with structured data through natural language interfaces.
"""

__version__ = "0.1.0"
__author__ = "DataQA Team"

# Import main API components
from .agent.agent import DataAgent
from .api import (
    DataQAClient,
    agent_session,
    create_agent,
    create_agent_async,
    quick_query,
    quick_query_async,
)
from .config import AgentConfig
from .models.document import Document
from .models.message import Message

__all__ = [
    # Core classes
    "DataAgent",
    "DataQAClient", 
    "AgentConfig",
    "Document",
    "Message",
    # Factory functions
    "create_agent",
    "create_agent_async",
    # Context managers
    "agent_session",
    # Convenience functions
    "quick_query",
    "quick_query_async",
]