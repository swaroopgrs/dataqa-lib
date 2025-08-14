"""
DBC Service Integration Package

This package provides integration with the DatabaseConnect (DBC) service,
enabling the DataQA library to work through DBC's callable functions
instead of direct client connections.
"""

from dataqa.dbc.models import (
    ConversationTurn,
    DBCRequest,
    DBCResponse,
    StepResponse,
    DBCClientConfig,
)
from dataqa.dbc.errors import DBCCallableError, DBCClientError

__all__ = [
    "ConversationTurn",
    "DBCRequest", 
    "DBCResponse",
    "StepResponse",
    "DBCClientConfig",
    "DBCCallableError",
    "DBCClientError",
]