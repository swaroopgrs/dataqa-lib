"""
dataqa: A powerful, modular library for building data-centric AI agents.
"""

# Expose the default local client for easy, out-of-the-box use.
# Expose the core data contracts for requests and responses.
from dataqa.core.client import (
    CoreConversationTurn,
    CoreRequest,
    CoreResponse,
    DataQAClient,
)
from dataqa.integrations.local.client import LocalClient

__all__ = [
    "LocalClient",
    "DataQAClient",
    "CoreRequest",
    "CoreResponse",
    "CoreConversationTurn",
]