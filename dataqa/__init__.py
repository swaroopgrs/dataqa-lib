
"""
dataqa: A powerful, modular library for building data-centric AI agents.
"""

# Expose the default local client for easy, out-of-the-box use.
from dataqa.integrations.local.client import LocalClient

# Expose the core data contracts for requests and responses.
from dataqa.core.client import CoreRequest, CoreResponse, CoreConversationTurn, DataQAClient

__all__ = [
    "LocalClient",
    "DataQAClient",
    "CoreRequest",
    "CoreResponse",
    "CoreConversationTurn",
]