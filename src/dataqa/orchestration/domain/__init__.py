"""
Domain knowledge integration and business rules management.
"""

from .knowledge import DomainKnowledgeManager
from .rules import BusinessRulesEngine
from .context import DomainContext

__all__ = [
    "DomainKnowledgeManager",
    "BusinessRulesEngine", 
    "DomainContext",
]