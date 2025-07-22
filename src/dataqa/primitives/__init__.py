"""Core primitive interfaces for DataQA components."""

from .knowledge import KnowledgePrimitive
from .executor import ExecutorPrimitive
from .llm import LLMInterface
from .in_memory_executor import InMemoryExecutor
from .faiss_knowledge import FAISSKnowledge

__all__ = ["KnowledgePrimitive", "ExecutorPrimitive", "LLMInterface", "InMemoryExecutor", "FAISSKnowledge"]