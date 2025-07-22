"""
Agent management and hierarchy components for multi-agent orchestration.
"""

from .base import BaseAgent, Task, TaskResult, ExecutionContext, ProgressUpdate, AssistanceRequest
from .manager import ManagerAgent, Escalation, Resolution, DelegationStrategy, CoordinationProtocol
from .worker import WorkerAgent
from .hierarchy import AgentHierarchy

__all__ = [
    "BaseAgent",
    "ManagerAgent", 
    "WorkerAgent",
    "AgentHierarchy",
    "Task",
    "TaskResult",
    "ExecutionContext",
    "ProgressUpdate",
    "AssistanceRequest",
    "Escalation",
    "Resolution",
    "DelegationStrategy",
    "CoordinationProtocol",
]