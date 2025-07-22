"""
Base agent class for multi-agent orchestration.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

from ..models import (
    AgentCapability,
    AgentConfiguration,
    AgentRole,
    ExecutionStatus,
    TaskAssignment,
)


class Task(BaseModel):
    """Represents a task to be executed by an agent."""
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    required_capabilities: List[str] = Field(default_factory=list)
    priority: int = 1
    deadline: Optional[datetime] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TaskResult(BaseModel):
    """Result of task execution."""
    task_id: str
    agent_id: str
    status: ExecutionStatus
    outputs: Dict[str, Any] = Field(default_factory=dict)
    execution_time_seconds: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class ProgressUpdate(BaseModel):
    """Progress update from an agent."""
    agent_id: str
    task_id: str
    progress_percentage: float = 0.0
    status_message: str
    intermediate_results: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AssistanceRequest(BaseModel):
    """Request for assistance from another agent."""
    requesting_agent_id: str
    task_id: str
    assistance_type: str
    description: str
    required_capabilities: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    urgency: str = "normal"  # "low", "normal", "high", "critical"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExecutionContext(BaseModel):
    """Context for task execution."""
    session_id: str
    workflow_id: str
    domain_context: Optional[Dict[str, Any]] = None
    available_resources: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent orchestration system.
    
    Provides common functionality for agent identification, capability management,
    and basic lifecycle operations.
    """
    
    def __init__(self, config: AgentConfiguration):
        """Initialize the agent with configuration."""
        self.config = config
        self.agent_id = config.agent_id
        self.name = config.name
        self.role = config.role
        self.capabilities = {cap.capability_id: cap for cap in config.capabilities}
        self.specialization = config.specialization
        self.max_concurrent_tasks = config.max_concurrent_tasks
        self.priority_level = config.priority_level
        self.enabled = config.enabled
        self.metadata = config.metadata.copy()
        
        # Runtime state
        self._active_tasks: Dict[str, Task] = {}
        self._task_history: List[TaskResult] = []
        self._status = ExecutionStatus.PENDING
        self._last_heartbeat = datetime.utcnow()
    
    @property
    def is_available(self) -> bool:
        """Check if agent is available for new tasks."""
        return (
            self.enabled and 
            self._status in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING] and
            len(self._active_tasks) < self.max_concurrent_tasks
        )
    
    @property
    def active_task_count(self) -> int:
        """Get number of currently active tasks."""
        return len(self._active_tasks)
    
    @property
    def capability_types(self) -> Set[str]:
        """Get set of capability types this agent supports."""
        return {
            cap.capability_type if isinstance(cap.capability_type, str) else cap.capability_type.value
            for cap in self.capabilities.values()
        }
    
    def has_capability(self, capability_type: str) -> bool:
        """Check if agent has a specific capability type."""
        return capability_type in self.capability_types
    
    def get_capability(self, capability_id: str) -> Optional[AgentCapability]:
        """Get a specific capability by ID."""
        return self.capabilities.get(capability_id)
    
    def can_handle_task(self, task: Task) -> bool:
        """
        Check if this agent can handle the given task.
        
        Args:
            task: Task to evaluate
            
        Returns:
            True if agent can handle the task, False otherwise
        """
        if not self.is_available:
            return False
        
        # Check if agent has required capabilities
        for required_cap in task.required_capabilities:
            if not self.has_capability(required_cap):
                return False
        
        return True
    
    @abstractmethod
    async def execute_task(self, task: Task, context: ExecutionContext) -> TaskResult:
        """
        Execute a task with the given context.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            Task execution result
        """
        pass
    
    async def start_task(self, task: Task, context: ExecutionContext) -> None:
        """
        Start executing a task (non-blocking).
        
        Args:
            task: Task to start
            context: Execution context
        """
        if not self.can_handle_task(task):
            raise ValueError(f"Agent {self.agent_id} cannot handle task {task.task_id}")
        
        self._active_tasks[task.task_id] = task
        self._status = ExecutionStatus.RUNNING
    
    async def complete_task(self, task_id: str, result: TaskResult) -> None:
        """
        Mark a task as completed and update internal state.
        
        Args:
            task_id: ID of completed task
            result: Task execution result
        """
        if task_id in self._active_tasks:
            del self._active_tasks[task_id]
            self._task_history.append(result)
        
        if not self._active_tasks:
            self._status = ExecutionStatus.PENDING
    
    async def cancel_task(self, task_id: str, reason: str = "Cancelled") -> None:
        """
        Cancel an active task.
        
        Args:
            task_id: ID of task to cancel
            reason: Reason for cancellation
        """
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            result = TaskResult(
                task_id=task_id,
                agent_id=self.agent_id,
                status=ExecutionStatus.CANCELLED,
                error_message=reason
            )
            await self.complete_task(task_id, result)
    
    async def report_progress(self, progress: ProgressUpdate) -> None:
        """
        Report progress on a task.
        
        Args:
            progress: Progress update information
        """
        # Default implementation - can be overridden by subclasses
        self._last_heartbeat = datetime.utcnow()
    
    async def request_assistance(self, assistance_request: AssistanceRequest) -> None:
        """
        Request assistance from other agents.
        
        Args:
            assistance_request: Details of assistance needed
        """
        # Default implementation - can be overridden by subclasses
        pass
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.value if hasattr(self.role, 'value') else self.role,
            "status": self._status.value if hasattr(self._status, 'value') else self._status,
            "is_available": self.is_available,
            "active_tasks": len(self._active_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "capabilities": list(self.capability_types),
            "last_heartbeat": self._last_heartbeat.isoformat(),
            "task_history_count": len(self._task_history)
        }
    
    def __repr__(self) -> str:
        role_value = self.role.value if hasattr(self.role, 'value') else self.role
        return f"{self.__class__.__name__}(id={self.agent_id}, name={self.name}, role={role_value})"