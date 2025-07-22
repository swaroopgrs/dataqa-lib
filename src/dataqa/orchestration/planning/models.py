"""
Planning-specific models and data structures.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from ..models import ExecutionStatus, CapabilityType


class ReplanningTriggerType(str, Enum):
    """Types of replanning triggers."""
    STEP_FAILURE = "step_failure"
    QUALITY_THRESHOLD = "quality_threshold"
    RESOURCE_CONSTRAINT = "resource_constraint"
    AGENT_UNAVAILABLE = "agent_unavailable"
    CONTEXT_CHANGE = "context_change"
    USER_REQUEST = "user_request"
    TIMEOUT = "timeout"
    DEPENDENCY_FAILURE = "dependency_failure"


class ContextPreservationStrategy(str, Enum):
    """Strategies for preserving context during replanning."""
    PRESERVE_ALL = "preserve_all"
    PRESERVE_SUCCESSFUL = "preserve_successful"
    PRESERVE_CRITICAL = "preserve_critical"
    MINIMAL_PRESERVATION = "minimal_preservation"


class ExecutionStep(BaseModel):
    """Individual step in an execution plan."""
    step_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    agent_id: Optional[str] = None
    capability_required: Optional[CapabilityType] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    success_criteria: List[str] = Field(default_factory=list)
    quality_threshold: Optional[float] = None
    timeout_seconds: Optional[int] = None
    
    class Config:
        use_enum_values = True


class IntermediateResult(BaseModel):
    """Intermediate result from an execution step."""
    result_id: str = Field(default_factory=lambda: str(uuid4()))
    step_id: str
    result_type: str
    data: Any
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    preserved: bool = True  # Whether to preserve during replanning


class Plan(BaseModel):
    """Execution plan for multi-agent workflow."""
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    version: int = 1
    steps: List[ExecutionStep] = Field(default_factory=list)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)  # step_id -> [prerequisite_step_ids]
    estimated_duration_minutes: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    parent_plan_id: Optional[str] = None  # For replanned versions
    replanning_context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReplanningTrigger(BaseModel):
    """Trigger condition for replanning."""
    trigger_id: str = Field(default_factory=lambda: str(uuid4()))
    trigger_type: ReplanningTriggerType
    condition: str  # Condition expression
    description: str
    priority: int = 1  # 1 = highest priority
    enabled: bool = True
    
    class Config:
        use_enum_values = True


class ReplanningEvent(BaseModel):
    """Event that triggered replanning."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    trigger_type: ReplanningTriggerType
    trigger_description: str
    occurred_at: datetime = Field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = Field(default_factory=dict)
    replanning_successful: bool = False
    new_plan_id: Optional[str] = None
    preserved_results: List[str] = Field(default_factory=list)  # result_ids
    
    class Config:
        use_enum_values = True


class ExecutionContext(BaseModel):
    """Context for plan execution and replanning."""
    context_id: str = Field(default_factory=lambda: str(uuid4()))
    user_query: str
    available_agents: List[str] = Field(default_factory=list)  # agent_ids
    agent_capabilities: Dict[str, List[CapabilityType]] = Field(default_factory=dict)
    domain_constraints: Dict[str, Any] = Field(default_factory=dict)
    quality_requirements: Dict[str, float] = Field(default_factory=dict)
    resource_limits: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)