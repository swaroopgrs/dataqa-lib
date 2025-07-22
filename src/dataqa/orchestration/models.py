"""
Core Pydantic models for advanced multi-agent orchestration.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class CapabilityType(str, Enum):
    """Types of agent capabilities."""
    DATA_RETRIEVAL = "data_retrieval"
    DATA_ANALYSIS = "data_analysis" 
    VISUALIZATION = "visualization"
    CODE_GENERATION = "code_generation"
    DOMAIN_EXPERTISE = "domain_expertise"
    COORDINATION = "coordination"
    APPROVAL = "approval"


class AgentRole(str, Enum):
    """Agent roles in the hierarchy."""
    MANAGER = "manager"
    WORKER = "worker"
    SPECIALIST = "specialist"


class AgentType(str, Enum):
    """Types of agents in the system."""
    MANAGER = "manager"
    WORKER = "worker"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"


class ExecutionPhase(str, Enum):
    """Phases of execution."""
    PLANNING = "planning"
    PREPARATION = "preparation"
    EXECUTION = "execution"
    VALIDATION = "validation"
    COMPLETION = "completion"


class ExecutionStatus(str, Enum):
    """Execution status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class ResourceRequirements(BaseModel):
    """Resource requirements for agent capabilities."""
    cpu_cores: Optional[int] = None
    memory_mb: Optional[int] = None
    gpu_required: bool = False
    network_access: bool = True
    storage_mb: Optional[int] = None
    execution_timeout_seconds: int = 300


class QualityGuarantee(BaseModel):
    """Quality guarantees for agent capabilities."""
    accuracy_threshold: Optional[float] = None
    response_time_ms: Optional[int] = None
    availability_percentage: Optional[float] = None
    error_rate_threshold: Optional[float] = None


class InputRequirement(BaseModel):
    """Input requirements for agent capabilities."""
    name: str
    type: str
    required: bool = True
    description: Optional[str] = None
    validation_schema: Optional[Dict[str, Any]] = None


class OutputSpecification(BaseModel):
    """Output specifications for agent capabilities."""
    name: str
    type: str
    description: Optional[str] = None
    output_schema: Optional[Dict[str, Any]] = None


class AgentCapability(BaseModel):
    """Defines a specific capability that an agent can perform."""
    capability_id: str = Field(default_factory=lambda: str(uuid4()))
    capability_type: CapabilityType
    name: str
    description: str
    input_requirements: List[InputRequirement] = Field(default_factory=list)
    output_specifications: List[OutputSpecification] = Field(default_factory=list)
    resource_requirements: ResourceRequirements = Field(default_factory=ResourceRequirements)
    quality_guarantees: List[QualityGuarantee] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)  # Other capability IDs
    version: str = "1.0.0"
    
    class Config:
        use_enum_values = True


class AgentConfiguration(BaseModel):
    """Configuration for an individual agent."""
    agent_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    role: AgentRole
    capabilities: List[AgentCapability] = Field(default_factory=list)
    specialization: Optional[str] = None
    max_concurrent_tasks: int = 1
    priority_level: int = 1  # 1 = highest, 10 = lowest
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class TaskAssignment(BaseModel):
    """Assignment of a task to a specific agent."""
    assignment_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    agent_id: str
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    priority: int = 1
    deadline: Optional[datetime] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    status: ExecutionStatus = ExecutionStatus.PENDING
    
    class Config:
        use_enum_values = True


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
    
    class Config:
        use_enum_values = True


class ExecutionMetrics(BaseModel):
    """Metrics collected during execution."""
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    total_execution_time_seconds: float = 0.0
    average_step_time_seconds: float = 0.0
    resource_utilization: Dict[str, float] = Field(default_factory=dict)
    quality_scores: Dict[str, float] = Field(default_factory=dict)


class ReplanningEvent(BaseModel):
    """Event that triggered replanning."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    trigger_type: str
    trigger_description: str
    occurred_at: datetime = Field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = Field(default_factory=dict)
    replanning_successful: bool = False
    new_plan_id: Optional[str] = None


class EscalationPoint(BaseModel):
    """Point where execution was escalated to human oversight."""
    escalation_id: str = Field(default_factory=lambda: str(uuid4()))
    reason: str
    escalated_at: datetime = Field(default_factory=datetime.utcnow)
    escalated_by: str  # agent_id
    context: Dict[str, Any] = Field(default_factory=dict)
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


class ExecutionState(BaseModel):
    """Current state of multi-agent execution."""
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    current_plan_id: Optional[str] = None
    completed_steps: List[ExecutionStep] = Field(default_factory=list)
    intermediate_results: Dict[str, Any] = Field(default_factory=dict)
    execution_metrics: ExecutionMetrics = Field(default_factory=ExecutionMetrics)
    replanning_history: List[ReplanningEvent] = Field(default_factory=list)
    escalation_points: List[EscalationPoint] = Field(default_factory=list)
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class BusinessRule(BaseModel):
    """Individual business rule definition."""
    rule_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    rule_type: str
    condition: str  # Rule condition expression
    action: str  # Action to take when rule applies
    priority: int = 1
    enabled: bool = True
    domain: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SchemaConstraint(BaseModel):
    """Schema constraint definition."""
    constraint_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    schema_path: str  # JSONPath or similar
    constraint_type: str  # "required", "type", "range", etc.
    constraint_value: Any
    error_message: str
    severity: str = "error"  # "error", "warning", "info"


class RegulatoryRequirement(BaseModel):
    """Regulatory requirement definition."""
    requirement_id: str = Field(default_factory=lambda: str(uuid4()))
    regulation_name: str
    requirement_text: str
    compliance_check: str  # How to verify compliance
    applicable_domains: List[str] = Field(default_factory=list)
    severity: str = "critical"  # "critical", "high", "medium", "low"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Policy(BaseModel):
    """Organizational policy definition."""
    policy_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    policy_text: str
    enforcement_level: str = "mandatory"  # "mandatory", "recommended", "optional"
    applicable_roles: List[str] = Field(default_factory=list)
    exceptions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DomainContext(BaseModel):
    """Context for domain-specific execution."""
    domain_name: str
    applicable_rules: List[BusinessRule] = Field(default_factory=list)
    schema_constraints: List[SchemaConstraint] = Field(default_factory=list)
    regulatory_requirements: List[RegulatoryRequirement] = Field(default_factory=list)
    organizational_policies: List[Policy] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ApprovalRequirement(BaseModel):
    """Requirement for human approval."""
    requirement_id: str = Field(default_factory=lambda: str(uuid4()))
    operation_type: str
    risk_level: str = "medium"  # "low", "medium", "high", "critical"
    required_approvers: List[str] = Field(default_factory=list)
    timeout_minutes: int = 60
    escalation_policy: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MonitoringConfig(BaseModel):
    """Configuration for monitoring and observability."""
    enable_telemetry: bool = True
    enable_structured_logging: bool = True
    enable_performance_metrics: bool = True
    enable_health_checks: bool = True
    metrics_collection_interval_seconds: int = 30
    log_level: str = "INFO"
    correlation_id_header: str = "X-Correlation-ID"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultiAgentWorkflow(BaseModel):
    """Complete multi-agent workflow definition."""
    workflow_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    version: str = "1.0.0"
    agents: List[AgentConfiguration] = Field(default_factory=list)
    domain_context: Optional[DomainContext] = None
    approval_requirements: List[ApprovalRequirement] = Field(default_factory=list)
    monitoring_config: MonitoringConfig = Field(default_factory=MonitoringConfig)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionSession(BaseModel):
    """Active execution session for a multi-agent workflow."""
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_id: str
    execution_state: ExecutionState = Field(default_factory=ExecutionState)
    task_assignments: List[TaskAssignment] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('execution_state', pre=True, always=True)
    def sync_session_id(cls, v, values):
        """Ensure execution state session_id matches session_id."""
        if isinstance(v, ExecutionState):
            v.session_id = values.get('session_id', v.session_id)
        return v