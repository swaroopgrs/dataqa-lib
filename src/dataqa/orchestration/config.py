"""
Configuration schemas for multi-agent workflows and domain contexts.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator

from .models import (
    AgentCapability,
    AgentConfiguration,
    AgentRole,
    ApprovalRequirement,
    BusinessRule,
    CapabilityType,
    DomainContext,
    MonitoringConfig,
    Policy,
    RegulatoryRequirement,
    SchemaConstraint,
)


class WorkflowTrigger(str, Enum):
    """Types of workflow triggers."""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    API_REQUEST = "api_request"


class WorkflowStatus(str, Enum):
    """Workflow status values."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class CapabilityConfigSchema(BaseModel):
    """Configuration schema for agent capabilities."""
    capability_type: CapabilityType
    name: str
    description: str
    version: str = "1.0.0"
    
    # Input/Output specifications
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    
    # Resource requirements
    cpu_cores: Optional[int] = None
    memory_mb: Optional[int] = None
    gpu_required: bool = False
    network_access: bool = True
    storage_mb: Optional[int] = None
    execution_timeout_seconds: int = 300
    
    # Quality guarantees
    accuracy_threshold: Optional[float] = None
    response_time_ms: Optional[int] = None
    availability_percentage: Optional[float] = None
    error_rate_threshold: Optional[float] = None
    
    # Dependencies and metadata
    dependencies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class AgentConfigSchema(BaseModel):
    """Configuration schema for individual agents."""
    name: str
    role: AgentRole
    agent_type: str = "worker"  # "manager", "worker", "specialist"
    specialization: Optional[str] = None
    
    # Capabilities
    capabilities: List[CapabilityConfigSchema] = Field(default_factory=list)
    
    # Operational settings
    max_concurrent_tasks: int = 1
    priority_level: int = 1
    enabled: bool = True
    
    # Hierarchy settings
    parent_agent: Optional[str] = None  # Name or ID of parent agent
    subordinates: List[str] = Field(default_factory=list)  # Names or IDs of subordinate agents
    
    # Configuration overrides
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
    
    @validator('agent_type')
    def validate_role_consistency(cls, v, values):
        """Ensure role is consistent with agent_type."""
        role = values.get('role')
        
        if v == 'manager' and role != AgentRole.MANAGER:
            raise ValueError("Manager agent_type must have MANAGER role")
        
        if v in ['worker', 'specialist'] and role == AgentRole.MANAGER:
            raise ValueError("Worker/specialist agent_type cannot have MANAGER role")
        
        return v


class BusinessRuleConfigSchema(BaseModel):
    """Configuration schema for business rules."""
    name: str
    description: str
    rule_type: str = "validation"  # "validation", "transformation", "routing", "approval"
    
    # Rule definition
    condition: str  # Rule condition expression (e.g., JSONPath, Python expression)
    action: str  # Action to take when rule applies
    
    # Rule properties
    priority: int = 1
    enabled: bool = True
    domain: Optional[str] = None
    
    # Error handling
    error_message: Optional[str] = None
    severity: str = "error"  # "error", "warning", "info"
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DomainConfigSchema(BaseModel):
    """Configuration schema for domain contexts."""
    domain_name: str
    description: str
    version: str = "1.0.0"
    
    # Business rules
    business_rules: List[BusinessRuleConfigSchema] = Field(default_factory=list)
    
    # Schema constraints
    schema_constraints: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Regulatory requirements
    regulatory_requirements: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Organizational policies
    organizational_policies: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Domain-specific settings
    default_data_sources: List[str] = Field(default_factory=list)
    required_approvals: List[str] = Field(default_factory=list)
    compliance_frameworks: List[str] = Field(default_factory=list)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ApprovalConfigSchema(BaseModel):
    """Configuration schema for approval requirements."""
    operation_type: str
    description: str
    
    # Risk assessment
    risk_level: str = "medium"  # "low", "medium", "high", "critical"
    risk_factors: List[str] = Field(default_factory=list)
    
    # Approval settings
    required_approvers: List[str] = Field(default_factory=list)  # Roles or specific users
    approval_threshold: int = 1  # Number of approvals required
    timeout_minutes: int = 60
    
    # Escalation
    escalation_policy: Optional[str] = None
    escalation_timeout_minutes: int = 120
    
    # Conditions
    conditions: List[str] = Field(default_factory=list)  # When this approval is required
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MonitoringConfigSchema(BaseModel):
    """Configuration schema for monitoring and observability."""
    # Telemetry settings
    enable_telemetry: bool = True
    telemetry_endpoint: Optional[str] = None
    telemetry_interval_seconds: int = 30
    
    # Logging settings
    enable_structured_logging: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Metrics settings
    enable_performance_metrics: bool = True
    metrics_endpoint: Optional[str] = None
    custom_metrics: List[str] = Field(default_factory=list)
    
    # Health checks
    enable_health_checks: bool = True
    health_check_interval_seconds: int = 60
    health_check_timeout_seconds: int = 10
    
    # Alerting
    enable_alerting: bool = False
    alert_thresholds: Dict[str, float] = Field(default_factory=dict)
    alert_endpoints: List[str] = Field(default_factory=list)
    
    # Correlation and tracing
    correlation_id_header: str = "X-Correlation-ID"
    enable_distributed_tracing: bool = False
    tracing_endpoint: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowConfigSchema(BaseModel):
    """Configuration schema for complete multi-agent workflows."""
    # Basic information
    name: str
    description: str
    version: str = "1.0.0"
    
    # Workflow properties
    workflow_type: str = "data_analysis"  # "data_analysis", "report_generation", "compliance_check", etc.
    trigger: WorkflowTrigger = WorkflowTrigger.MANUAL
    status: WorkflowStatus = WorkflowStatus.DRAFT
    
    # Agent configuration
    agents: List[AgentConfigSchema] = Field(default_factory=list)
    
    # Domain and business context
    domain_context: Optional[DomainConfigSchema] = None
    
    # Approval and compliance
    approval_requirements: List[ApprovalConfigSchema] = Field(default_factory=list)
    
    # Monitoring and observability
    monitoring_config: MonitoringConfigSchema = Field(default_factory=MonitoringConfigSchema)
    
    # Execution settings
    max_execution_time_minutes: int = 60
    max_retries: int = 3
    enable_checkpointing: bool = True
    enable_rollback: bool = True
    
    # Resource limits
    max_concurrent_agents: int = 10
    resource_limits: Dict[str, Any] = Field(default_factory=dict)
    
    # Scheduling (for scheduled workflows)
    schedule_expression: Optional[str] = None  # Cron expression
    timezone: str = "UTC"
    
    # Event handling (for event-driven workflows)
    event_sources: List[str] = Field(default_factory=list)
    event_filters: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata and tags
    tags: List[str] = Field(default_factory=list)
    owner: Optional[str] = None
    team: Optional[str] = None
    environment: str = "development"  # "development", "staging", "production"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True
    
    @validator('agents')
    def validate_agent_hierarchy(cls, v):
        """Validate agent hierarchy consistency."""
        agent_names = {agent.name for agent in v}
        
        for agent in v:
            # Check parent references
            if agent.parent_agent and agent.parent_agent not in agent_names:
                raise ValueError(f"Agent {agent.name} references non-existent parent {agent.parent_agent}")
            
            # Check subordinate references
            for subordinate in agent.subordinates:
                if subordinate not in agent_names:
                    raise ValueError(f"Agent {agent.name} references non-existent subordinate {subordinate}")
        
        return v
    
    @validator('schedule_expression')
    def validate_schedule_expression(cls, v, values):
        """Validate cron expression for scheduled workflows."""
        if v and values.get('trigger') != WorkflowTrigger.SCHEDULED:
            raise ValueError("Schedule expression only valid for scheduled workflows")
        
        if values.get('trigger') == WorkflowTrigger.SCHEDULED and not v:
            raise ValueError("Schedule expression required for scheduled workflows")
        
        # Basic cron validation (simplified)
        if v:
            parts = v.split()
            if len(parts) not in [5, 6]:  # Standard cron (5) or with seconds (6)
                raise ValueError("Invalid cron expression format")
        
        return v


class DeploymentConfigSchema(BaseModel):
    """Configuration schema for workflow deployment."""
    # Deployment target
    target_environment: str = "local"  # "local", "docker", "kubernetes", "cloud"
    
    # Resource allocation
    cpu_limit: Optional[str] = None  # e.g., "2", "500m"
    memory_limit: Optional[str] = None  # e.g., "1Gi", "512Mi"
    storage_limit: Optional[str] = None
    
    # Scaling settings
    min_replicas: int = 1
    max_replicas: int = 1
    auto_scaling_enabled: bool = False
    scaling_metrics: List[str] = Field(default_factory=list)
    
    # Network settings
    expose_ports: List[int] = Field(default_factory=list)
    ingress_enabled: bool = False
    ingress_host: Optional[str] = None
    
    # Security settings
    security_context: Dict[str, Any] = Field(default_factory=dict)
    secrets: List[str] = Field(default_factory=list)
    config_maps: List[str] = Field(default_factory=list)
    
    # Health and readiness
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    startup_timeout_seconds: int = 300
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultiAgentWorkflowConfig(BaseModel):
    """Complete configuration for a multi-agent workflow system."""
    # Workflow definition
    workflow: WorkflowConfigSchema
    
    # Deployment configuration
    deployment: Optional[DeploymentConfigSchema] = None
    
    # Global settings
    global_settings: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuration metadata
    config_version: str = "1.0"
    config_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_runtime_config(self) -> Dict[str, Any]:
        """Convert configuration to runtime format."""
        # This would convert the Pydantic models to the runtime models
        # used by the actual orchestration system
        return {
            "workflow_id": self.config_id,
            "workflow": self.workflow.dict(),
            "deployment": self.deployment.dict() if self.deployment else None,
            "global_settings": self.global_settings,
            "config_metadata": {
                "version": self.config_version,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat()
            }
        }