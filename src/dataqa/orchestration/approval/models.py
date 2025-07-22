"""
Pydantic models for human-in-the-loop approval system.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class OperationType(str, Enum):
    """Types of operations that may require approval."""
    DATA_MODIFICATION = "data_modification"
    SCHEMA_CHANGE = "schema_change"
    EXTERNAL_API_CALL = "external_api_call"
    SENSITIVE_DATA_ACCESS = "sensitive_data_access"
    FINANCIAL_CALCULATION = "financial_calculation"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    SYSTEM_CONFIGURATION = "system_configuration"
    USER_DATA_EXPORT = "user_data_export"


class RiskLevel(str, Enum):
    """Risk levels for operations requiring approval."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(str, Enum):
    """Status of approval requests."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class FeedbackType(str, Enum):
    """Types of human feedback."""
    APPROVAL = "approval"
    REJECTION = "rejection"
    MODIFICATION = "modification"
    ESCALATION = "escalation"
    LEARNING = "learning"


class TimeoutPolicy(BaseModel):
    """Policy for handling approval timeouts."""
    timeout_minutes: int = 60
    escalation_enabled: bool = True
    escalation_delay_minutes: int = 30
    auto_reject_on_timeout: bool = False
    fallback_action: Optional[str] = None
    notification_intervals: List[int] = Field(default_factory=lambda: [15, 30, 45])  # minutes


class RiskAssessment(BaseModel):
    """Risk assessment for operations requiring approval."""
    risk_level: RiskLevel
    risk_factors: List[str] = Field(default_factory=list)
    impact_description: str
    likelihood_score: float = Field(ge=0.0, le=1.0)  # 0.0 = unlikely, 1.0 = certain
    severity_score: float = Field(ge=0.0, le=1.0)  # 0.0 = minimal, 1.0 = severe
    mitigation_strategies: List[str] = Field(default_factory=list)
    compliance_implications: List[str] = Field(default_factory=list)
    
    @property
    def risk_score(self) -> float:
        """Calculate overall risk score."""
        return (self.likelihood_score * self.severity_score)


class AlternativeAction(BaseModel):
    """Alternative action that could be taken instead."""
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    risk_level: RiskLevel
    trade_offs: List[str] = Field(default_factory=list)
    implementation_complexity: str = "medium"  # "low", "medium", "high"


class ApprovalRequest(BaseModel):
    """Request for human approval of a sensitive operation."""
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    operation_type: OperationType
    operation_description: str
    context_explanation: str
    risk_assessment: RiskAssessment
    alternative_options: List[AlternativeAction] = Field(default_factory=list)
    timeout_policy: TimeoutPolicy = Field(default_factory=TimeoutPolicy)
    
    # Request metadata
    requested_by: str  # agent_id or user_id
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    session_id: str
    workflow_id: str
    step_id: Optional[str] = None
    
    # Approval requirements
    required_approvers: List[str] = Field(default_factory=list)
    minimum_approvals: int = 1
    approval_roles: List[str] = Field(default_factory=list)
    
    # Additional context
    affected_resources: List[str] = Field(default_factory=list)
    data_sensitivity_level: Optional[str] = None
    regulatory_context: List[str] = Field(default_factory=list)
    business_justification: Optional[str] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class ApprovalResponse(BaseModel):
    """Response to an approval request."""
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str
    status: ApprovalStatus
    
    # Response details
    approved_by: Optional[str] = None  # user_id or role
    responded_at: datetime = Field(default_factory=datetime.utcnow)
    comments: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)  # Conditions for approval
    
    # Modifications or alternatives
    suggested_modifications: List[str] = Field(default_factory=list)
    approved_alternative: Optional[str] = None  # alternative_action_id
    
    # Escalation info
    escalated_to: Optional[str] = None
    escalation_reason: Optional[str] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class HumanFeedback(BaseModel):
    """Feedback from humans for continuous learning."""
    feedback_id: str = Field(default_factory=lambda: str(uuid4()))
    feedback_type: FeedbackType
    request_id: str
    response_id: Optional[str] = None
    
    # Feedback content
    feedback_text: str
    rating: Optional[int] = Field(None, ge=1, le=5)  # 1-5 star rating
    improvement_suggestions: List[str] = Field(default_factory=list)
    
    # Learning context
    context_tags: List[str] = Field(default_factory=list)
    similar_scenarios: List[str] = Field(default_factory=list)
    learning_priority: str = "medium"  # "low", "medium", "high"
    
    # Feedback metadata
    provided_by: str  # user_id
    provided_at: datetime = Field(default_factory=datetime.utcnow)
    session_id: str
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class ApprovalPolicy(BaseModel):
    """Policy defining when approvals are required."""
    policy_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    
    # Trigger conditions
    operation_types: List[OperationType] = Field(default_factory=list)
    risk_threshold: RiskLevel = RiskLevel.MEDIUM
    data_sensitivity_levels: List[str] = Field(default_factory=list)
    resource_patterns: List[str] = Field(default_factory=list)  # Regex patterns
    
    # Approval requirements
    required_roles: List[str] = Field(default_factory=list)
    minimum_approvals: int = 1
    timeout_policy: TimeoutPolicy = Field(default_factory=TimeoutPolicy)
    
    # Policy metadata
    enabled: bool = True
    priority: int = 1  # Higher number = higher priority
    applicable_domains: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class EscalationRule(BaseModel):
    """Rule for escalating approval requests."""
    rule_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    
    # Escalation triggers
    trigger_conditions: List[str] = Field(default_factory=list)
    timeout_minutes: int = 60
    risk_threshold: RiskLevel = RiskLevel.HIGH
    
    # Escalation actions
    escalate_to_roles: List[str] = Field(default_factory=list)
    escalate_to_users: List[str] = Field(default_factory=list)
    notification_channels: List[str] = Field(default_factory=list)
    
    # Rule metadata
    enabled: bool = True
    priority: int = 1
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class TimeoutEvent(BaseModel):
    """Event representing an approval timeout."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str
    timeout_occurred_at: datetime = Field(default_factory=datetime.utcnow)
    timeout_duration_minutes: int
    
    # Timeout handling
    escalation_triggered: bool = False
    fallback_action_taken: Optional[str] = None
    auto_rejected: bool = False
    
    # Context
    pending_approvers: List[str] = Field(default_factory=list)
    notifications_sent: int = 0
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TimeoutResolution(BaseModel):
    """Resolution for a timeout event."""
    resolution_id: str = Field(default_factory=lambda: str(uuid4()))
    timeout_event_id: str
    resolution_type: str  # "escalated", "auto_rejected", "fallback", "manual"
    resolution_description: str
    resolved_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_by: Optional[str] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ApprovalQueue(BaseModel):
    """Queue of pending approval requests."""
    queue_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    
    # Queue configuration
    assigned_roles: List[str] = Field(default_factory=list)
    assigned_users: List[str] = Field(default_factory=list)
    priority_rules: List[str] = Field(default_factory=list)
    
    # Queue state
    pending_requests: List[str] = Field(default_factory=list)  # request_ids
    active_requests: List[str] = Field(default_factory=list)  # request_ids being processed
    
    # Queue metrics
    total_processed: int = 0
    average_response_time_minutes: float = 0.0
    approval_rate: float = 0.0
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)