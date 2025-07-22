"""
Tests for approval system Pydantic models.
"""

import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from src.dataqa.orchestration.approval.models import (
    AlternativeAction,
    ApprovalPolicy,
    ApprovalQueue,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    EscalationRule,
    FeedbackType,
    HumanFeedback,
    OperationType,
    RiskAssessment,
    RiskLevel,
    TimeoutEvent,
    TimeoutPolicy,
    TimeoutResolution,
)


class TestRiskAssessment:
    """Test cases for RiskAssessment model."""
    
    def test_risk_assessment_creation(self):
        """Test creating a valid risk assessment."""
        risk = RiskAssessment(
            risk_level=RiskLevel.HIGH,
            risk_factors=["Sensitive data", "Financial impact"],
            impact_description="Potential data breach",
            likelihood_score=0.7,
            severity_score=0.9,
            mitigation_strategies=["Human approval", "Audit trail"],
            compliance_implications=["GDPR", "SOX"],
        )
        
        assert risk.risk_level == RiskLevel.HIGH
        assert len(risk.risk_factors) == 2
        assert risk.likelihood_score == 0.7
        assert risk.severity_score == 0.9
        assert risk.risk_score == 0.63  # 0.7 * 0.9
    
    def test_risk_assessment_invalid_scores(self):
        """Test risk assessment with invalid scores."""
        with pytest.raises(ValidationError):
            RiskAssessment(
                risk_level=RiskLevel.HIGH,
                impact_description="Test",
                likelihood_score=1.5,  # Invalid: > 1.0
                severity_score=0.5,
            )
        
        with pytest.raises(ValidationError):
            RiskAssessment(
                risk_level=RiskLevel.HIGH,
                impact_description="Test",
                likelihood_score=0.5,
                severity_score=-0.1,  # Invalid: < 0.0
            )
    
    def test_risk_score_calculation(self):
        """Test risk score calculation property."""
        risk = RiskAssessment(
            risk_level=RiskLevel.MEDIUM,
            impact_description="Test impact",
            likelihood_score=0.6,
            severity_score=0.8,
        )
        
        assert risk.risk_score == 0.48  # 0.6 * 0.8


class TestTimeoutPolicy:
    """Test cases for TimeoutPolicy model."""
    
    def test_timeout_policy_defaults(self):
        """Test timeout policy with default values."""
        policy = TimeoutPolicy()
        
        assert policy.timeout_minutes == 60
        assert policy.escalation_enabled is True
        assert policy.escalation_delay_minutes == 30
        assert policy.auto_reject_on_timeout is False
        assert policy.fallback_action is None
        assert policy.notification_intervals == [15, 30, 45]
    
    def test_timeout_policy_custom_values(self):
        """Test timeout policy with custom values."""
        policy = TimeoutPolicy(
            timeout_minutes=120,
            escalation_enabled=False,
            auto_reject_on_timeout=True,
            fallback_action="default_approve",
            notification_intervals=[30, 60, 90],
        )
        
        assert policy.timeout_minutes == 120
        assert policy.escalation_enabled is False
        assert policy.auto_reject_on_timeout is True
        assert policy.fallback_action == "default_approve"
        assert policy.notification_intervals == [30, 60, 90]


class TestApprovalRequest:
    """Test cases for ApprovalRequest model."""
    
    @pytest.fixture
    def sample_risk_assessment(self):
        """Create a sample risk assessment."""
        return RiskAssessment(
            risk_level=RiskLevel.HIGH,
            risk_factors=["Sensitive data"],
            impact_description="Data modification",
            likelihood_score=0.7,
            severity_score=0.8,
        )
    
    def test_approval_request_creation(self, sample_risk_assessment):
        """Test creating a valid approval request."""
        request = ApprovalRequest(
            operation_type=OperationType.DATA_MODIFICATION,
            operation_description="Update customer records",
            context_explanation="Updating customer contact information",
            risk_assessment=sample_risk_assessment,
            requested_by="agent_123",
            session_id="session_456",
            workflow_id="workflow_789",
        )
        
        assert request.operation_type == OperationType.DATA_MODIFICATION
        assert request.operation_description == "Update customer records"
        assert request.risk_assessment == sample_risk_assessment
        assert request.requested_by == "agent_123"
        assert request.session_id == "session_456"
        assert request.workflow_id == "workflow_789"
        assert isinstance(request.requested_at, datetime)
        assert request.minimum_approvals == 1
        assert len(request.required_approvers) == 0  # Default empty
    
    def test_approval_request_with_alternatives(self, sample_risk_assessment):
        """Test approval request with alternative actions."""
        alternative = AlternativeAction(
            description="Read-only access instead",
            risk_level=RiskLevel.LOW,
            trade_offs=["Limited functionality", "Safer operation"],
        )
        
        request = ApprovalRequest(
            operation_type=OperationType.SENSITIVE_DATA_ACCESS,
            operation_description="Access sensitive data",
            context_explanation="Need to analyze customer data",
            risk_assessment=sample_risk_assessment,
            alternative_options=[alternative],
            requested_by="agent_123",
            session_id="session_456",
            workflow_id="workflow_789",
        )
        
        assert len(request.alternative_options) == 1
        assert request.alternative_options[0].description == "Read-only access instead"
        assert request.alternative_options[0].risk_level == RiskLevel.LOW
    
    def test_approval_request_with_custom_timeout(self, sample_risk_assessment):
        """Test approval request with custom timeout policy."""
        timeout_policy = TimeoutPolicy(
            timeout_minutes=30,
            auto_reject_on_timeout=True,
        )
        
        request = ApprovalRequest(
            operation_type=OperationType.SCHEMA_CHANGE,
            operation_description="Modify database schema",
            context_explanation="Adding new column for feature",
            risk_assessment=sample_risk_assessment,
            timeout_policy=timeout_policy,
            requested_by="agent_123",
            session_id="session_456",
            workflow_id="workflow_789",
        )
        
        assert request.timeout_policy.timeout_minutes == 30
        assert request.timeout_policy.auto_reject_on_timeout is True


class TestApprovalResponse:
    """Test cases for ApprovalResponse model."""
    
    def test_approval_response_approved(self):
        """Test creating an approved response."""
        response = ApprovalResponse(
            request_id="request_123",
            status=ApprovalStatus.APPROVED,
            approved_by="user_456",
            comments="Looks good, approved for data quality improvement",
            conditions=["Must complete within 24 hours", "Notify on completion"],
        )
        
        assert response.request_id == "request_123"
        assert response.status == ApprovalStatus.APPROVED
        assert response.approved_by == "user_456"
        assert response.comments == "Looks good, approved for data quality improvement"
        assert len(response.conditions) == 2
        assert isinstance(response.responded_at, datetime)
    
    def test_approval_response_rejected(self):
        """Test creating a rejected response."""
        response = ApprovalResponse(
            request_id="request_123",
            status=ApprovalStatus.REJECTED,
            approved_by="user_456",
            comments="Insufficient business justification",
            suggested_modifications=[
                "Provide detailed business case",
                "Reduce scope to critical data only",
            ],
        )
        
        assert response.status == ApprovalStatus.REJECTED
        assert len(response.suggested_modifications) == 2
        assert "business case" in response.suggested_modifications[0]
    
    def test_approval_response_with_escalation(self):
        """Test creating a response with escalation."""
        response = ApprovalResponse(
            request_id="request_123",
            status=ApprovalStatus.PENDING,
            escalated_to="senior_manager",
            escalation_reason="High risk operation requires senior approval",
        )
        
        assert response.status == ApprovalStatus.PENDING
        assert response.escalated_to == "senior_manager"
        assert response.escalation_reason == "High risk operation requires senior approval"


class TestHumanFeedback:
    """Test cases for HumanFeedback model."""
    
    def test_human_feedback_creation(self):
        """Test creating human feedback."""
        feedback = HumanFeedback(
            feedback_type=FeedbackType.APPROVAL,
            request_id="request_123",
            feedback_text="This was a good decision, approved quickly",
            rating=4,
            improvement_suggestions=["Could provide more context", "Faster notification"],
            context_tags=["data_modification", "high_risk", "financial"],
            provided_by="user_456",
            session_id="session_789",
        )
        
        assert feedback.feedback_type == FeedbackType.APPROVAL
        assert feedback.request_id == "request_123"
        assert feedback.rating == 4
        assert len(feedback.improvement_suggestions) == 2
        assert len(feedback.context_tags) == 3
        assert feedback.learning_priority == "medium"  # Default
        assert isinstance(feedback.provided_at, datetime)
    
    def test_human_feedback_invalid_rating(self):
        """Test human feedback with invalid rating."""
        with pytest.raises(ValidationError):
            HumanFeedback(
                feedback_type=FeedbackType.APPROVAL,
                request_id="request_123",
                feedback_text="Test feedback",
                rating=6,  # Invalid: > 5
                provided_by="user_456",
                session_id="session_789",
            )
        
        with pytest.raises(ValidationError):
            HumanFeedback(
                feedback_type=FeedbackType.APPROVAL,
                request_id="request_123",
                feedback_text="Test feedback",
                rating=0,  # Invalid: < 1
                provided_by="user_456",
                session_id="session_789",
            )
    
    def test_human_feedback_learning_type(self):
        """Test human feedback for learning."""
        feedback = HumanFeedback(
            feedback_type=FeedbackType.LEARNING,
            request_id="request_123",
            feedback_text="System should learn that financial operations need extra scrutiny",
            context_tags=["financial", "learning_opportunity"],
            similar_scenarios=["previous_financial_op_1", "previous_financial_op_2"],
            learning_priority="high",
            provided_by="user_456",
            session_id="session_789",
        )
        
        assert feedback.feedback_type == FeedbackType.LEARNING
        assert feedback.learning_priority == "high"
        assert len(feedback.similar_scenarios) == 2


class TestApprovalPolicy:
    """Test cases for ApprovalPolicy model."""
    
    def test_approval_policy_creation(self):
        """Test creating an approval policy."""
        timeout_policy = TimeoutPolicy(timeout_minutes=45)
        
        policy = ApprovalPolicy(
            name="Financial Data Policy",
            description="Requires approval for financial data operations",
            operation_types=[OperationType.DATA_MODIFICATION, OperationType.FINANCIAL_CALCULATION],
            risk_threshold=RiskLevel.MEDIUM,
            data_sensitivity_levels=["sensitive", "confidential"],
            required_roles=["financial_approver", "data_steward"],
            minimum_approvals=2,
            timeout_policy=timeout_policy,
            applicable_domains=["finance", "accounting"],
        )
        
        assert policy.name == "Financial Data Policy"
        assert len(policy.operation_types) == 2
        assert policy.risk_threshold == RiskLevel.MEDIUM
        assert len(policy.data_sensitivity_levels) == 2
        assert policy.minimum_approvals == 2
        assert policy.timeout_policy.timeout_minutes == 45
        assert policy.enabled is True  # Default
        assert policy.priority == 1  # Default
    
    def test_approval_policy_with_resource_patterns(self):
        """Test approval policy with resource patterns."""
        policy = ApprovalPolicy(
            name="Sensitive Table Policy",
            description="Requires approval for sensitive table access",
            resource_patterns=["*_sensitive", "*_pii", "customer_*"],
            risk_threshold=RiskLevel.LOW,
        )
        
        assert len(policy.resource_patterns) == 3
        assert "*_sensitive" in policy.resource_patterns


class TestEscalationRule:
    """Test cases for EscalationRule model."""
    
    def test_escalation_rule_creation(self):
        """Test creating an escalation rule."""
        rule = EscalationRule(
            name="High Risk Escalation",
            description="Escalate high risk operations to senior management",
            trigger_conditions=["risk_level >= HIGH", "financial_impact > 10000"],
            timeout_minutes=30,
            risk_threshold=RiskLevel.HIGH,
            escalate_to_roles=["senior_manager", "risk_officer"],
            escalate_to_users=["john.doe", "jane.smith"],
            notification_channels=["email", "slack", "sms"],
        )
        
        assert rule.name == "High Risk Escalation"
        assert len(rule.trigger_conditions) == 2
        assert rule.timeout_minutes == 30
        assert rule.risk_threshold == RiskLevel.HIGH
        assert len(rule.escalate_to_roles) == 2
        assert len(rule.escalate_to_users) == 2
        assert len(rule.notification_channels) == 3
        assert rule.enabled is True  # Default
        assert rule.priority == 1  # Default


class TestTimeoutEvent:
    """Test cases for TimeoutEvent model."""
    
    def test_timeout_event_creation(self):
        """Test creating a timeout event."""
        event = TimeoutEvent(
            request_id="request_123",
            timeout_duration_minutes=60,
            escalation_triggered=True,
            fallback_action_taken="auto_approve",
            pending_approvers=["user1", "user2"],
            notifications_sent=3,
        )
        
        assert event.request_id == "request_123"
        assert event.timeout_duration_minutes == 60
        assert event.escalation_triggered is True
        assert event.fallback_action_taken == "auto_approve"
        assert event.auto_rejected is False  # Default
        assert len(event.pending_approvers) == 2
        assert event.notifications_sent == 3
        assert isinstance(event.timeout_occurred_at, datetime)


class TestTimeoutResolution:
    """Test cases for TimeoutResolution model."""
    
    def test_timeout_resolution_creation(self):
        """Test creating a timeout resolution."""
        resolution = TimeoutResolution(
            timeout_event_id="event_123",
            resolution_type="escalated",
            resolution_description="Request was escalated to senior management",
            resolved_by="system",
        )
        
        assert resolution.timeout_event_id == "event_123"
        assert resolution.resolution_type == "escalated"
        assert resolution.resolution_description == "Request was escalated to senior management"
        assert resolution.resolved_by == "system"
        assert isinstance(resolution.resolved_at, datetime)


class TestApprovalQueue:
    """Test cases for ApprovalQueue model."""
    
    def test_approval_queue_creation(self):
        """Test creating an approval queue."""
        queue = ApprovalQueue(
            name="Data Steward Queue",
            description="Queue for data steward approvals",
            assigned_roles=["data_steward", "senior_data_steward"],
            assigned_users=["alice", "bob"],
            priority_rules=["high_risk_first", "oldest_first"],
        )
        
        assert queue.name == "Data Steward Queue"
        assert queue.description == "Queue for data steward approvals"
        assert len(queue.assigned_roles) == 2
        assert len(queue.assigned_users) == 2
        assert len(queue.priority_rules) == 2
        assert len(queue.pending_requests) == 0  # Default empty
        assert len(queue.active_requests) == 0  # Default empty
        assert queue.total_processed == 0  # Default
        assert queue.average_response_time_minutes == 0.0  # Default
        assert queue.approval_rate == 0.0  # Default
        assert isinstance(queue.created_at, datetime)
        assert isinstance(queue.updated_at, datetime)
    
    def test_approval_queue_with_requests(self):
        """Test approval queue with requests."""
        queue = ApprovalQueue(
            name="Test Queue",
            pending_requests=["req1", "req2", "req3"],
            active_requests=["req4", "req5"],
            total_processed=10,
            average_response_time_minutes=45.5,
            approval_rate=0.8,
        )
        
        assert len(queue.pending_requests) == 3
        assert len(queue.active_requests) == 2
        assert queue.total_processed == 10
        assert queue.average_response_time_minutes == 45.5
        assert queue.approval_rate == 0.8


class TestAlternativeAction:
    """Test cases for AlternativeAction model."""
    
    def test_alternative_action_creation(self):
        """Test creating an alternative action."""
        action = AlternativeAction(
            description="Use read-only access instead of full access",
            risk_level=RiskLevel.LOW,
            trade_offs=["Limited functionality", "Reduced risk", "Faster approval"],
            implementation_complexity="low",
        )
        
        assert action.description == "Use read-only access instead of full access"
        assert action.risk_level == RiskLevel.LOW
        assert len(action.trade_offs) == 3
        assert action.implementation_complexity == "low"
        assert action.action_id is not None  # Auto-generated UUID