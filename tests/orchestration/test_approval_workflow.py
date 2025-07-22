"""
Tests for ApprovalWorkflow class.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.dataqa.orchestration.approval.models import (
    AlternativeAction,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    EscalationRule,
    OperationType,
    RiskAssessment,
    RiskLevel,
    TimeoutPolicy,
)
from src.dataqa.orchestration.approval.workflow import ApprovalWorkflow, ApprovalWorkflowError


class TestApprovalWorkflow:
    """Test cases for ApprovalWorkflow."""
    
    @pytest.fixture
    def sample_policy(self):
        """Create a sample approval policy."""
        return ApprovalPolicy(
            name="Data Modification Policy",
            description="Requires approval for data modifications",
            operation_types=[OperationType.DATA_MODIFICATION],
            risk_threshold=RiskLevel.MEDIUM,
            required_roles=["data_steward"],
            minimum_approvals=1,
            timeout_policy=TimeoutPolicy(timeout_minutes=30),
        )
    
    @pytest.fixture
    def sample_escalation_rule(self):
        """Create a sample escalation rule."""
        return EscalationRule(
            name="High Risk Escalation",
            description="Escalate high risk operations",
            risk_threshold=RiskLevel.HIGH,
            escalate_to_roles=["senior_manager"],
            timeout_minutes=15,
        )
    
    @pytest.fixture
    def approval_workflow(self, sample_policy, sample_escalation_rule):
        """Create an ApprovalWorkflow instance."""
        return ApprovalWorkflow(
            policies=[sample_policy],
            escalation_rules=[sample_escalation_rule],
            default_timeout_minutes=60,
        )
    
    @pytest.fixture
    def sample_risk_assessment(self):
        """Create a sample risk assessment."""
        return RiskAssessment(
            risk_level=RiskLevel.HIGH,
            risk_factors=["Sensitive data", "Financial impact"],
            impact_description="Modification of customer financial data",
            likelihood_score=0.7,
            severity_score=0.9,
            mitigation_strategies=["Human approval", "Audit trail"],
            compliance_implications=["SOX compliance", "GDPR requirements"],
        )
    
    @pytest.mark.asyncio
    async def test_requires_approval_with_matching_policy(self, approval_workflow):
        """Test that approval is required when policy matches."""
        context = {
            "data_sensitivity_level": "sensitive",
            "affected_resources": ["customer_data"],
        }
        
        result = await approval_workflow.requires_approval(
            OperationType.DATA_MODIFICATION,
            context,
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_requires_approval_no_matching_policy(self, approval_workflow):
        """Test that approval is not required when no policy matches."""
        context = {}
        
        result = await approval_workflow.requires_approval(
            OperationType.EXTERNAL_API_CALL,  # Not in policy
            context,
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_create_approval_request(self, approval_workflow, sample_risk_assessment):
        """Test creating an approval request."""
        context = {
            "affected_resources": ["customer_table"],
            "business_justification": "Data quality improvement",
        }
        
        request = await approval_workflow.create_approval_request(
            operation_type=OperationType.DATA_MODIFICATION,
            operation_description="Update customer records",
            context=context,
            risk_assessment=sample_risk_assessment,
            session_id="session_123",
            workflow_id="workflow_456",
            requested_by="agent_789",
        )
        
        assert isinstance(request, ApprovalRequest)
        assert request.operation_type == OperationType.DATA_MODIFICATION
        assert request.operation_description == "Update customer records"
        assert request.risk_assessment == sample_risk_assessment
        assert request.session_id == "session_123"
        assert request.workflow_id == "workflow_456"
        assert request.requested_by == "agent_789"
        assert "customer_table" in request.affected_resources
        assert request.business_justification == "Data quality improvement"
        assert "data_steward" in request.required_approvers
    
    @pytest.mark.asyncio
    async def test_create_approval_request_generates_risk_assessment(self, approval_workflow):
        """Test that risk assessment is generated when not provided."""
        context = {"data_sensitivity_level": "confidential"}
        
        request = await approval_workflow.create_approval_request(
            operation_type=OperationType.SENSITIVE_DATA_ACCESS,
            operation_description="Access sensitive customer data",
            context=context,
            session_id="session_123",
            workflow_id="workflow_456",
            requested_by="agent_789",
        )
        
        assert request.risk_assessment is not None
        assert request.risk_assessment.risk_level == RiskLevel.HIGH
        assert "Sensitive data involved" in request.risk_assessment.risk_factors
    
    @pytest.mark.asyncio
    async def test_process_response_success(self, approval_workflow, sample_risk_assessment):
        """Test processing a successful approval response."""
        # Create a request first
        request = await approval_workflow.create_approval_request(
            operation_type=OperationType.DATA_MODIFICATION,
            operation_description="Test operation",
            context={},
            risk_assessment=sample_risk_assessment,
            session_id="session_123",
            workflow_id="workflow_456",
            requested_by="agent_789",
        )
        
        # Create response
        response = ApprovalResponse(
            request_id=request.request_id,
            status=ApprovalStatus.APPROVED,
            approved_by="user_123",
            comments="Approved for data quality improvement",
        )
        
        # Process response
        await approval_workflow.process_response(request.request_id, response)
        
        # Request should be removed from pending
        assert request.request_id not in approval_workflow._pending_requests
    
    @pytest.mark.asyncio
    async def test_process_response_nonexistent_request(self, approval_workflow):
        """Test processing response for non-existent request raises error."""
        response = ApprovalResponse(
            request_id="nonexistent_123",
            status=ApprovalStatus.APPROVED,
        )
        
        with pytest.raises(ApprovalWorkflowError, match="not found"):
            await approval_workflow.process_response("nonexistent_123", response)
    
    @pytest.mark.asyncio
    async def test_handle_timeout_with_escalation(self, approval_workflow, sample_risk_assessment):
        """Test handling timeout with escalation."""
        # Create a high-risk request that should trigger escalation
        request = await approval_workflow.create_approval_request(
            operation_type=OperationType.DATA_MODIFICATION,
            operation_description="High risk operation",
            context={},
            risk_assessment=sample_risk_assessment,  # HIGH risk level
            session_id="session_123",
            workflow_id="workflow_456",
            requested_by="agent_789",
        )
        
        # Handle timeout
        timeout_event = await approval_workflow.handle_timeout(request.request_id)
        
        assert timeout_event.request_id == request.request_id
        assert timeout_event.escalation_triggered is True
        assert timeout_event.timeout_duration_minutes == 30  # From policy
    
    @pytest.mark.asyncio
    async def test_handle_timeout_with_auto_rejection(self, approval_workflow):
        """Test handling timeout with auto-rejection."""
        # Create policy with auto-rejection
        policy = ApprovalPolicy(
            name="Auto Reject Policy",
            description="Policy that auto-rejects on timeout",
            operation_types=[OperationType.EXTERNAL_API_CALL],
            timeout_policy=TimeoutPolicy(
                timeout_minutes=15,
                auto_reject_on_timeout=True,
            ),
        )
        
        workflow = ApprovalWorkflow(policies=[policy])
        
        # Create request
        request = await workflow.create_approval_request(
            operation_type=OperationType.EXTERNAL_API_CALL,
            operation_description="API call",
            context={},
            session_id="session_123",
            workflow_id="workflow_456",
            requested_by="agent_789",
        )
        
        # Handle timeout
        timeout_event = await workflow.handle_timeout(request.request_id)
        
        assert timeout_event.auto_rejected is True
        assert request.request_id not in workflow._pending_requests
    
    @pytest.mark.asyncio
    async def test_get_pending_requests_no_filter(self, approval_workflow, sample_risk_assessment):
        """Test getting pending requests without filters."""
        # Create multiple requests
        request1 = await approval_workflow.create_approval_request(
            operation_type=OperationType.DATA_MODIFICATION,
            operation_description="Request 1",
            context={},
            risk_assessment=sample_risk_assessment,
            session_id="session_1",
            workflow_id="workflow_1",
            requested_by="agent_1",
        )
        
        low_risk = RiskAssessment(
            risk_level=RiskLevel.LOW,
            risk_factors=[],
            impact_description="Low impact operation",
            likelihood_score=0.2,
            severity_score=0.3,
        )
        
        request2 = await approval_workflow.create_approval_request(
            operation_type=OperationType.EXTERNAL_API_CALL,
            operation_description="Request 2",
            context={},
            risk_assessment=low_risk,
            session_id="session_2",
            workflow_id="workflow_2",
            requested_by="agent_2",
        )
        
        pending = await approval_workflow.get_pending_requests()
        
        assert len(pending) == 2
        # Should be sorted by risk level (highest first)
        assert pending[0].risk_assessment.risk_level == RiskLevel.HIGH
        assert pending[1].risk_assessment.risk_level == RiskLevel.LOW
    
    @pytest.mark.asyncio
    async def test_get_pending_requests_with_role_filter(self, approval_workflow, sample_risk_assessment):
        """Test getting pending requests filtered by approver role."""
        request = await approval_workflow.create_approval_request(
            operation_type=OperationType.DATA_MODIFICATION,
            operation_description="Request with role",
            context={},
            risk_assessment=sample_risk_assessment,
            session_id="session_123",
            workflow_id="workflow_456",
            requested_by="agent_789",
        )
        
        # Filter by matching role
        pending = await approval_workflow.get_pending_requests(approver_role="data_steward")
        assert len(pending) == 1
        assert pending[0].request_id == request.request_id
        
        # Filter by non-matching role
        pending = await approval_workflow.get_pending_requests(approver_role="admin")
        assert len(pending) == 0
    
    @pytest.mark.asyncio
    async def test_get_pending_requests_with_risk_filter(self, approval_workflow, sample_risk_assessment):
        """Test getting pending requests filtered by risk level."""
        request = await approval_workflow.create_approval_request(
            operation_type=OperationType.DATA_MODIFICATION,
            operation_description="High risk request",
            context={},
            risk_assessment=sample_risk_assessment,  # HIGH risk
            session_id="session_123",
            workflow_id="workflow_456",
            requested_by="agent_789",
        )
        
        # Filter by HIGH risk
        pending = await approval_workflow.get_pending_requests(risk_level=RiskLevel.HIGH)
        assert len(pending) == 1
        
        # Filter by CRITICAL risk (higher than HIGH)
        pending = await approval_workflow.get_pending_requests(risk_level=RiskLevel.CRITICAL)
        assert len(pending) == 0
    
    @pytest.mark.asyncio
    async def test_cleanup(self, approval_workflow, sample_risk_assessment):
        """Test cleanup cancels timers and clears state."""
        # Create a request to have some state
        request = await approval_workflow.create_approval_request(
            operation_type=OperationType.DATA_MODIFICATION,
            operation_description="Test cleanup",
            context={},
            risk_assessment=sample_risk_assessment,
            session_id="session_123",
            workflow_id="workflow_456",
            requested_by="agent_789",
        )
        
        # Verify state exists
        assert len(approval_workflow._pending_requests) == 1
        assert len(approval_workflow._request_timers) == 1
        
        # Cleanup
        await approval_workflow.cleanup()
        
        # Verify state is cleared
        assert len(approval_workflow._pending_requests) == 0
        assert len(approval_workflow._request_timers) == 0
    
    def test_risk_level_priority(self, approval_workflow):
        """Test risk level priority calculation."""
        assert approval_workflow._risk_level_priority(RiskLevel.LOW) == 1
        assert approval_workflow._risk_level_priority(RiskLevel.MEDIUM) == 2
        assert approval_workflow._risk_level_priority(RiskLevel.HIGH) == 3
        assert approval_workflow._risk_level_priority(RiskLevel.CRITICAL) == 4
    
    @pytest.mark.asyncio
    async def test_generate_risk_assessment_data_modification(self, approval_workflow):
        """Test risk assessment generation for data modification."""
        context = {
            "data_sensitivity_level": "sensitive",
            "regulatory_context": ["GDPR", "SOX"],
            "affected_resources": ["table1", "table2"],
        }
        
        risk_assessment = await approval_workflow._generate_risk_assessment(
            OperationType.DATA_MODIFICATION,
            context,
        )
        
        assert risk_assessment.risk_level == RiskLevel.HIGH
        assert "Sensitive data involved" in risk_assessment.risk_factors
        assert "regulatory implications" in risk_assessment.impact_description.lower()
        assert risk_assessment.likelihood_score > 0
        assert risk_assessment.severity_score > 0
        assert len(risk_assessment.compliance_implications) == 2
    
    @pytest.mark.asyncio
    async def test_generate_risk_assessment_schema_change(self, approval_workflow):
        """Test risk assessment generation for schema changes."""
        context = {}
        
        risk_assessment = await approval_workflow._generate_risk_assessment(
            OperationType.SCHEMA_CHANGE,
            context,
        )
        
        assert risk_assessment.risk_level == RiskLevel.CRITICAL
        assert risk_assessment.likelihood_score == 0.9
        assert risk_assessment.severity_score == 1.0
    
    def test_generate_context_explanation(self, approval_workflow, sample_risk_assessment):
        """Test context explanation generation."""
        context = {
            "affected_resources": ["table1", "table2", "table3"],
            "business_justification": "Data quality improvement",
            "regulatory_context": ["GDPR"],
        }
        
        explanation = approval_workflow._generate_context_explanation(
            OperationType.DATA_MODIFICATION,
            "Update customer records",
            context,
            sample_risk_assessment,
        )
        
        assert "Operation: Update customer records" in explanation
        assert "Type: Data Modification" in explanation
        assert "Risk Level: High" in explanation
        assert "Risk Factors: Sensitive data, Financial impact" in explanation
        assert "Affected Resources: table1, table2, table3" in explanation
        assert "Business Justification: Data quality improvement" in explanation
        assert "Regulatory Context: GDPR" in explanation
    
    def test_generate_context_explanation_many_resources(self, approval_workflow, sample_risk_assessment):
        """Test context explanation with many affected resources."""
        context = {
            "affected_resources": [f"table_{i}" for i in range(10)],
        }
        
        explanation = approval_workflow._generate_context_explanation(
            OperationType.DATA_MODIFICATION,
            "Bulk update",
            context,
            sample_risk_assessment,
        )
        
        # Should truncate resource list
        assert "table_0, table_1, table_2 and 7 others" in explanation
    
    def test_find_applicable_policy_match(self, approval_workflow):
        """Test finding applicable policy with match."""
        context = {"domain": "finance"}
        
        policy = approval_workflow._find_applicable_policy(
            OperationType.DATA_MODIFICATION,
            context,
        )
        
        assert policy is not None
        assert policy.name == "Data Modification Policy"
    
    def test_find_applicable_policy_no_match(self, approval_workflow):
        """Test finding applicable policy with no match."""
        context = {}
        
        policy = approval_workflow._find_applicable_policy(
            OperationType.EXTERNAL_API_CALL,  # Not in policy
            context,
        )
        
        assert policy is None
    
    @pytest.mark.asyncio
    async def test_should_escalate_high_risk(self, approval_workflow, sample_risk_assessment):
        """Test escalation decision for high risk request."""
        request = ApprovalRequest(
            operation_type=OperationType.DATA_MODIFICATION,
            operation_description="Test",
            context_explanation="Test",
            risk_assessment=sample_risk_assessment,  # HIGH risk
            requested_by="agent",
            session_id="session",
            workflow_id="workflow",
        )
        
        should_escalate = await approval_workflow._should_escalate(request)
        assert should_escalate is True
    
    @pytest.mark.asyncio
    async def test_should_escalate_low_risk(self, approval_workflow):
        """Test escalation decision for low risk request."""
        low_risk = RiskAssessment(
            risk_level=RiskLevel.LOW,
            risk_factors=[],
            impact_description="Low impact",
            likelihood_score=0.1,
            severity_score=0.2,
        )
        
        request = ApprovalRequest(
            operation_type=OperationType.EXTERNAL_API_CALL,
            operation_description="Test",
            context_explanation="Test",
            risk_assessment=low_risk,
            requested_by="agent",
            session_id="session",
            workflow_id="workflow",
        )
        
        should_escalate = await approval_workflow._should_escalate(request)
        assert should_escalate is False