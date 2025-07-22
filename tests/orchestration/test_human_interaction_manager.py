"""
Tests for HumanInteractionManager class.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.dataqa.orchestration.approval.models import (
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
)
from src.dataqa.orchestration.approval.manager import (
    FeedbackIntegrator,
    HumanInteractionManager,
    HumanInteractionError,
)


class TestFeedbackIntegrator:
    """Test cases for FeedbackIntegrator."""
    
    @pytest.fixture
    def feedback_integrator(self):
        """Create a FeedbackIntegrator instance."""
        return FeedbackIntegrator()
    
    @pytest.fixture
    def sample_feedback(self):
        """Create sample human feedback."""
        return HumanFeedback(
            feedback_type=FeedbackType.APPROVAL,
            request_id="request_123",
            feedback_text="Good decision, approved quickly",
            rating=4,
            context_tags=["data_modification", "high_risk", "financial"],
            session_id="session_123",
            provided_by="user_456",
        )
    
    @pytest.mark.asyncio
    async def test_process_feedback(self, feedback_integrator, sample_feedback):
        """Test processing human feedback."""
        await feedback_integrator.process_feedback(sample_feedback)
        
        assert len(feedback_integrator.feedback_history) == 1
        assert feedback_integrator.feedback_history[0] == sample_feedback
        
        # Check learning patterns were updated
        assert "data_modification" in feedback_integrator.learning_patterns
        pattern = feedback_integrator.learning_patterns["data_modification"]
        assert pattern["approval_count"] == 1
        assert pattern["total_count"] == 1
    
    @pytest.mark.asyncio
    async def test_get_learning_insights_with_similar_cases(self, feedback_integrator):
        """Test getting learning insights with similar cases."""
        # Add some feedback history
        feedback1 = HumanFeedback(
            feedback_type=FeedbackType.APPROVAL,
            request_id="req1",
            feedback_text="Approved",
            context_tags=["data_modification", "medium_risk"],
            session_id="session1",
            provided_by="user1",
        )
        
        feedback2 = HumanFeedback(
            feedback_type=FeedbackType.REJECTION,
            request_id="req2",
            feedback_text="Rejected",
            context_tags=["data_modification", "high_risk"],
            session_id="session2",
            provided_by="user2",
        )
        
        await feedback_integrator.process_feedback(feedback1)
        await feedback_integrator.process_feedback(feedback2)
        
        # Get insights for similar context
        context = {"tags": ["data_modification"]}
        insights = await feedback_integrator.get_learning_insights(context)
        
        assert insights["approval_likelihood"] == 0.5  # 1 approval out of 2 cases
        assert "data_modification" in insights["common_patterns"]
    
    @pytest.mark.asyncio
    async def test_get_learning_insights_no_similar_cases(self, feedback_integrator):
        """Test getting learning insights with no similar cases."""
        context = {"tags": ["unknown_operation"]}
        insights = await feedback_integrator.get_learning_insights(context)
        
        assert insights["approval_likelihood"] == 0.5  # Default
        assert len(insights["similar_cases"]) == 0
        assert len(insights["common_patterns"]) == 0
    
    @pytest.mark.asyncio
    async def test_extract_learning_patterns_multiple_feedback_types(self, feedback_integrator):
        """Test extracting learning patterns from multiple feedback types."""
        feedbacks = [
            HumanFeedback(
                feedback_type=FeedbackType.APPROVAL,
                request_id="req1",
                feedback_text="Approved",
                context_tags=["financial"],
                session_id="session1",
                provided_by="user1",
            ),
            HumanFeedback(
                feedback_type=FeedbackType.REJECTION,
                request_id="req2",
                feedback_text="Rejected",
                context_tags=["financial"],
                session_id="session2",
                provided_by="user2",
            ),
            HumanFeedback(
                feedback_type=FeedbackType.MODIFICATION,
                request_id="req3",
                feedback_text="Modified",
                context_tags=["financial"],
                session_id="session3",
                provided_by="user3",
            ),
        ]
        
        for feedback in feedbacks:
            await feedback_integrator.process_feedback(feedback)
        
        pattern = feedback_integrator.learning_patterns["financial"]
        assert pattern["approval_count"] == 1
        assert pattern["rejection_count"] == 1
        assert pattern["modification_count"] == 1
        assert pattern["total_count"] == 3


class TestHumanInteractionManager:
    """Test cases for HumanInteractionManager."""
    
    @pytest.fixture
    def sample_policy(self):
        """Create a sample approval policy."""
        return ApprovalPolicy(
            name="Test Policy",
            description="Test approval policy for data modifications",
            operation_types=[OperationType.DATA_MODIFICATION],
            required_roles=["data_steward"],
            timeout_policy=TimeoutPolicy(timeout_minutes=30),
        )
    
    @pytest.fixture
    def sample_escalation_rule(self):
        """Create a sample escalation rule."""
        return EscalationRule(
            name="Test Escalation",
            description="Test escalation rule for high risk operations",
            risk_threshold=RiskLevel.HIGH,
            escalate_to_roles=["senior_manager"],
        )
    
    @pytest.fixture
    def notification_callback(self):
        """Create a mock notification callback."""
        return AsyncMock()
    
    @pytest.fixture
    def interaction_manager(self, sample_policy, sample_escalation_rule, notification_callback):
        """Create a HumanInteractionManager instance."""
        return HumanInteractionManager(
            approval_policies=[sample_policy],
            escalation_rules=[sample_escalation_rule],
            notification_callback=notification_callback,
        )
    
    @pytest.fixture
    def sample_approval_request(self):
        """Create a sample approval request."""
        risk_assessment = RiskAssessment(
            risk_level=RiskLevel.MEDIUM,
            risk_factors=["Data modification"],
            impact_description="Update customer data",
            likelihood_score=0.5,
            severity_score=0.6,
        )
        
        return ApprovalRequest(
            operation_type=OperationType.DATA_MODIFICATION,
            operation_description="Update customer records",
            context_explanation="Updating customer contact information",
            risk_assessment=risk_assessment,
            requested_by="agent_123",
            session_id="session_456",
            workflow_id="workflow_789",
            required_approvers=["data_steward"],
        )
    
    @pytest.mark.asyncio
    async def test_request_approval_success(self, interaction_manager, sample_approval_request, notification_callback):
        """Test successful approval request submission."""
        request_id = await interaction_manager.request_approval(sample_approval_request)
        
        assert request_id == sample_approval_request.request_id
        assert request_id in interaction_manager.active_requests
        assert interaction_manager.metrics["total_requests"] == 1
        
        # Check notification was sent
        notification_callback.assert_called_once()
        call_args = notification_callback.call_args[0][0]
        assert call_args["type"] == "approval_request"
        assert call_args["request_id"] == request_id
    
    @pytest.mark.asyncio
    async def test_request_approval_creates_default_queue(self, sample_policy, sample_escalation_rule):
        """Test that default queue is created when none exists."""
        manager = HumanInteractionManager(
            approval_policies=[sample_policy],
            escalation_rules=[sample_escalation_rule],
        )
        
        risk_assessment = RiskAssessment(
            risk_level=RiskLevel.MEDIUM,
            risk_factors=[],
            impact_description="Test",
            likelihood_score=0.5,
            severity_score=0.5,
        )
        
        request = ApprovalRequest(
            operation_type=OperationType.DATA_MODIFICATION,
            operation_description="Test",
            context_explanation="Test",
            risk_assessment=risk_assessment,
            requested_by="agent",
            session_id="session",
            workflow_id="workflow",
        )
        
        await manager.request_approval(request)
        
        assert "default" in manager.approval_queues
        default_queue = manager.approval_queues["default"]
        assert request.request_id in default_queue.pending_requests
    
    @pytest.mark.asyncio
    async def test_process_approval_response_approved(self, interaction_manager, sample_approval_request):
        """Test processing an approved response."""
        # Submit request first
        request_id = await interaction_manager.request_approval(sample_approval_request)
        
        # Create approval response
        response = ApprovalResponse(
            request_id=request_id,
            status=ApprovalStatus.APPROVED,
            approved_by="user_123",
            comments="Looks good, approved",
        )
        
        # Process response
        await interaction_manager.process_approval_response(request_id, response)
        
        # Check metrics
        assert interaction_manager.metrics["approved_requests"] == 1
        assert interaction_manager.metrics["average_response_time_minutes"] > 0
        
        # Check request is no longer active
        assert request_id not in interaction_manager.active_requests
        assert request_id in interaction_manager.processed_requests
    
    @pytest.mark.asyncio
    async def test_process_approval_response_rejected(self, interaction_manager, sample_approval_request):
        """Test processing a rejected response."""
        request_id = await interaction_manager.request_approval(sample_approval_request)
        
        response = ApprovalResponse(
            request_id=request_id,
            status=ApprovalStatus.REJECTED,
            approved_by="user_123",
            comments="Insufficient justification",
        )
        
        await interaction_manager.process_approval_response(request_id, response)
        
        assert interaction_manager.metrics["rejected_requests"] == 1
        assert request_id not in interaction_manager.active_requests
    
    @pytest.mark.asyncio
    async def test_process_approval_response_nonexistent_request(self, interaction_manager):
        """Test processing response for non-existent request."""
        response = ApprovalResponse(
            request_id="nonexistent_123",
            status=ApprovalStatus.APPROVED,
        )
        
        with pytest.raises(HumanInteractionError, match="not found"):
            await interaction_manager.process_approval_response("nonexistent_123", response)
    
    @pytest.mark.asyncio
    async def test_handle_timeout_with_escalation(self, interaction_manager, sample_approval_request):
        """Test handling timeout with escalation."""
        # Create high-risk request
        sample_approval_request.risk_assessment.risk_level = RiskLevel.HIGH
        request_id = await interaction_manager.request_approval(sample_approval_request)
        
        timeout_event = TimeoutEvent(
            request_id=request_id,
            timeout_duration_minutes=30,
            escalation_triggered=True,
        )
        
        resolution = await interaction_manager.handle_timeout(request_id, timeout_event)
        
        assert resolution.resolution_type == "escalated"
        assert "escalated" in resolution.resolution_description
        assert "escalation" in interaction_manager.approval_queues
    
    @pytest.mark.asyncio
    async def test_handle_timeout_with_auto_rejection(self, interaction_manager, sample_approval_request):
        """Test handling timeout with auto-rejection."""
        request_id = await interaction_manager.request_approval(sample_approval_request)
        
        timeout_event = TimeoutEvent(
            request_id=request_id,
            timeout_duration_minutes=30,
            auto_rejected=True,
        )
        
        resolution = await interaction_manager.handle_timeout(request_id, timeout_event)
        
        assert resolution.resolution_type == "auto_rejected"
        assert "automatically rejected" in resolution.resolution_description
        assert request_id not in interaction_manager.active_requests
    
    @pytest.mark.asyncio
    async def test_integrate_feedback(self, interaction_manager):
        """Test integrating human feedback."""
        feedback = HumanFeedback(
            feedback_type=FeedbackType.LEARNING,
            request_id="request_123",
            feedback_text="This was a good decision",
            context_tags=["data_modification"],
            session_id="session_123",
            provided_by="user_456",
        )
        
        await interaction_manager.integrate_feedback(feedback)
        
        # Check feedback was processed
        assert len(interaction_manager.feedback_integrator.feedback_history) == 1
    
    @pytest.mark.asyncio
    async def test_get_queue_status_specific_queue(self, interaction_manager):
        """Test getting status for a specific queue."""
        # Create a queue
        queue = await interaction_manager.create_queue(
            name="test_queue",
            description="Test queue",
            assigned_roles=["tester"],
        )
        
        status = await interaction_manager.get_queue_status("test_queue")
        
        assert status["queue_name"] == "test_queue"
        assert status["pending_requests"] == 0
        assert status["active_requests"] == 0
        assert status["total_processed"] == 0
    
    @pytest.mark.asyncio
    async def test_get_queue_status_all_queues(self, interaction_manager):
        """Test getting status for all queues."""
        await interaction_manager.create_queue("queue1")
        await interaction_manager.create_queue("queue2")
        
        status = await interaction_manager.get_queue_status()
        
        assert status["total_queues"] == 2
        assert "queue1" in status["queues"]
        assert "queue2" in status["queues"]
        assert "overall_metrics" in status
    
    @pytest.mark.asyncio
    async def test_get_queue_status_nonexistent_queue(self, interaction_manager):
        """Test getting status for non-existent queue."""
        status = await interaction_manager.get_queue_status("nonexistent")
        
        assert "error" in status
        assert "not found" in status["error"]
    
    @pytest.mark.asyncio
    async def test_create_queue_success(self, interaction_manager):
        """Test creating a new approval queue."""
        queue = await interaction_manager.create_queue(
            name="custom_queue",
            description="Custom approval queue",
            assigned_roles=["approver", "manager"],
            assigned_users=["user1", "user2"],
        )
        
        assert queue.name == "custom_queue"
        assert queue.description == "Custom approval queue"
        assert "approver" in queue.assigned_roles
        assert "user1" in queue.assigned_users
        assert "custom_queue" in interaction_manager.approval_queues
    
    @pytest.mark.asyncio
    async def test_create_queue_duplicate_name(self, interaction_manager):
        """Test creating queue with duplicate name raises error."""
        await interaction_manager.create_queue("duplicate")
        
        with pytest.raises(HumanInteractionError, match="already exists"):
            await interaction_manager.create_queue("duplicate")
    
    @pytest.mark.asyncio
    async def test_route_to_queue_preferred_queue(self, interaction_manager, sample_approval_request):
        """Test routing to preferred queue."""
        # Create a specific queue
        await interaction_manager.create_queue("preferred_queue")
        
        queue = await interaction_manager._route_to_queue(sample_approval_request, "preferred_queue")
        
        assert queue is not None
        assert queue.name == "preferred_queue"
    
    @pytest.mark.asyncio
    async def test_route_to_queue_by_role_match(self, interaction_manager, sample_approval_request):
        """Test routing to queue by role matching."""
        # Create queue with matching role
        await interaction_manager.create_queue(
            name="role_queue",
            assigned_roles=["data_steward"],
        )
        
        # Set approval roles on request
        sample_approval_request.approval_roles = ["data_steward"]
        
        queue = await interaction_manager._route_to_queue(sample_approval_request)
        
        assert queue is not None
        assert queue.name == "role_queue"
    
    @pytest.mark.asyncio
    async def test_remove_from_queues(self, interaction_manager, sample_approval_request):
        """Test removing request from queues."""
        # Create queue and add request
        queue = await interaction_manager.create_queue("test_queue")
        queue.pending_requests.append("request_123")
        queue.active_requests.append("request_456")
        
        # Remove requests
        await interaction_manager._remove_from_queues("request_123")
        await interaction_manager._remove_from_queues("request_456")
        
        assert "request_123" not in queue.pending_requests
        assert "request_456" not in queue.active_requests
    
    @pytest.mark.asyncio
    async def test_handle_escalation(self, interaction_manager, sample_approval_request):
        """Test handling escalation."""
        timeout_event = TimeoutEvent(
            request_id=sample_approval_request.request_id,
            timeout_duration_minutes=30,
            escalation_triggered=True,
        )
        
        await interaction_manager._handle_escalation(sample_approval_request, timeout_event)
        
        # Check escalation queue was created
        assert "escalation" in interaction_manager.approval_queues
        escalation_queue = interaction_manager.approval_queues["escalation"]
        assert sample_approval_request.request_id in escalation_queue.pending_requests
        assert "senior_approver" in escalation_queue.assigned_roles
    
    @pytest.mark.asyncio
    async def test_generate_learning_feedback_approval(self, interaction_manager, sample_approval_request):
        """Test generating learning feedback for approval."""
        response = ApprovalResponse(
            request_id=sample_approval_request.request_id,
            status=ApprovalStatus.APPROVED,
            comments="Good request",
        )
        
        await interaction_manager._generate_learning_feedback(sample_approval_request, response)
        
        # Check feedback was generated and processed
        feedback_history = interaction_manager.feedback_integrator.feedback_history
        assert len(feedback_history) == 1
        assert feedback_history[0].feedback_type == FeedbackType.APPROVAL
        assert feedback_history[0].request_id == sample_approval_request.request_id
    
    @pytest.mark.asyncio
    async def test_generate_learning_feedback_rejection(self, interaction_manager, sample_approval_request):
        """Test generating learning feedback for rejection."""
        response = ApprovalResponse(
            request_id=sample_approval_request.request_id,
            status=ApprovalStatus.REJECTED,
            comments="Insufficient justification",
        )
        
        await interaction_manager._generate_learning_feedback(sample_approval_request, response)
        
        feedback_history = interaction_manager.feedback_integrator.feedback_history
        assert len(feedback_history) == 1
        assert feedback_history[0].feedback_type == FeedbackType.REJECTION
    
    def test_update_average_response_time_first_response(self, interaction_manager):
        """Test updating average response time for first response."""
        interaction_manager.metrics["approved_requests"] = 1
        interaction_manager._update_average_response_time(30.0)
        
        assert interaction_manager.metrics["average_response_time_minutes"] == 30.0
    
    def test_update_average_response_time_multiple_responses(self, interaction_manager):
        """Test updating average response time for multiple responses."""
        # Set up initial state
        interaction_manager.metrics["approved_requests"] = 1
        interaction_manager.metrics["average_response_time_minutes"] = 20.0
        
        # Add second response
        interaction_manager.metrics["approved_requests"] = 2
        interaction_manager._update_average_response_time(40.0)
        
        # Average should be (20 + 40) / 2 = 30
        assert interaction_manager.metrics["average_response_time_minutes"] == 30.0
    
    @pytest.mark.asyncio
    async def test_cleanup(self, interaction_manager, sample_approval_request):
        """Test cleanup clears all state."""
        # Add some state
        await interaction_manager.request_approval(sample_approval_request)
        await interaction_manager.create_queue("test_queue")
        
        # Verify state exists
        assert len(interaction_manager.active_requests) > 0
        assert len(interaction_manager.approval_queues) > 0
        
        # Cleanup
        await interaction_manager.cleanup()
        
        # Verify state is cleared
        assert len(interaction_manager.active_requests) == 0
        assert len(interaction_manager.processed_requests) == 0
        assert len(interaction_manager.approval_queues) == 0