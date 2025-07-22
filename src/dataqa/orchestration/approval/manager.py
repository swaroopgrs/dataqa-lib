"""
HumanInteractionManager for managing approval queues and human feedback integration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

from ...exceptions import DataQAError
from .models import (
    ApprovalPolicy,
    ApprovalQueue,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    EscalationRule,
    FeedbackType,
    HumanFeedback,
    TimeoutEvent,
    TimeoutResolution,
)
from .workflow import ApprovalWorkflow

logger = logging.getLogger(__name__)


class HumanInteractionError(DataQAError):
    """Errors related to human interaction management."""
    pass


class FeedbackIntegrator:
    """Integrates human feedback for continuous learning."""
    
    def __init__(self):
        self.feedback_history: List[HumanFeedback] = []
        self.learning_patterns: Dict[str, Any] = {}
    
    async def process_feedback(self, feedback: HumanFeedback) -> None:
        """Process and learn from human feedback."""
        try:
            self.feedback_history.append(feedback)
            
            # Extract learning patterns
            await self._extract_learning_patterns(feedback)
            
            # Update risk assessment models (simplified)
            await self._update_risk_models(feedback)
            
            logger.info(f"Processed feedback {feedback.feedback_id} of type {feedback.feedback_type}")
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            raise HumanInteractionError(f"Failed to process feedback: {e}")
    
    async def get_learning_insights(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get learning insights for similar contexts."""
        insights = {
            "similar_cases": [],
            "common_patterns": [],
            "risk_adjustments": {},
            "approval_likelihood": 0.5,
        }
        
        try:
            # Find similar feedback cases
            similar_feedback = [
                fb for fb in self.feedback_history
                if any(tag in context.get("tags", []) for tag in fb.context_tags)
            ]
            
            if similar_feedback:
                # Calculate approval likelihood
                approvals = sum(1 for fb in similar_feedback if fb.feedback_type == FeedbackType.APPROVAL)
                insights["approval_likelihood"] = approvals / len(similar_feedback)
                
                # Extract common patterns
                all_tags = []
                for fb in similar_feedback:
                    all_tags.extend(fb.context_tags)
                
                tag_counts = {}
                for tag in all_tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
                insights["common_patterns"] = [
                    tag for tag, count in tag_counts.items()
                    if count >= len(similar_feedback) * 0.5  # Appears in 50%+ of cases
                ]
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return insights
    
    async def _extract_learning_patterns(self, feedback: HumanFeedback) -> None:
        """Extract learning patterns from feedback."""
        # Update pattern tracking
        for tag in feedback.context_tags:
            if tag not in self.learning_patterns:
                self.learning_patterns[tag] = {
                    "approval_count": 0,
                    "rejection_count": 0,
                    "modification_count": 0,
                    "total_count": 0,
                }
            
            pattern = self.learning_patterns[tag]
            pattern["total_count"] += 1
            
            if feedback.feedback_type == FeedbackType.APPROVAL:
                pattern["approval_count"] += 1
            elif feedback.feedback_type == FeedbackType.REJECTION:
                pattern["rejection_count"] += 1
            elif feedback.feedback_type == FeedbackType.MODIFICATION:
                pattern["modification_count"] += 1
    
    async def _update_risk_models(self, feedback: HumanFeedback) -> None:
        """Update risk assessment models based on feedback."""
        # This would integrate with ML models in a real implementation
        # For now, we just log the learning opportunity
        logger.info(f"Learning opportunity: {feedback.feedback_type} for tags {feedback.context_tags}")


class HumanInteractionManager:
    """
    Manages human approval queues, timeout handling, and feedback integration.
    
    This class handles:
    - Managing approval request queues
    - Routing requests to appropriate approvers
    - Handling timeouts and escalations
    - Integrating human feedback for learning
    """
    
    def __init__(
        self,
        approval_policies: Optional[List[ApprovalPolicy]] = None,
        escalation_rules: Optional[List[EscalationRule]] = None,
        notification_callback: Optional[Callable] = None,
    ):
        """
        Initialize the human interaction manager.
        
        Args:
            approval_policies: List of approval policies
            escalation_rules: List of escalation rules
            notification_callback: Callback function for sending notifications
        """
        self.approval_workflow = ApprovalWorkflow(
            policies=approval_policies or [],
            escalation_rules=escalation_rules or [],
        )
        self.feedback_integrator = FeedbackIntegrator()
        self.notification_callback = notification_callback
        
        # Approval queues
        self.approval_queues: Dict[str, ApprovalQueue] = {}
        self.active_requests: Dict[str, ApprovalRequest] = {}
        self.processed_requests: Dict[str, ApprovalResponse] = {}
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "approved_requests": 0,
            "rejected_requests": 0,
            "timed_out_requests": 0,
            "average_response_time_minutes": 0.0,
        }
        
        logger.info("HumanInteractionManager initialized")
    
    async def request_approval(
        self,
        approval_request: ApprovalRequest,
        queue_name: Optional[str] = None,
    ) -> str:
        """
        Submit an approval request for human review.
        
        Args:
            approval_request: The approval request to submit
            queue_name: Optional specific queue to route to
            
        Returns:
            Request ID for tracking
        """
        try:
            # Store the request
            self.active_requests[approval_request.request_id] = approval_request
            self.metrics["total_requests"] += 1
            
            # Also store in approval workflow for processing
            self.approval_workflow._pending_requests[approval_request.request_id] = approval_request
            
            # Route to appropriate queue
            target_queue = await self._route_to_queue(approval_request, queue_name)
            
            # Add to queue
            if target_queue:
                target_queue.pending_requests.append(approval_request.request_id)
                target_queue.updated_at = datetime.utcnow()
            
            # Send notifications
            if self.notification_callback:
                await self._send_notifications(approval_request, target_queue)
            
            # Get learning insights
            context = {
                "operation_type": approval_request.operation_type,
                "risk_level": approval_request.risk_assessment.risk_level,
                "tags": approval_request.context_explanation.split(),
            }
            insights = await self.feedback_integrator.get_learning_insights(context)
            
            logger.info(
                f"Approval request {approval_request.request_id} submitted to queue "
                f"{target_queue.name if target_queue else 'default'} "
                f"(predicted approval likelihood: {insights['approval_likelihood']:.2f})"
            )
            
            return approval_request.request_id
            
        except Exception as e:
            logger.error(f"Error requesting approval: {e}")
            raise HumanInteractionError(f"Failed to request approval: {e}")
    
    async def process_approval_response(
        self,
        request_id: str,
        response: ApprovalResponse,
    ) -> None:
        """
        Process a response to an approval request.
        
        Args:
            request_id: ID of the approval request
            response: Response from the approver
        """
        try:
            if request_id not in self.active_requests:
                raise HumanInteractionError(f"Active approval request {request_id} not found")
            
            request = self.active_requests[request_id]
            
            # Update metrics
            if response.status == ApprovalStatus.APPROVED:
                self.metrics["approved_requests"] += 1
            elif response.status == ApprovalStatus.REJECTED:
                self.metrics["rejected_requests"] += 1
            elif response.status == ApprovalStatus.TIMEOUT:
                self.metrics["timed_out_requests"] += 1
            
            # Calculate response time
            response_time = (response.responded_at - request.requested_at).total_seconds() / 60
            self._update_average_response_time(response_time)
            
            # Remove from active requests and queues
            del self.active_requests[request_id]
            await self._remove_from_queues(request_id)
            
            # Store processed response
            self.processed_requests[request_id] = response
            
            # Process with approval workflow
            await self.approval_workflow.process_response(request_id, response)
            
            # Generate learning feedback
            await self._generate_learning_feedback(request, response)
            
            logger.info(
                f"Processed approval response for {request_id}: {response.status} "
                f"(response time: {response_time:.1f} minutes)"
            )
            
        except Exception as e:
            logger.error(f"Error processing approval response: {e}")
            raise HumanInteractionError(f"Failed to process approval response: {e}")
    
    async def handle_timeout(
        self,
        request_id: str,
        timeout_event: TimeoutEvent,
    ) -> TimeoutResolution:
        """
        Handle timeout for an approval request.
        
        Args:
            request_id: ID of the timed-out request
            timeout_event: Details of the timeout event
            
        Returns:
            TimeoutResolution describing how the timeout was handled
        """
        try:
            if request_id not in self.active_requests:
                raise HumanInteractionError(f"Active approval request {request_id} not found")
            
            request = self.active_requests[request_id]
            
            # Determine resolution type
            resolution_type = "timeout"
            resolution_description = f"Request timed out after {timeout_event.timeout_duration_minutes} minutes"
            
            if timeout_event.escalation_triggered:
                resolution_type = "escalated"
                resolution_description += " and was escalated"
                
                # Handle escalation
                await self._handle_escalation(request, timeout_event)
            
            elif timeout_event.auto_rejected:
                resolution_type = "auto_rejected"
                resolution_description += " and was automatically rejected"
                
                # Create auto-rejection response
                auto_response = ApprovalResponse(
                    request_id=request_id,
                    status=ApprovalStatus.TIMEOUT,
                    comments="Automatically rejected due to timeout",
                )
                await self.process_approval_response(request_id, auto_response)
            
            elif timeout_event.fallback_action_taken:
                resolution_type = "fallback"
                resolution_description += f" and fallback action '{timeout_event.fallback_action_taken}' was taken"
            
            # Create resolution
            resolution = TimeoutResolution(
                timeout_event_id=timeout_event.event_id,
                resolution_type=resolution_type,
                resolution_description=resolution_description,
            )
            
            logger.warning(f"Handled timeout for request {request_id}: {resolution_type}")
            
            return resolution
            
        except Exception as e:
            logger.error(f"Error handling timeout: {e}")
            raise HumanInteractionError(f"Failed to handle timeout: {e}")
    
    async def integrate_feedback(self, feedback: HumanFeedback) -> None:
        """
        Integrate human feedback for continuous learning.
        
        Args:
            feedback: Human feedback to integrate
        """
        try:
            await self.feedback_integrator.process_feedback(feedback)
            
            logger.info(f"Integrated feedback {feedback.feedback_id}")
            
        except Exception as e:
            logger.error(f"Error integrating feedback: {e}")
            raise HumanInteractionError(f"Failed to integrate feedback: {e}")
    
    async def get_queue_status(self, queue_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of approval queues.
        
        Args:
            queue_name: Optional specific queue name
            
        Returns:
            Queue status information
        """
        if queue_name:
            if queue_name not in self.approval_queues:
                return {"error": f"Queue '{queue_name}' not found"}
            
            queue = self.approval_queues[queue_name]
            return {
                "queue_name": queue.name,
                "pending_requests": len(queue.pending_requests),
                "active_requests": len(queue.active_requests),
                "total_processed": queue.total_processed,
                "average_response_time_minutes": queue.average_response_time_minutes,
                "approval_rate": queue.approval_rate,
            }
        else:
            # Return status for all queues
            return {
                "total_queues": len(self.approval_queues),
                "queues": {
                    name: {
                        "pending_requests": len(queue.pending_requests),
                        "active_requests": len(queue.active_requests),
                        "total_processed": queue.total_processed,
                    }
                    for name, queue in self.approval_queues.items()
                },
                "overall_metrics": self.metrics,
            }
    
    async def create_queue(
        self,
        name: str,
        description: Optional[str] = None,
        assigned_roles: Optional[List[str]] = None,
        assigned_users: Optional[List[str]] = None,
    ) -> ApprovalQueue:
        """
        Create a new approval queue.
        
        Args:
            name: Name of the queue
            description: Optional description
            assigned_roles: Roles assigned to this queue
            assigned_users: Users assigned to this queue
            
        Returns:
            Created ApprovalQueue
        """
        if name in self.approval_queues:
            raise HumanInteractionError(f"Queue '{name}' already exists")
        
        queue = ApprovalQueue(
            name=name,
            description=description,
            assigned_roles=assigned_roles or [],
            assigned_users=assigned_users or [],
        )
        
        self.approval_queues[name] = queue
        
        logger.info(f"Created approval queue '{name}'")
        
        return queue
    
    async def _route_to_queue(
        self,
        request: ApprovalRequest,
        preferred_queue: Optional[str] = None,
    ) -> Optional[ApprovalQueue]:
        """Route an approval request to the appropriate queue."""
        if preferred_queue and preferred_queue in self.approval_queues:
            return self.approval_queues[preferred_queue]
        
        # Find queue based on required approvers
        for queue in self.approval_queues.values():
            # Check if any required approvers match queue assignments
            if (any(role in queue.assigned_roles for role in request.approval_roles) or
                any(user in queue.assigned_users for user in request.required_approvers)):
                return queue
        
        # Create default queue if none exists
        if not self.approval_queues:
            return await self.create_queue(
                name="default",
                description="Default approval queue",
            )
        
        # Return first available queue
        return next(iter(self.approval_queues.values()))
    
    async def _send_notifications(
        self,
        request: ApprovalRequest,
        queue: Optional[ApprovalQueue],
    ) -> None:
        """Send notifications for a new approval request."""
        if not self.notification_callback:
            return
        
        notification_data = {
            "type": "approval_request",
            "request_id": request.request_id,
            "operation_type": request.operation_type,
            "risk_level": request.risk_assessment.risk_level,
            "queue_name": queue.name if queue else "default",
            "timeout_minutes": request.timeout_policy.timeout_minutes,
        }
        
        try:
            await self.notification_callback(notification_data)
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    async def _remove_from_queues(self, request_id: str) -> None:
        """Remove a request from all queues."""
        for queue in self.approval_queues.values():
            if request_id in queue.pending_requests:
                queue.pending_requests.remove(request_id)
            if request_id in queue.active_requests:
                queue.active_requests.remove(request_id)
            queue.updated_at = datetime.utcnow()
    
    async def _handle_escalation(
        self,
        request: ApprovalRequest,
        timeout_event: TimeoutEvent,
    ) -> None:
        """Handle escalation of a timed-out request."""
        # In a real implementation, this would:
        # 1. Move request to higher-priority queue
        # 2. Notify escalation contacts
        # 3. Adjust timeout policies
        # 4. Update request metadata
        
        logger.info(f"Escalating request {request.request_id}")
        
        # Find or create escalation queue
        escalation_queue_name = "escalation"
        if escalation_queue_name not in self.approval_queues:
            await self.create_queue(
                name=escalation_queue_name,
                description="Escalated approval requests",
                assigned_roles=["senior_approver", "manager"],
            )
        
        escalation_queue = self.approval_queues[escalation_queue_name]
        escalation_queue.pending_requests.append(request.request_id)
        escalation_queue.updated_at = datetime.utcnow()
    
    async def _generate_learning_feedback(
        self,
        request: ApprovalRequest,
        response: ApprovalResponse,
    ) -> None:
        """Generate learning feedback from approval interactions."""
        feedback_type = {
            ApprovalStatus.APPROVED: FeedbackType.APPROVAL,
            ApprovalStatus.REJECTED: FeedbackType.REJECTION,
            ApprovalStatus.TIMEOUT: FeedbackType.ESCALATION,
        }.get(response.status, FeedbackType.LEARNING)
        
        # Extract context tags from request
        context_tags = [
            request.operation_type,
            request.risk_assessment.risk_level,
        ]
        context_tags.extend(request.risk_assessment.risk_factors)
        
        if request.data_sensitivity_level:
            context_tags.append(request.data_sensitivity_level)
        
        # Create learning feedback
        learning_feedback = HumanFeedback(
            feedback_type=feedback_type,
            request_id=request.request_id,
            response_id=response.response_id,
            feedback_text=response.comments or f"System-generated feedback for {response.status}",
            context_tags=context_tags,
            session_id=request.session_id,
            provided_by="system",  # System-generated feedback
        )
        
        await self.feedback_integrator.process_feedback(learning_feedback)
    
    def _update_average_response_time(self, new_response_time: float) -> None:
        """Update the running average response time."""
        total_responses = (
            self.metrics["approved_requests"] + 
            self.metrics["rejected_requests"] + 
            self.metrics["timed_out_requests"]
        )
        
        if total_responses == 1:
            self.metrics["average_response_time_minutes"] = new_response_time
        else:
            current_avg = self.metrics["average_response_time_minutes"]
            self.metrics["average_response_time_minutes"] = (
                (current_avg * (total_responses - 1) + new_response_time) / total_responses
            )
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.approval_workflow.cleanup()
        
        self.active_requests.clear()
        self.processed_requests.clear()
        self.approval_queues.clear()
        
        logger.info("HumanInteractionManager cleanup completed")