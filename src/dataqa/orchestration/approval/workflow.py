"""
ApprovalWorkflow for identifying and managing sensitive operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from ...exceptions import DataQAError
from .models import (
    AlternativeAction,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    EscalationRule,
    OperationType,
    RiskAssessment,
    RiskLevel,
    TimeoutEvent,
    TimeoutPolicy,
)

logger = logging.getLogger(__name__)


class ApprovalWorkflowError(DataQAError):
    """Errors related to approval workflow operations."""
    pass


class ApprovalWorkflow:
    """
    Workflow for identifying and managing sensitive operations that require human approval.
    
    This class handles:
    - Identifying operations that require approval based on policies
    - Generating approval requests with risk assessments
    - Managing approval queues and routing
    - Handling timeouts and escalations
    """
    
    def __init__(
        self,
        policies: Optional[List[ApprovalPolicy]] = None,
        escalation_rules: Optional[List[EscalationRule]] = None,
        default_timeout_minutes: int = 60,
    ):
        """
        Initialize the approval workflow.
        
        Args:
            policies: List of approval policies to apply
            escalation_rules: List of escalation rules for timeouts
            default_timeout_minutes: Default timeout for approval requests
        """
        self.policies = policies or []
        self.escalation_rules = escalation_rules or []
        self.default_timeout_minutes = default_timeout_minutes
        
        # Internal state
        self._pending_requests: Dict[str, ApprovalRequest] = {}
        self._request_timers: Dict[str, asyncio.Task] = {}
        
        logger.info(
            f"ApprovalWorkflow initialized with {len(self.policies)} policies "
            f"and {len(self.escalation_rules)} escalation rules"
        )
    
    async def requires_approval(
        self,
        operation_type: OperationType,
        context: Dict[str, Any],
        risk_assessment: Optional[RiskAssessment] = None,
    ) -> bool:
        """
        Determine if an operation requires human approval.
        
        Args:
            operation_type: Type of operation being performed
            context: Context information about the operation
            risk_assessment: Optional pre-computed risk assessment
            
        Returns:
            True if approval is required, False otherwise
        """
        try:
            # Check each policy to see if it applies
            for policy in self.policies:
                if not policy.enabled:
                    continue
                
                # Check operation type match
                if policy.operation_types and operation_type not in policy.operation_types:
                    continue
                
                # Check risk threshold
                if risk_assessment and risk_assessment.risk_level.value < policy.risk_threshold.value:
                    continue
                
                # Check data sensitivity levels
                data_sensitivity = context.get("data_sensitivity_level")
                if (policy.data_sensitivity_levels and 
                    data_sensitivity not in policy.data_sensitivity_levels):
                    continue
                
                # Check resource patterns
                if policy.resource_patterns:
                    affected_resources = context.get("affected_resources", [])
                    if not any(
                        any(pattern in resource for pattern in policy.resource_patterns)
                        for resource in affected_resources
                    ):
                        continue
                
                # Check domain applicability
                domain = context.get("domain")
                if policy.applicable_domains and domain not in policy.applicable_domains:
                    continue
                
                logger.info(f"Policy '{policy.name}' requires approval for {operation_type}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking approval requirements: {e}")
            # Fail safe - require approval if we can't determine
            return True
    
    async def create_approval_request(
        self,
        operation_type: OperationType,
        operation_description: str,
        context: Dict[str, Any],
        risk_assessment: Optional[RiskAssessment] = None,
        alternatives: Optional[List[AlternativeAction]] = None,
        session_id: str = "",
        workflow_id: str = "",
        requested_by: str = "",
    ) -> ApprovalRequest:
        """
        Create an approval request for a sensitive operation.
        
        Args:
            operation_type: Type of operation requiring approval
            operation_description: Description of the operation
            context: Context information about the operation
            risk_assessment: Risk assessment for the operation
            alternatives: Alternative actions that could be taken
            session_id: ID of the execution session
            workflow_id: ID of the workflow
            requested_by: ID of the agent or user requesting approval
            
        Returns:
            ApprovalRequest object
        """
        try:
            # Generate risk assessment if not provided
            if risk_assessment is None:
                risk_assessment = await self._generate_risk_assessment(
                    operation_type, context
                )
            
            # Find applicable policy for timeout and approval requirements
            applicable_policy = self._find_applicable_policy(operation_type, context)
            
            # Create timeout policy
            timeout_policy = TimeoutPolicy()
            if applicable_policy:
                timeout_policy = applicable_policy.timeout_policy
            else:
                timeout_policy.timeout_minutes = self.default_timeout_minutes
            
            # Generate context explanation
            context_explanation = self._generate_context_explanation(
                operation_type, operation_description, context, risk_assessment
            )
            
            # Create the approval request
            request = ApprovalRequest(
                operation_type=operation_type,
                operation_description=operation_description,
                context_explanation=context_explanation,
                risk_assessment=risk_assessment,
                alternative_options=alternatives or [],
                timeout_policy=timeout_policy,
                requested_by=requested_by,
                session_id=session_id,
                workflow_id=workflow_id,
                required_approvers=applicable_policy.required_roles if applicable_policy else [],
                minimum_approvals=applicable_policy.minimum_approvals if applicable_policy else 1,
                affected_resources=context.get("affected_resources", []),
                data_sensitivity_level=context.get("data_sensitivity_level"),
                regulatory_context=context.get("regulatory_context", []),
                business_justification=context.get("business_justification"),
                metadata=context.get("metadata", {}),
            )
            
            # Store the request
            self._pending_requests[request.request_id] = request
            
            # Start timeout timer
            await self._start_timeout_timer(request)
            
            logger.info(
                f"Created approval request {request.request_id} for {operation_type} "
                f"with risk level {risk_assessment.risk_level}"
            )
            
            return request
            
        except Exception as e:
            logger.error(f"Error creating approval request: {e}")
            raise ApprovalWorkflowError(f"Failed to create approval request: {e}")
    
    async def process_response(
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
            if request_id not in self._pending_requests:
                raise ApprovalWorkflowError(f"Approval request {request_id} not found")
            
            request = self._pending_requests[request_id]
            
            # Cancel timeout timer
            if request_id in self._request_timers:
                self._request_timers[request_id].cancel()
                del self._request_timers[request_id]
            
            # Remove from pending requests
            del self._pending_requests[request_id]
            
            logger.info(
                f"Processed approval response for request {request_id}: {response.status}"
            )
            
        except Exception as e:
            logger.error(f"Error processing approval response: {e}")
            raise ApprovalWorkflowError(f"Failed to process approval response: {e}")
    
    async def handle_timeout(self, request_id: str) -> TimeoutEvent:
        """
        Handle timeout for an approval request.
        
        Args:
            request_id: ID of the timed-out request
            
        Returns:
            TimeoutEvent describing what happened
        """
        try:
            if request_id not in self._pending_requests:
                raise ApprovalWorkflowError(f"Approval request {request_id} not found")
            
            request = self._pending_requests[request_id]
            
            # Create timeout event
            timeout_event = TimeoutEvent(
                request_id=request_id,
                timeout_duration_minutes=request.timeout_policy.timeout_minutes,
                pending_approvers=request.required_approvers.copy(),
            )
            
            # Check if escalation should be triggered
            should_escalate = await self._should_escalate(request)
            if should_escalate:
                timeout_event.escalation_triggered = True
                await self._trigger_escalation(request)
            
            # Check if auto-rejection should occur
            if request.timeout_policy.auto_reject_on_timeout:
                timeout_event.auto_rejected = True
                # Create rejection response
                rejection_response = ApprovalResponse(
                    request_id=request_id,
                    status=ApprovalStatus.TIMEOUT,
                    comments="Request timed out and was automatically rejected",
                )
                await self.process_response(request_id, rejection_response)
            
            # Execute fallback action if specified
            if request.timeout_policy.fallback_action:
                timeout_event.fallback_action_taken = request.timeout_policy.fallback_action
                # Note: Actual fallback execution would be handled by the calling system
            
            logger.warning(
                f"Approval request {request_id} timed out after "
                f"{request.timeout_policy.timeout_minutes} minutes"
            )
            
            return timeout_event
            
        except Exception as e:
            logger.error(f"Error handling timeout for request {request_id}: {e}")
            raise ApprovalWorkflowError(f"Failed to handle timeout: {e}")
    
    async def get_pending_requests(
        self,
        approver_role: Optional[str] = None,
        risk_level: Optional[RiskLevel] = None,
    ) -> List[ApprovalRequest]:
        """
        Get pending approval requests, optionally filtered.
        
        Args:
            approver_role: Filter by required approver role
            risk_level: Filter by minimum risk level
            
        Returns:
            List of pending approval requests
        """
        requests = list(self._pending_requests.values())
        
        if approver_role:
            requests = [
                req for req in requests
                if approver_role in req.required_approvers or approver_role in req.approval_roles
            ]
        
        if risk_level:
            requests = [
                req for req in requests
                if self._risk_level_priority(req.risk_assessment.risk_level) >= self._risk_level_priority(risk_level)
            ]
        
        # Sort by risk level (highest first) and then by request time
        requests.sort(
            key=lambda r: (
                -self._risk_level_priority(r.risk_assessment.risk_level),
                r.requested_at
            )
        )
        
        return requests
    
    def _find_applicable_policy(
        self,
        operation_type: OperationType,
        context: Dict[str, Any],
    ) -> Optional[ApprovalPolicy]:
        """Find the most applicable policy for an operation."""
        applicable_policies = []
        
        for policy in self.policies:
            if not policy.enabled:
                continue
            
            if policy.operation_types and operation_type not in policy.operation_types:
                continue
            
            # Check domain applicability
            domain = context.get("domain")
            if policy.applicable_domains and domain not in policy.applicable_domains:
                continue
            
            applicable_policies.append(policy)
        
        # Return highest priority policy
        if applicable_policies:
            return max(applicable_policies, key=lambda p: p.priority)
        
        return None
    
    async def _generate_risk_assessment(
        self,
        operation_type: OperationType,
        context: Dict[str, Any],
    ) -> RiskAssessment:
        """Generate a risk assessment for an operation."""
        # This is a simplified risk assessment - in practice, this would be more sophisticated
        risk_factors = []
        impact_description = f"Operation of type {operation_type}"
        
        # Assess based on operation type
        base_risk = {
            OperationType.DATA_MODIFICATION: RiskLevel.HIGH,
            OperationType.SCHEMA_CHANGE: RiskLevel.CRITICAL,
            OperationType.EXTERNAL_API_CALL: RiskLevel.MEDIUM,
            OperationType.SENSITIVE_DATA_ACCESS: RiskLevel.HIGH,
            OperationType.FINANCIAL_CALCULATION: RiskLevel.HIGH,
            OperationType.REGULATORY_COMPLIANCE: RiskLevel.CRITICAL,
            OperationType.SYSTEM_CONFIGURATION: RiskLevel.HIGH,
            OperationType.USER_DATA_EXPORT: RiskLevel.HIGH,
        }.get(operation_type, RiskLevel.MEDIUM)
        
        # Adjust based on context
        data_sensitivity = context.get("data_sensitivity_level", "")
        if "sensitive" in data_sensitivity.lower() or "confidential" in data_sensitivity.lower():
            risk_factors.append("Sensitive data involved")
            if base_risk == RiskLevel.LOW:
                base_risk = RiskLevel.MEDIUM
            elif base_risk == RiskLevel.MEDIUM:
                base_risk = RiskLevel.HIGH
        
        if context.get("regulatory_context"):
            risk_factors.append("Regulatory implications")
            impact_description += " with regulatory implications"
        
        if context.get("affected_resources"):
            resource_count = len(context["affected_resources"])
            if resource_count > 10:
                risk_factors.append(f"Large number of affected resources ({resource_count})")
        
        # Calculate likelihood and severity scores
        likelihood_score = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.7,
            RiskLevel.CRITICAL: 0.9,
        }[base_risk]
        
        severity_score = {
            RiskLevel.LOW: 0.3,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 1.0,
        }[base_risk]
        
        return RiskAssessment(
            risk_level=base_risk,
            risk_factors=risk_factors,
            impact_description=impact_description,
            likelihood_score=likelihood_score,
            severity_score=severity_score,
            mitigation_strategies=[
                "Human approval required",
                "Audit trail maintained",
                "Rollback capability available",
            ],
            compliance_implications=context.get("regulatory_context", []),
        )
    
    def _generate_context_explanation(
        self,
        operation_type: OperationType,
        operation_description: str,
        context: Dict[str, Any],
        risk_assessment: RiskAssessment,
    ) -> str:
        """Generate a human-readable explanation of the operation context."""
        explanation_parts = [
            f"Operation: {operation_description}",
            f"Type: {operation_type.value.replace('_', ' ').title()}",
            f"Risk Level: {risk_assessment.risk_level.value.title()}",
        ]
        
        if risk_assessment.risk_factors:
            explanation_parts.append(f"Risk Factors: {', '.join(risk_assessment.risk_factors)}")
        
        if context.get("affected_resources"):
            resources = context["affected_resources"]
            if len(resources) <= 3:
                explanation_parts.append(f"Affected Resources: {', '.join(resources)}")
            else:
                explanation_parts.append(
                    f"Affected Resources: {', '.join(resources[:3])} and {len(resources) - 3} others"
                )
        
        if context.get("business_justification"):
            explanation_parts.append(f"Business Justification: {context['business_justification']}")
        
        if context.get("regulatory_context"):
            explanation_parts.append(f"Regulatory Context: {', '.join(context['regulatory_context'])}")
        
        return "\n".join(explanation_parts)
    
    async def _start_timeout_timer(self, request: ApprovalRequest) -> None:
        """Start a timeout timer for an approval request."""
        async def timeout_handler():
            try:
                await asyncio.sleep(request.timeout_policy.timeout_minutes * 60)
                await self.handle_timeout(request.request_id)
            except asyncio.CancelledError:
                # Timer was cancelled (normal when request is processed)
                pass
            except Exception as e:
                logger.error(f"Error in timeout handler for request {request.request_id}: {e}")
        
        timer_task = asyncio.create_task(timeout_handler())
        self._request_timers[request.request_id] = timer_task
    
    async def _should_escalate(self, request: ApprovalRequest) -> bool:
        """Determine if a request should be escalated based on escalation rules."""
        for rule in self.escalation_rules:
            if not rule.enabled:
                continue
            
            # Check risk threshold
            request_risk_value = self._risk_level_priority(request.risk_assessment.risk_level)
            rule_risk_value = self._risk_level_priority(rule.risk_threshold)
            if request_risk_value >= rule_risk_value:
                return True
            
            # Check other trigger conditions (simplified)
            if "high_value" in rule.trigger_conditions and request.metadata.get("high_value"):
                return True
        
        return False
    
    async def _trigger_escalation(self, request: ApprovalRequest) -> None:
        """Trigger escalation for a request."""
        # In a real implementation, this would send notifications, update queues, etc.
        logger.warning(f"Escalating approval request {request.request_id}")
        
        # Find applicable escalation rules
        for rule in self.escalation_rules:
            if not rule.enabled:
                continue
            
            request_risk_value = self._risk_level_priority(request.risk_assessment.risk_level)
            rule_risk_value = self._risk_level_priority(rule.risk_threshold)
            if request_risk_value >= rule_risk_value:
                # Add escalation roles to required approvers
                request.required_approvers.extend(rule.escalate_to_roles)
                request.required_approvers.extend(rule.escalate_to_users)
                
                # Remove duplicates
                request.required_approvers = list(set(request.required_approvers))
                
                logger.info(
                    f"Escalated request {request.request_id} to roles: {rule.escalate_to_roles}"
                )
                break
    
    def _risk_level_priority(self, risk_level: RiskLevel) -> int:
        """Get numeric priority for risk level (higher = more urgent)."""
        return {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }[risk_level]
    
    async def cleanup(self) -> None:
        """Clean up resources and cancel pending timers."""
        for timer in self._request_timers.values():
            timer.cancel()
        
        self._request_timers.clear()
        self._pending_requests.clear()
        
        logger.info("ApprovalWorkflow cleanup completed")