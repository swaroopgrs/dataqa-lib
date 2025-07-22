"""
Replanning engine for dynamic plan adaptation.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..models import ExecutionState, ExecutionStatus
from .models import (
    ContextPreservationStrategy,
    ExecutionContext,
    ExecutionStep,
    IntermediateResult,
    Plan,
    ReplanningEvent,
    ReplanningTrigger,
    ReplanningTriggerType,
)
from .planner import AdaptivePlanner

logger = logging.getLogger(__name__)


class ReplanningEngine:
    """
    Engine for detecting replanning triggers and generating revised plans.
    
    Monitors execution state, detects when replanning is needed based on configurable
    triggers, and generates revised plans while preserving successful intermediate results.
    """
    
    def __init__(
        self,
        max_replanning_iterations: int = 3,
        context_preservation_strategy: ContextPreservationStrategy = ContextPreservationStrategy.PRESERVE_SUCCESSFUL,
        planner: Optional[AdaptivePlanner] = None
    ):
        """
        Initialize the replanning engine.
        
        Args:
            max_replanning_iterations: Maximum number of replanning attempts
            context_preservation_strategy: Strategy for preserving context during replanning
            planner: Adaptive planner instance for generating revised plans
        """
        self.max_replanning_iterations = max_replanning_iterations
        self.context_preservation_strategy = context_preservation_strategy
        self.planner = planner or AdaptivePlanner()
        self.replanning_triggers: List[ReplanningTrigger] = self._initialize_default_triggers()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_default_triggers(self) -> List[ReplanningTrigger]:
        """Initialize default replanning triggers."""
        return [
            ReplanningTrigger(
                trigger_type=ReplanningTriggerType.STEP_FAILURE,
                condition="step.status == 'failed' and step.retry_count >= step.max_retries",
                description="Step failed after maximum retries",
                priority=1
            ),
            ReplanningTrigger(
                trigger_type=ReplanningTriggerType.QUALITY_THRESHOLD,
                condition="result.quality_score < step.quality_threshold",
                description="Result quality below threshold",
                priority=2
            ),
            ReplanningTrigger(
                trigger_type=ReplanningTriggerType.AGENT_UNAVAILABLE,
                condition="assigned_agent.status == 'unavailable'",
                description="Assigned agent became unavailable",
                priority=1
            ),
            ReplanningTrigger(
                trigger_type=ReplanningTriggerType.TIMEOUT,
                condition="execution_time > step.timeout_seconds",
                description="Step execution timeout exceeded",
                priority=2
            ),
            ReplanningTrigger(
                trigger_type=ReplanningTriggerType.RESOURCE_CONSTRAINT,
                condition="available_resources < required_resources",
                description="Insufficient resources available",
                priority=3
            ),
        ]
    
    async def should_replan(
        self,
        execution_state: ExecutionState,
        current_plan: Plan,
        context: ExecutionContext
    ) -> Optional[ReplanningTriggerType]:
        """
        Determine if replanning is needed based on execution state and triggers.
        
        Args:
            execution_state: Current execution state
            current_plan: Current execution plan
            context: Execution context
            
        Returns:
            Trigger type if replanning needed, None otherwise
        """
        # Check if we've exceeded max replanning iterations
        if len(execution_state.replanning_history) >= self.max_replanning_iterations:
            self.logger.warning(f"Maximum replanning iterations ({self.max_replanning_iterations}) reached")
            return None
        
        # Evaluate each trigger
        for trigger in sorted(self.replanning_triggers, key=lambda t: t.priority):
            if not trigger.enabled:
                continue
            
            if await self._evaluate_trigger(trigger, execution_state, current_plan, context):
                self.logger.info(f"Replanning triggered by: {trigger.description}")
                return trigger.trigger_type
        
        return None
    
    async def _evaluate_trigger(
        self,
        trigger: ReplanningTrigger,
        execution_state: ExecutionState,
        current_plan: Plan,
        context: ExecutionContext
    ) -> bool:
        """Evaluate a specific replanning trigger."""
        try:
            # Simple condition evaluation based on trigger type
            if trigger.trigger_type == ReplanningTriggerType.STEP_FAILURE:
                return await self._check_step_failures(execution_state, current_plan)
            
            elif trigger.trigger_type == ReplanningTriggerType.QUALITY_THRESHOLD:
                return await self._check_quality_thresholds(execution_state, current_plan)
            
            elif trigger.trigger_type == ReplanningTriggerType.AGENT_UNAVAILABLE:
                return await self._check_agent_availability(execution_state, current_plan, context)
            
            elif trigger.trigger_type == ReplanningTriggerType.TIMEOUT:
                return await self._check_timeouts(execution_state, current_plan)
            
            elif trigger.trigger_type == ReplanningTriggerType.RESOURCE_CONSTRAINT:
                return await self._check_resource_constraints(execution_state, current_plan, context)
            
            elif trigger.trigger_type == ReplanningTriggerType.CONTEXT_CHANGE:
                return await self._check_context_changes(execution_state, current_plan, context)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating trigger {trigger.trigger_type}: {e}")
            return False
    
    async def _check_step_failures(self, execution_state: ExecutionState, current_plan: Plan) -> bool:
        """Check for step failures that require replanning."""
        for step in execution_state.completed_steps:
            if (step.status == ExecutionStatus.FAILED and 
                step.retry_count >= step.max_retries):
                return True
        return False
    
    async def _check_quality_thresholds(self, execution_state: ExecutionState, current_plan: Plan) -> bool:
        """Check if quality thresholds are not being met."""
        # Check intermediate results quality
        for result_data in execution_state.intermediate_results.values():
            if isinstance(result_data, dict) and "quality_score" in result_data:
                quality_score = result_data["quality_score"]
                threshold = result_data.get("quality_threshold", 0.8)
                if quality_score < threshold:
                    return True
        return False
    
    async def _check_agent_availability(
        self, 
        execution_state: ExecutionState, 
        current_plan: Plan, 
        context: ExecutionContext
    ) -> bool:
        """Check if assigned agents are still available."""
        # Check if any assigned agents are no longer available
        for step in current_plan.steps:
            if (step.agent_id and 
                step.status in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING] and
                step.agent_id not in context.available_agents):
                return True
        return False
    
    async def _check_timeouts(self, execution_state: ExecutionState, current_plan: Plan) -> bool:
        """Check for step timeouts."""
        current_time = datetime.utcnow()
        
        for step in current_plan.steps:
            if (step.status == ExecutionStatus.RUNNING and 
                step.started_at):
                # Check if step has timeout_seconds attribute (from planning models)
                timeout_seconds = getattr(step, 'timeout_seconds', None)
                if timeout_seconds:
                    elapsed = (current_time - step.started_at).total_seconds()
                    if elapsed > timeout_seconds:
                        return True
        return False
    
    async def _check_resource_constraints(
        self, 
        execution_state: ExecutionState, 
        current_plan: Plan, 
        context: ExecutionContext
    ) -> bool:
        """Check for resource constraint violations."""
        # Simple check based on resource limits in context
        if context.resource_limits:
            # Check if we're approaching resource limits
            current_usage = execution_state.execution_metrics.resource_utilization
            for resource, limit in context.resource_limits.items():
                if current_usage.get(resource, 0) > limit * 0.9:  # 90% threshold
                    return True
        return False
    
    async def _check_context_changes(
        self, 
        execution_state: ExecutionState, 
        current_plan: Plan, 
        context: ExecutionContext
    ) -> bool:
        """Check for significant context changes."""
        # Check if available agents have changed significantly
        plan_agents = {step.agent_id for step in current_plan.steps if step.agent_id}
        available_agents = set(context.available_agents)
        
        # If more than 50% of assigned agents are unavailable, trigger replanning
        if plan_agents:
            unavailable_ratio = len(plan_agents - available_agents) / len(plan_agents)
            if unavailable_ratio > 0.5:
                return True
        
        return False
    
    async def generate_revised_plan(
        self,
        current_plan: Plan,
        execution_state: ExecutionState,
        context: ExecutionContext,
        trigger_type: ReplanningTriggerType
    ) -> Plan:
        """
        Generate a revised execution plan based on current state and trigger.
        
        Args:
            current_plan: Current execution plan
            execution_state: Current execution state
            context: Execution context
            trigger_type: Type of trigger that caused replanning
            
        Returns:
            Revised execution plan
        """
        self.logger.info(f"Generating revised plan due to {trigger_type}")
        
        # Preserve intermediate results based on strategy
        preserved_results = await self._preserve_intermediate_results(
            execution_state, current_plan
        )
        
        # Update context based on current state
        updated_context = await self._update_context_for_replanning(
            context, execution_state, trigger_type
        )
        
        # Generate new plan with preserved results
        revised_plan = await self.planner.generate_plan(
            query=current_plan.metadata.get("query", "Revised execution"),
            context=updated_context,
            preserve_results=preserved_results
        )
        
        # Set plan relationships
        revised_plan.parent_plan_id = current_plan.plan_id
        revised_plan.version = current_plan.version + 1
        revised_plan.replanning_context = {
            "trigger_type": trigger_type.value,
            "preserved_results_count": len(preserved_results),
            "original_plan_id": current_plan.plan_id,
            "replanning_iteration": len(execution_state.replanning_history) + 1,
        }
        
        self.logger.info(f"Generated revised plan {revised_plan.plan_id} (v{revised_plan.version})")
        return revised_plan
    
    async def _preserve_intermediate_results(
        self,
        execution_state: ExecutionState,
        current_plan: Plan
    ) -> List[IntermediateResult]:
        """Preserve intermediate results based on preservation strategy."""
        preserved_results = []
        
        # Convert execution state data to IntermediateResult objects
        for step in execution_state.completed_steps:
            if step.status == ExecutionStatus.COMPLETED:
                should_preserve = False
                
                if self.context_preservation_strategy == ContextPreservationStrategy.PRESERVE_ALL:
                    should_preserve = True
                elif self.context_preservation_strategy == ContextPreservationStrategy.PRESERVE_SUCCESSFUL:
                    should_preserve = step.status == ExecutionStatus.COMPLETED
                elif self.context_preservation_strategy == ContextPreservationStrategy.PRESERVE_CRITICAL:
                    # Preserve steps marked as critical or high-quality results
                    should_preserve = (
                        step.outputs.get("critical", False) or
                        step.outputs.get("quality_score", 0) > 0.9
                    )
                elif self.context_preservation_strategy == ContextPreservationStrategy.MINIMAL_PRESERVATION:
                    # Only preserve final outputs
                    should_preserve = step.outputs.get("final_output", False)
                
                if should_preserve:
                    result = IntermediateResult(
                        step_id=step.step_id,
                        result_type=step.name,
                        data=step.outputs,
                        quality_score=step.outputs.get("quality_score"),
                        metadata={"preserved_from": current_plan.plan_id},
                        created_at=step.completed_at or datetime.utcnow(),
                        preserved=True
                    )
                    preserved_results.append(result)
        
        self.logger.info(f"Preserved {len(preserved_results)} intermediate results")
        return preserved_results
    
    async def _update_context_for_replanning(
        self,
        original_context: ExecutionContext,
        execution_state: ExecutionState,
        trigger_type: ReplanningTriggerType
    ) -> ExecutionContext:
        """Update execution context for replanning."""
        # Create updated context with current state
        updated_context = ExecutionContext(
            context_id=original_context.context_id,
            user_query=original_context.user_query,
            available_agents=original_context.available_agents.copy(),
            agent_capabilities=original_context.agent_capabilities.copy(),
            domain_constraints=original_context.domain_constraints.copy(),
            quality_requirements=original_context.quality_requirements.copy(),
            resource_limits=original_context.resource_limits.copy(),
            timeout_seconds=original_context.timeout_seconds,
            metadata=original_context.metadata.copy()
        )
        
        # Adjust context based on trigger type
        if trigger_type == ReplanningTriggerType.AGENT_UNAVAILABLE:
            # Remove unavailable agents
            failed_agents = {
                step.agent_id for step in execution_state.completed_steps
                if step.status == ExecutionStatus.FAILED and step.agent_id
            }
            updated_context.available_agents = [
                agent_id for agent_id in updated_context.available_agents
                if agent_id not in failed_agents
            ]
        
        elif trigger_type == ReplanningTriggerType.QUALITY_THRESHOLD:
            # Increase quality requirements
            for key in updated_context.quality_requirements:
                updated_context.quality_requirements[key] = min(
                    updated_context.quality_requirements[key] + 0.1, 1.0
                )
        
        elif trigger_type == ReplanningTriggerType.TIMEOUT:
            # Increase timeout allowances
            if updated_context.timeout_seconds:
                updated_context.timeout_seconds = int(updated_context.timeout_seconds * 1.5)
        
        # Add replanning metadata
        updated_context.metadata["replanning_trigger"] = trigger_type.value
        updated_context.metadata["replanning_iteration"] = len(execution_state.replanning_history) + 1
        
        return updated_context
    
    async def record_replanning_event(
        self,
        execution_state: ExecutionState,
        trigger_type: ReplanningTriggerType,
        old_plan: Plan,
        new_plan: Plan,
        context: Dict[str, Any]
    ) -> ReplanningEvent:
        """Record a replanning event in the execution state."""
        event = ReplanningEvent(
            trigger_type=trigger_type,
            trigger_description=f"Replanning triggered by {trigger_type.value}",
            context=context,
            replanning_successful=True,
            new_plan_id=new_plan.plan_id,
            preserved_results=[
                result.get("result_id") for result in new_plan.metadata.get("intermediate_results", [])
            ]
        )
        
        execution_state.replanning_history.append(event)
        self.logger.info(f"Recorded replanning event {event.event_id}")
        return event
    
    def add_custom_trigger(self, trigger: ReplanningTrigger) -> None:
        """Add a custom replanning trigger."""
        self.replanning_triggers.append(trigger)
        self.logger.info(f"Added custom trigger: {trigger.description}")
    
    def remove_trigger(self, trigger_type: ReplanningTriggerType) -> None:
        """Remove triggers of a specific type."""
        self.replanning_triggers = [
            t for t in self.replanning_triggers if t.trigger_type != trigger_type
        ]
        self.logger.info(f"Removed triggers of type: {trigger_type}")
    
    def set_max_iterations(self, max_iterations: int) -> None:
        """Set maximum replanning iterations."""
        self.max_replanning_iterations = max_iterations
        self.logger.info(f"Set max replanning iterations to: {max_iterations}")
    
    def set_preservation_strategy(self, strategy: ContextPreservationStrategy) -> None:
        """Set context preservation strategy."""
        self.context_preservation_strategy = strategy
        self.logger.info(f"Set preservation strategy to: {strategy}")