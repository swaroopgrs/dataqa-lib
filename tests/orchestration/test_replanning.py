"""
Tests for the ReplanningEngine.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.dataqa.orchestration.models import ExecutionState, ExecutionStatus, ExecutionMetrics
from src.dataqa.orchestration.models import ExecutionStep as ModelsExecutionStep
from src.dataqa.orchestration.planning.models import (
    ContextPreservationStrategy,
    ExecutionContext,
    ExecutionStep as PlanningExecutionStep,
    IntermediateResult,
    Plan,
    ReplanningEvent,
    ReplanningTrigger,
    ReplanningTriggerType,
)
from src.dataqa.orchestration.planning.planner import AdaptivePlanner
from src.dataqa.orchestration.planning.replanning import ReplanningEngine


class TestReplanningEngine:
    """Test ReplanningEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_planner = AsyncMock(spec=AdaptivePlanner)
        self.engine = ReplanningEngine(
            max_replanning_iterations=3,
            context_preservation_strategy=ContextPreservationStrategy.PRESERVE_SUCCESSFUL,
            planner=self.mock_planner
        )
    
    def test_initialization(self):
        """Test replanning engine initialization."""
        assert self.engine.max_replanning_iterations == 3
        assert self.engine.context_preservation_strategy == ContextPreservationStrategy.PRESERVE_SUCCESSFUL
        assert len(self.engine.replanning_triggers) > 0
        
        # Check default triggers are present
        trigger_types = [t.trigger_type for t in self.engine.replanning_triggers]
        assert ReplanningTriggerType.STEP_FAILURE in trigger_types
        assert ReplanningTriggerType.QUALITY_THRESHOLD in trigger_types
        assert ReplanningTriggerType.AGENT_UNAVAILABLE in trigger_types
    
    @pytest.mark.asyncio
    async def test_should_replan_no_triggers(self):
        """Test should_replan when no triggers are met."""
        execution_state = ExecutionState()
        plan = Plan(name="Test", description="Test plan")
        context = ExecutionContext(user_query="test")
        
        result = await self.engine.should_replan(execution_state, plan, context)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_should_replan_max_iterations_reached(self):
        """Test should_replan when max iterations reached."""
        execution_state = ExecutionState()
        # Add replanning history to exceed max iterations
        for i in range(4):
            event = ReplanningEvent(
                trigger_type=ReplanningTriggerType.STEP_FAILURE,
                trigger_description=f"Replanning {i}"
            )
            execution_state.replanning_history.append(event)
        
        plan = Plan(name="Test", description="Test plan")
        context = ExecutionContext(user_query="test")
        
        result = await self.engine.should_replan(execution_state, plan, context)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_should_replan_step_failure(self):
        """Test should_replan with step failure trigger."""
        # Create execution state with failed step
        failed_step = ModelsExecutionStep(
            name="Failed Step",
            description="Test",
            status=ExecutionStatus.FAILED,
            retry_count=3,
            max_retries=3
        )
        execution_state = ExecutionState(completed_steps=[failed_step])
        
        plan = Plan(name="Test", description="Test plan")
        context = ExecutionContext(user_query="test")
        
        result = await self.engine.should_replan(execution_state, plan, context)
        assert result == ReplanningTriggerType.STEP_FAILURE
    
    @pytest.mark.asyncio
    async def test_should_replan_quality_threshold(self):
        """Test should_replan with quality threshold trigger."""
        # Create execution state with low quality result
        execution_state = ExecutionState()
        execution_state.intermediate_results["result-1"] = {
            "quality_score": 0.5,
            "quality_threshold": 0.8
        }
        
        plan = Plan(name="Test", description="Test plan")
        context = ExecutionContext(user_query="test")
        
        result = await self.engine.should_replan(execution_state, plan, context)
        assert result == ReplanningTriggerType.QUALITY_THRESHOLD
    
    @pytest.mark.asyncio
    async def test_should_replan_agent_unavailable(self):
        """Test should_replan with agent unavailable trigger."""
        # Create plan with assigned agent
        step = PlanningExecutionStep(
            name="Test Step",
            description="Test",
            agent_id="agent-1",
            status=ExecutionStatus.PENDING
        )
        plan = Plan(name="Test", description="Test plan", steps=[step])
        
        # Context without the assigned agent
        context = ExecutionContext(
            user_query="test",
            available_agents=["agent-2", "agent-3"]
        )
        execution_state = ExecutionState()
        
        result = await self.engine.should_replan(execution_state, plan, context)
        assert result == ReplanningTriggerType.AGENT_UNAVAILABLE
    
    @pytest.mark.asyncio
    async def test_should_replan_timeout(self):
        """Test should_replan with timeout trigger."""
        # Create step that started long ago and is still running
        step = PlanningExecutionStep(
            name="Long Running Step",
            description="Test",
            status=ExecutionStatus.RUNNING,
            started_at=datetime.utcnow() - timedelta(seconds=600),
            timeout_seconds=300
        )
        plan = Plan(name="Test", description="Test plan", steps=[step])
        execution_state = ExecutionState()
        context = ExecutionContext(user_query="test")
        
        result = await self.engine.should_replan(execution_state, plan, context)
        assert result == ReplanningTriggerType.TIMEOUT
    
    @pytest.mark.asyncio
    async def test_should_replan_resource_constraint(self):
        """Test should_replan with resource constraint trigger."""
        execution_state = ExecutionState()
        execution_state.execution_metrics.resource_utilization = {"cpu": 95}
        
        plan = Plan(name="Test", description="Test plan")
        context = ExecutionContext(
            user_query="test",
            resource_limits={"cpu": 100}
        )
        
        result = await self.engine.should_replan(execution_state, plan, context)
        assert result == ReplanningTriggerType.RESOURCE_CONSTRAINT
    
    @pytest.mark.asyncio
    async def test_should_replan_context_change(self):
        """Test should_replan with context change trigger."""
        # Create plan with agents that are no longer available
        steps = [
            PlanningExecutionStep(name="Step 1", description="Test", agent_id="agent-1"),
            PlanningExecutionStep(name="Step 2", description="Test", agent_id="agent-2")
        ]
        plan = Plan(name="Test", description="Test plan", steps=steps)
        
        # Context with only one of the original agents
        context = ExecutionContext(
            user_query="test",
            available_agents=["agent-1", "agent-3"]
        )
        execution_state = ExecutionState()
        
        result = await self.engine.should_replan(execution_state, plan, context)
        # Agent unavailable trigger has higher priority than context change
        assert result == ReplanningTriggerType.AGENT_UNAVAILABLE
    
    @pytest.mark.asyncio
    async def test_generate_revised_plan(self):
        """Test generating a revised plan."""
        # Set up mock planner
        revised_plan = Plan(
            name="Revised Plan",
            description="Revised execution plan",
            version=2
        )
        self.mock_planner.generate_plan.return_value = revised_plan
        
        # Create original plan and execution state
        original_plan = Plan(
            name="Original Plan",
            description="Original execution plan",
            metadata={"query": "test query"}
        )
        execution_state = ExecutionState()
        context = ExecutionContext(user_query="test query")
        
        result = await self.engine.generate_revised_plan(
            original_plan,
            execution_state,
            context,
            ReplanningTriggerType.STEP_FAILURE
        )
        
        # Verify revised plan properties
        assert result.parent_plan_id == original_plan.plan_id
        assert result.version == 2
        assert "trigger_type" in result.replanning_context
        assert result.replanning_context["trigger_type"] == "step_failure"
        
        # Verify planner was called
        self.mock_planner.generate_plan.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_preserve_intermediate_results_preserve_all(self):
        """Test preserving intermediate results with PRESERVE_ALL strategy."""
        self.engine.context_preservation_strategy = ContextPreservationStrategy.PRESERVE_ALL
        
        # Create execution state with completed steps
        completed_steps = [
            ModelsExecutionStep(
                name="Step 1",
                description="Test",
                status=ExecutionStatus.COMPLETED,
                completed_at=datetime.utcnow(),
                outputs={"result": "success"}
            ),
            ModelsExecutionStep(
                name="Step 2",
                description="Test",
                status=ExecutionStatus.FAILED,
                outputs={"error": "failed"}
            )
        ]
        execution_state = ExecutionState(completed_steps=completed_steps)
        plan = Plan(name="Test", description="Test")
        
        preserved = await self.engine._preserve_intermediate_results(execution_state, plan)
        
        # Should preserve only completed steps
        assert len(preserved) == 1
        assert preserved[0].step_id == completed_steps[0].step_id
        assert preserved[0].data == {"result": "success"}
    
    @pytest.mark.asyncio
    async def test_preserve_intermediate_results_preserve_critical(self):
        """Test preserving intermediate results with PRESERVE_CRITICAL strategy."""
        self.engine.context_preservation_strategy = ContextPreservationStrategy.PRESERVE_CRITICAL
        
        # Create execution state with critical and non-critical steps
        completed_steps = [
            ModelsExecutionStep(
                name="Critical Step",
                description="Test",
                status=ExecutionStatus.COMPLETED,
                outputs={"result": "success", "critical": True}
            ),
            ModelsExecutionStep(
                name="High Quality Step",
                description="Test",
                status=ExecutionStatus.COMPLETED,
                outputs={"result": "success", "quality_score": 0.95}
            ),
            ModelsExecutionStep(
                name="Regular Step",
                description="Test",
                status=ExecutionStatus.COMPLETED,
                outputs={"result": "success"}
            )
        ]
        execution_state = ExecutionState(completed_steps=completed_steps)
        plan = Plan(name="Test", description="Test")
        
        preserved = await self.engine._preserve_intermediate_results(execution_state, plan)
        
        # Should preserve critical and high-quality steps
        assert len(preserved) == 2
        preserved_names = [r.result_type for r in preserved]
        assert "Critical Step" in preserved_names
        assert "High Quality Step" in preserved_names
    
    @pytest.mark.asyncio
    async def test_update_context_for_replanning_agent_unavailable(self):
        """Test updating context for replanning due to agent unavailable."""
        # Create execution state with failed agent
        failed_step = ModelsExecutionStep(
            name="Failed Step",
            description="Test",
            agent_id="failed-agent",
            status=ExecutionStatus.FAILED
        )
        execution_state = ExecutionState(completed_steps=[failed_step])
        
        original_context = ExecutionContext(
            user_query="test",
            available_agents=["failed-agent", "good-agent"]
        )
        
        updated_context = await self.engine._update_context_for_replanning(
            original_context,
            execution_state,
            ReplanningTriggerType.AGENT_UNAVAILABLE
        )
        
        # Should remove failed agent
        assert "failed-agent" not in updated_context.available_agents
        assert "good-agent" in updated_context.available_agents
        assert updated_context.metadata["replanning_trigger"] == "agent_unavailable"
    
    @pytest.mark.asyncio
    async def test_update_context_for_replanning_quality_threshold(self):
        """Test updating context for replanning due to quality threshold."""
        execution_state = ExecutionState()
        original_context = ExecutionContext(
            user_query="test",
            quality_requirements={"analysis": 0.8, "visualization": 0.7}
        )
        
        updated_context = await self.engine._update_context_for_replanning(
            original_context,
            execution_state,
            ReplanningTriggerType.QUALITY_THRESHOLD
        )
        
        # Should increase quality requirements
        assert updated_context.quality_requirements["analysis"] == 0.9
        assert abs(updated_context.quality_requirements["visualization"] - 0.8) < 0.001  # Handle floating point precision
    
    @pytest.mark.asyncio
    async def test_update_context_for_replanning_timeout(self):
        """Test updating context for replanning due to timeout."""
        execution_state = ExecutionState()
        original_context = ExecutionContext(
            user_query="test",
            timeout_seconds=300
        )
        
        updated_context = await self.engine._update_context_for_replanning(
            original_context,
            execution_state,
            ReplanningTriggerType.TIMEOUT
        )
        
        # Should increase timeout
        assert updated_context.timeout_seconds == 450  # 300 * 1.5
    
    @pytest.mark.asyncio
    async def test_record_replanning_event(self):
        """Test recording a replanning event."""
        execution_state = ExecutionState()
        old_plan = Plan(name="Old Plan", description="Old")
        new_plan = Plan(name="New Plan", description="New")
        context = {"reason": "test"}
        
        event = await self.engine.record_replanning_event(
            execution_state,
            ReplanningTriggerType.STEP_FAILURE,
            old_plan,
            new_plan,
            context
        )
        
        assert event.trigger_type == ReplanningTriggerType.STEP_FAILURE
        assert event.new_plan_id == new_plan.plan_id
        assert event.context == context
        assert event.replanning_successful is True
        assert len(execution_state.replanning_history) == 1
        assert execution_state.replanning_history[0] == event
    
    def test_add_custom_trigger(self):
        """Test adding a custom replanning trigger."""
        custom_trigger = ReplanningTrigger(
            trigger_type=ReplanningTriggerType.USER_REQUEST,
            condition="user.requested_replanning == true",
            description="User requested replanning"
        )
        
        initial_count = len(self.engine.replanning_triggers)
        self.engine.add_custom_trigger(custom_trigger)
        
        assert len(self.engine.replanning_triggers) == initial_count + 1
        assert custom_trigger in self.engine.replanning_triggers
    
    def test_remove_trigger(self):
        """Test removing triggers of a specific type."""
        # Add multiple triggers of the same type
        trigger1 = ReplanningTrigger(
            trigger_type=ReplanningTriggerType.USER_REQUEST,
            condition="condition1",
            description="Test 1"
        )
        trigger2 = ReplanningTrigger(
            trigger_type=ReplanningTriggerType.USER_REQUEST,
            condition="condition2",
            description="Test 2"
        )
        
        self.engine.add_custom_trigger(trigger1)
        self.engine.add_custom_trigger(trigger2)
        
        # Remove all USER_REQUEST triggers
        self.engine.remove_trigger(ReplanningTriggerType.USER_REQUEST)
        
        # Verify they're removed
        trigger_types = [t.trigger_type for t in self.engine.replanning_triggers]
        assert ReplanningTriggerType.USER_REQUEST not in trigger_types
    
    def test_set_max_iterations(self):
        """Test setting maximum replanning iterations."""
        self.engine.set_max_iterations(5)
        assert self.engine.max_replanning_iterations == 5
    
    def test_set_preservation_strategy(self):
        """Test setting context preservation strategy."""
        self.engine.set_preservation_strategy(ContextPreservationStrategy.PRESERVE_ALL)
        assert self.engine.context_preservation_strategy == ContextPreservationStrategy.PRESERVE_ALL
    
    @pytest.mark.asyncio
    async def test_check_step_failures_no_failures(self):
        """Test checking step failures when there are none."""
        execution_state = ExecutionState(
            completed_steps=[
                ModelsExecutionStep(
                    name="Success Step",
                    description="Test",
                    status=ExecutionStatus.COMPLETED
                )
            ]
        )
        plan = Plan(name="Test", description="Test")
        
        result = await self.engine._check_step_failures(execution_state, plan)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_step_failures_with_retries_remaining(self):
        """Test checking step failures when retries remain."""
        execution_state = ExecutionState(
            completed_steps=[
                ModelsExecutionStep(
                    name="Retry Step",
                    description="Test",
                    status=ExecutionStatus.FAILED,
                    retry_count=1,
                    max_retries=3
                )
            ]
        )
        plan = Plan(name="Test", description="Test")
        
        result = await self.engine._check_step_failures(execution_state, plan)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_evaluate_trigger_exception_handling(self):
        """Test trigger evaluation with exception handling."""
        # Create a trigger that will cause an exception
        bad_trigger = ReplanningTrigger(
            trigger_type=ReplanningTriggerType.STEP_FAILURE,
            condition="invalid condition",
            description="Bad trigger"
        )
        
        execution_state = ExecutionState()
        plan = Plan(name="Test", description="Test")
        context = ExecutionContext(user_query="test")
        
        # Should handle exception gracefully
        result = await self.engine._evaluate_trigger(bad_trigger, execution_state, plan, context)
        assert result is False