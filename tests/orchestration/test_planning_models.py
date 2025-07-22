"""
Tests for orchestration planning models.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from src.dataqa.orchestration.models import CapabilityType, ExecutionStatus
from src.dataqa.orchestration.planning.models import (
    ContextPreservationStrategy,
    ExecutionContext,
    ExecutionStep,
    IntermediateResult,
    Plan,
    ReplanningEvent,
    ReplanningTrigger,
    ReplanningTriggerType,
)


class TestExecutionStep:
    """Test ExecutionStep model."""
    
    def test_execution_step_creation(self):
        """Test creating an execution step."""
        step = ExecutionStep(
            name="Test Step",
            description="A test execution step",
            capability_required=CapabilityType.DATA_ANALYSIS
        )
        
        assert step.name == "Test Step"
        assert step.description == "A test execution step"
        assert step.capability_required == CapabilityType.DATA_ANALYSIS
        assert step.status == ExecutionStatus.PENDING
        assert step.retry_count == 0
        assert step.max_retries == 3
        assert len(step.step_id) > 0
    
    def test_execution_step_with_custom_values(self):
        """Test creating execution step with custom values."""
        step = ExecutionStep(
            name="Custom Step",
            description="Custom description",
            agent_id="agent-123",
            capability_required=CapabilityType.VISUALIZATION,
            max_retries=5,
            timeout_seconds=600,
            quality_threshold=0.9
        )
        
        assert step.agent_id == "agent-123"
        assert step.max_retries == 5
        assert step.timeout_seconds == 600
        assert step.quality_threshold == 0.9
    
    def test_execution_step_success_criteria(self):
        """Test execution step with success criteria."""
        criteria = ["Data retrieved successfully", "No errors occurred"]
        step = ExecutionStep(
            name="Test Step",
            description="Test",
            success_criteria=criteria
        )
        
        assert step.success_criteria == criteria


class TestIntermediateResult:
    """Test IntermediateResult model."""
    
    def test_intermediate_result_creation(self):
        """Test creating an intermediate result."""
        step_id = str(uuid4())
        result = IntermediateResult(
            step_id=step_id,
            result_type="analysis_output",
            data={"key": "value"},
            quality_score=0.85
        )
        
        assert result.step_id == step_id
        assert result.result_type == "analysis_output"
        assert result.data == {"key": "value"}
        assert result.quality_score == 0.85
        assert result.preserved is True
        assert len(result.result_id) > 0
    
    def test_intermediate_result_with_metadata(self):
        """Test intermediate result with metadata."""
        metadata = {"source": "test", "version": "1.0"}
        result = IntermediateResult(
            step_id="step-123",
            result_type="test_result",
            data="test data",
            metadata=metadata
        )
        
        assert result.metadata == metadata


class TestPlan:
    """Test Plan model."""
    
    def test_plan_creation(self):
        """Test creating a plan."""
        steps = [
            ExecutionStep(name="Step 1", description="First step"),
            ExecutionStep(name="Step 2", description="Second step")
        ]
        
        plan = Plan(
            name="Test Plan",
            description="A test plan",
            steps=steps,
            estimated_duration_minutes=30
        )
        
        assert plan.name == "Test Plan"
        assert plan.description == "A test plan"
        assert len(plan.steps) == 2
        assert plan.estimated_duration_minutes == 30
        assert plan.version == 1
        assert len(plan.plan_id) > 0
    
    def test_plan_with_dependencies(self):
        """Test plan with step dependencies."""
        steps = [
            ExecutionStep(name="Step 1", description="First step"),
            ExecutionStep(name="Step 2", description="Second step")
        ]
        dependencies = {
            steps[1].step_id: [steps[0].step_id]
        }
        
        plan = Plan(
            name="Test Plan",
            description="Plan with dependencies",
            steps=steps,
            dependencies=dependencies
        )
        
        assert plan.dependencies == dependencies
    
    def test_plan_replanning_context(self):
        """Test plan with replanning context."""
        parent_plan_id = str(uuid4())
        replanning_context = {"trigger": "step_failure", "iteration": 2}
        
        plan = Plan(
            name="Replanned Plan",
            description="A replanned execution",
            parent_plan_id=parent_plan_id,
            replanning_context=replanning_context,
            version=2
        )
        
        assert plan.parent_plan_id == parent_plan_id
        assert plan.replanning_context == replanning_context
        assert plan.version == 2


class TestReplanningTrigger:
    """Test ReplanningTrigger model."""
    
    def test_replanning_trigger_creation(self):
        """Test creating a replanning trigger."""
        trigger = ReplanningTrigger(
            trigger_type=ReplanningTriggerType.STEP_FAILURE,
            condition="step.status == 'failed'",
            description="Step failed",
            priority=1
        )
        
        assert trigger.trigger_type == ReplanningTriggerType.STEP_FAILURE
        assert trigger.condition == "step.status == 'failed'"
        assert trigger.description == "Step failed"
        assert trigger.priority == 1
        assert trigger.enabled is True
    
    def test_replanning_trigger_types(self):
        """Test all replanning trigger types."""
        trigger_types = [
            ReplanningTriggerType.STEP_FAILURE,
            ReplanningTriggerType.QUALITY_THRESHOLD,
            ReplanningTriggerType.RESOURCE_CONSTRAINT,
            ReplanningTriggerType.AGENT_UNAVAILABLE,
            ReplanningTriggerType.CONTEXT_CHANGE,
            ReplanningTriggerType.USER_REQUEST,
            ReplanningTriggerType.TIMEOUT,
            ReplanningTriggerType.DEPENDENCY_FAILURE,
        ]
        
        for trigger_type in trigger_types:
            trigger = ReplanningTrigger(
                trigger_type=trigger_type,
                condition="test condition",
                description=f"Test {trigger_type.value}"
            )
            assert trigger.trigger_type == trigger_type


class TestReplanningEvent:
    """Test ReplanningEvent model."""
    
    def test_replanning_event_creation(self):
        """Test creating a replanning event."""
        event = ReplanningEvent(
            trigger_type=ReplanningTriggerType.QUALITY_THRESHOLD,
            trigger_description="Quality below threshold",
            context={"threshold": 0.8, "actual": 0.6},
            replanning_successful=True,
            new_plan_id="plan-456"
        )
        
        assert event.trigger_type == ReplanningTriggerType.QUALITY_THRESHOLD
        assert event.trigger_description == "Quality below threshold"
        assert event.context == {"threshold": 0.8, "actual": 0.6}
        assert event.replanning_successful is True
        assert event.new_plan_id == "plan-456"
        assert len(event.event_id) > 0
    
    def test_replanning_event_with_preserved_results(self):
        """Test replanning event with preserved results."""
        preserved_results = ["result-1", "result-2", "result-3"]
        event = ReplanningEvent(
            trigger_type=ReplanningTriggerType.AGENT_UNAVAILABLE,
            trigger_description="Agent became unavailable",
            preserved_results=preserved_results
        )
        
        assert event.preserved_results == preserved_results


class TestExecutionContext:
    """Test ExecutionContext model."""
    
    def test_execution_context_creation(self):
        """Test creating an execution context."""
        context = ExecutionContext(
            user_query="Analyze sales data",
            available_agents=["agent-1", "agent-2"],
            agent_capabilities={
                "agent-1": [CapabilityType.DATA_RETRIEVAL, CapabilityType.DATA_ANALYSIS],
                "agent-2": [CapabilityType.VISUALIZATION]
            },
            quality_requirements={"analysis": 0.9},
            timeout_seconds=600
        )
        
        assert context.user_query == "Analyze sales data"
        assert context.available_agents == ["agent-1", "agent-2"]
        assert len(context.agent_capabilities) == 2
        assert context.quality_requirements == {"analysis": 0.9}
        assert context.timeout_seconds == 600
        assert len(context.context_id) > 0
    
    def test_execution_context_with_constraints(self):
        """Test execution context with domain constraints."""
        domain_constraints = {
            "data_sources": ["database_a", "database_b"],
            "compliance_rules": ["gdpr", "hipaa"]
        }
        resource_limits = {
            "cpu": 80,
            "memory": 4096,
            "storage": 1024
        }
        
        context = ExecutionContext(
            user_query="Test query",
            domain_constraints=domain_constraints,
            resource_limits=resource_limits
        )
        
        assert context.domain_constraints == domain_constraints
        assert context.resource_limits == resource_limits


class TestContextPreservationStrategy:
    """Test ContextPreservationStrategy enum."""
    
    def test_preservation_strategies(self):
        """Test all preservation strategies."""
        strategies = [
            ContextPreservationStrategy.PRESERVE_ALL,
            ContextPreservationStrategy.PRESERVE_SUCCESSFUL,
            ContextPreservationStrategy.PRESERVE_CRITICAL,
            ContextPreservationStrategy.MINIMAL_PRESERVATION,
        ]
        
        for strategy in strategies:
            assert isinstance(strategy.value, str)
            assert len(strategy.value) > 0


@pytest.mark.asyncio
class TestModelIntegration:
    """Test integration between planning models."""
    
    async def test_plan_execution_flow(self):
        """Test complete plan execution flow with models."""
        # Create execution context
        context = ExecutionContext(
            user_query="Test execution flow",
            available_agents=["agent-1"],
            agent_capabilities={"agent-1": [CapabilityType.DATA_ANALYSIS]}
        )
        
        # Create execution steps
        steps = [
            ExecutionStep(
                name="Analysis Step",
                description="Perform data analysis",
                capability_required=CapabilityType.DATA_ANALYSIS,
                agent_id="agent-1"
            )
        ]
        
        # Create plan
        plan = Plan(
            name="Test Execution Plan",
            description="Plan for testing execution flow",
            steps=steps
        )
        
        # Create intermediate result
        result = IntermediateResult(
            step_id=steps[0].step_id,
            result_type="analysis_output",
            data={"result": "success"},
            quality_score=0.95
        )
        
        # Create replanning event
        event = ReplanningEvent(
            trigger_type=ReplanningTriggerType.QUALITY_THRESHOLD,
            trigger_description="Quality check passed",
            new_plan_id=plan.plan_id,
            replanning_successful=True
        )
        
        # Verify all components work together
        assert plan.steps[0].step_id == result.step_id
        assert event.new_plan_id == plan.plan_id
        assert result.quality_score > context.quality_requirements.get("analysis", 0.8)