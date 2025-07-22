"""
Tests for the AdaptivePlanner.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.dataqa.orchestration.models import CapabilityType, ExecutionStatus
from src.dataqa.orchestration.planning.models import (
    ExecutionContext,
    ExecutionStep,
    IntermediateResult,
    Plan,
)
from src.dataqa.orchestration.planning.planner import AdaptivePlanner, PlanningStrategy


class TestPlanningStrategy:
    """Test PlanningStrategy base class."""
    
    def test_estimate_step_duration(self):
        """Test step duration estimation."""
        strategy = PlanningStrategy()
        
        # Test different capability types
        step_retrieval = ExecutionStep(
            name="Retrieval",
            description="Test",
            capability_required=CapabilityType.DATA_RETRIEVAL
        )
        step_analysis = ExecutionStep(
            name="Analysis",
            description="Test",
            capability_required=CapabilityType.DATA_ANALYSIS
        )
        step_approval = ExecutionStep(
            name="Approval",
            description="Test",
            capability_required=CapabilityType.APPROVAL
        )
        
        context = ExecutionContext(user_query="test")
        
        assert strategy.estimate_step_duration(step_retrieval, context) == 2
        assert strategy.estimate_step_duration(step_analysis, context) == 5
        assert strategy.estimate_step_duration(step_approval, context) == 10
    
    def test_validate_dependencies_valid(self):
        """Test dependency validation with valid DAG."""
        strategy = PlanningStrategy()
        
        steps = [
            ExecutionStep(name="Step 1", description="First"),
            ExecutionStep(name="Step 2", description="Second"),
            ExecutionStep(name="Step 3", description="Third")
        ]
        
        # Valid dependencies: 1 -> 2 -> 3
        dependencies = {
            steps[0].step_id: [],
            steps[1].step_id: [steps[0].step_id],
            steps[2].step_id: [steps[1].step_id]
        }
        
        assert strategy.validate_dependencies(steps, dependencies) is True
    
    def test_validate_dependencies_cycle(self):
        """Test dependency validation with cycle."""
        strategy = PlanningStrategy()
        
        steps = [
            ExecutionStep(name="Step 1", description="First"),
            ExecutionStep(name="Step 2", description="Second")
        ]
        
        # Circular dependencies: 1 -> 2 -> 1
        dependencies = {
            steps[0].step_id: [steps[1].step_id],
            steps[1].step_id: [steps[0].step_id]
        }
        
        assert strategy.validate_dependencies(steps, dependencies) is False
    
    def test_validate_dependencies_invalid_reference(self):
        """Test dependency validation with invalid step reference."""
        strategy = PlanningStrategy()
        
        steps = [
            ExecutionStep(name="Step 1", description="First")
        ]
        
        # Invalid dependency reference
        dependencies = {
            steps[0].step_id: ["non-existent-step"]
        }
        
        assert strategy.validate_dependencies(steps, dependencies) is False


class TestAdaptivePlanner:
    """Test AdaptivePlanner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.planner = AdaptivePlanner()
    
    @pytest.mark.asyncio
    async def test_generate_plan_basic(self):
        """Test basic plan generation."""
        context = ExecutionContext(
            user_query="Analyze sales data",
            available_agents=["agent-1"],
            agent_capabilities={
                "agent-1": [CapabilityType.DATA_ANALYSIS, CapabilityType.DATA_RETRIEVAL]
            }
        )
        
        plan = await self.planner.generate_plan("Analyze sales data", context)
        
        assert isinstance(plan, Plan)
        assert plan.name.startswith("Plan for:")
        assert "Analyze sales data" in plan.description
        assert len(plan.steps) > 0
        assert plan.estimated_duration_minutes > 0
        assert plan.version == 1
    
    @pytest.mark.asyncio
    async def test_generate_plan_with_preserved_results(self):
        """Test plan generation with preserved results."""
        context = ExecutionContext(
            user_query="Continue analysis",
            available_agents=["agent-1"],
            agent_capabilities={"agent-1": [CapabilityType.DATA_ANALYSIS]}
        )
        
        # Create preserved results
        preserved_results = [
            IntermediateResult(
                step_id="old-step-1",
                result_type="data_retrieval",
                data={"data": "preserved"},
                preserved=True
            )
        ]
        
        plan = await self.planner.generate_plan(
            "Continue analysis", 
            context, 
            preserve_results=preserved_results
        )
        
        # Should include preserved results as completed steps
        preserved_steps = [s for s in plan.steps if s.status == ExecutionStatus.COMPLETED]
        assert len(preserved_steps) > 0
        assert plan.metadata["preserved_results"] == 1
    
    @pytest.mark.asyncio
    async def test_analyze_query_requirements(self):
        """Test query analysis for capability requirements."""
        context = ExecutionContext(user_query="test")
        
        # Test different query types
        queries_and_expected = [
            ("Get sales data", [CapabilityType.DATA_RETRIEVAL]),
            ("Analyze customer trends", [CapabilityType.DATA_ANALYSIS]),  # Single capability, no coordination
            ("Create a chart", [CapabilityType.VISUALIZATION]),
            ("Generate code for analysis", [CapabilityType.CODE_GENERATION]),
            ("Plot sales trends and analyze", [
                CapabilityType.DATA_ANALYSIS, 
                CapabilityType.VISUALIZATION, 
                CapabilityType.COORDINATION
            ]),
        ]
        
        for query, expected_caps in queries_and_expected:
            capabilities = await self.planner._analyze_query_requirements(query, context)
            for cap in expected_caps:
                assert cap in capabilities
    
    @pytest.mark.asyncio
    async def test_find_suitable_agent(self):
        """Test finding suitable agents for capabilities."""
        context = ExecutionContext(
            user_query="test",
            available_agents=["agent-1", "agent-2", "agent-3"],
            agent_capabilities={
                "agent-1": [CapabilityType.DATA_RETRIEVAL],
                "agent-2": [CapabilityType.DATA_ANALYSIS, CapabilityType.VISUALIZATION],
                "agent-3": [CapabilityType.CODE_GENERATION]
            }
        )
        
        # Test finding agents for different capabilities
        agent_retrieval = await self.planner._find_suitable_agent(
            CapabilityType.DATA_RETRIEVAL, context
        )
        agent_analysis = await self.planner._find_suitable_agent(
            CapabilityType.DATA_ANALYSIS, context
        )
        agent_missing = await self.planner._find_suitable_agent(
            CapabilityType.APPROVAL, context
        )
        
        assert agent_retrieval == "agent-1"
        assert agent_analysis == "agent-2"
        assert agent_missing is None
    
    @pytest.mark.asyncio
    async def test_generate_success_criteria(self):
        """Test success criteria generation."""
        criteria_retrieval = await self.planner._generate_success_criteria(
            CapabilityType.DATA_RETRIEVAL
        )
        criteria_analysis = await self.planner._generate_success_criteria(
            CapabilityType.DATA_ANALYSIS
        )
        
        assert isinstance(criteria_retrieval, list)
        assert len(criteria_retrieval) > 0
        assert "Data successfully retrieved" in criteria_retrieval
        
        assert isinstance(criteria_analysis, list)
        assert len(criteria_analysis) > 0
        assert "Analysis completed without errors" in criteria_analysis
    
    @pytest.mark.asyncio
    async def test_build_dependency_graph(self):
        """Test dependency graph building."""
        steps = [
            ExecutionStep(name="Step 1", description="First"),
            ExecutionStep(name="Step 2", description="Second"),
            ExecutionStep(name="Step 3", description="Third")
        ]
        context = ExecutionContext(user_query="test")
        
        dependencies = await self.planner._build_dependency_graph(steps, context)
        
        # Should create sequential dependencies
        assert dependencies[steps[0].step_id] == []
        assert dependencies[steps[1].step_id] == [steps[0].step_id]
        assert dependencies[steps[2].step_id] == [steps[1].step_id]
    
    @pytest.mark.asyncio
    async def test_update_plan_with_results(self):
        """Test updating plan with execution results."""
        # Create initial plan
        steps = [
            ExecutionStep(name="Step 1", description="First"),
            ExecutionStep(name="Step 2", description="Second")
        ]
        plan = Plan(
            name="Test Plan",
            description="Test",
            steps=steps
        )
        
        # Create completed steps
        completed_steps = [
            ExecutionStep(
                step_id=steps[0].step_id,
                name="Step 1",
                description="First",
                status=ExecutionStatus.COMPLETED,
                outputs={"result": "success"}
            )
        ]
        
        # Create intermediate results
        intermediate_results = [
            IntermediateResult(
                step_id=steps[0].step_id,
                result_type="output",
                data={"key": "value"},
                quality_score=0.9
            )
        ]
        
        # Update plan
        updated_plan = await self.planner.update_plan_with_results(
            plan, completed_steps, intermediate_results
        )
        
        # Verify updates
        assert updated_plan.steps[0].status == ExecutionStatus.COMPLETED
        assert updated_plan.steps[0].outputs == {"result": "success"}
        assert "intermediate_results" in updated_plan.metadata
        assert len(updated_plan.metadata["intermediate_results"]) == 1
    
    @pytest.mark.asyncio
    async def test_generate_plan_invalid_dependencies(self):
        """Test plan generation with invalid dependencies."""
        # Mock strategy to return invalid dependencies
        mock_strategy = MagicMock()
        mock_strategy.validate_dependencies.return_value = False
        mock_strategy.estimate_step_duration.return_value = 5
        
        planner = AdaptivePlanner(strategy=mock_strategy)
        context = ExecutionContext(
            user_query="test",
            available_agents=["agent-1"],
            agent_capabilities={"agent-1": [CapabilityType.DATA_ANALYSIS]}
        )
        
        with pytest.raises(ValueError, match="invalid dependencies"):
            await planner.generate_plan("test query", context)
    
    @pytest.mark.asyncio
    async def test_generate_execution_steps_no_agents(self):
        """Test step generation when no suitable agents available."""
        context = ExecutionContext(
            user_query="Analyze data",
            available_agents=[],
            agent_capabilities={}
        )
        
        required_capabilities = [CapabilityType.DATA_ANALYSIS]
        steps = await self.planner._generate_execution_steps(
            required_capabilities, context
        )
        
        # Should still generate steps but without agent assignment
        assert len(steps) > 0
        assert steps[0].agent_id is None
        assert steps[0].capability_required == CapabilityType.DATA_ANALYSIS
    
    @pytest.mark.asyncio
    async def test_generate_plan_complex_query(self):
        """Test plan generation for complex multi-capability query."""
        context = ExecutionContext(
            user_query="Retrieve sales data, analyze trends, and create visualization",
            available_agents=["retrieval-agent", "analysis-agent", "viz-agent"],
            agent_capabilities={
                "retrieval-agent": [CapabilityType.DATA_RETRIEVAL],
                "analysis-agent": [CapabilityType.DATA_ANALYSIS],
                "viz-agent": [CapabilityType.VISUALIZATION]
            },
            quality_requirements={"data_analysis": 0.9, "visualization": 0.8},
            timeout_seconds=1800
        )
        
        plan = await self.planner.generate_plan(
            "Retrieve sales data, analyze trends, and create visualization",
            context
        )
        
        # Should generate multiple steps for different capabilities
        assert len(plan.steps) >= 3  # At least retrieval, analysis, visualization
        
        # Should have coordination step for multiple capabilities
        capability_types = [step.capability_required for step in plan.steps]
        assert CapabilityType.COORDINATION in capability_types
        
        # Should assign appropriate agents
        agent_assignments = [step.agent_id for step in plan.steps if step.agent_id]
        assert "retrieval-agent" in agent_assignments
        assert "analysis-agent" in agent_assignments
        assert "viz-agent" in agent_assignments
        
        # Should set quality thresholds
        for step in plan.steps:
            if step.capability_required == CapabilityType.DATA_ANALYSIS:
                assert step.quality_threshold == 0.9
            elif step.capability_required == CapabilityType.VISUALIZATION:
                assert step.quality_threshold == 0.8
        
        # Should set timeouts
        for step in plan.steps:
            assert step.timeout_seconds == 1800