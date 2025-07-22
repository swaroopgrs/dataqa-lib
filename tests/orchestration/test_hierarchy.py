"""
Comprehensive tests for hierarchical agent management system.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.dataqa.orchestration.models import (
    AgentCapability,
    AgentConfiguration,
    AgentRole,
    CapabilityType,
    ExecutionStatus,
)
from src.dataqa.orchestration.agents import (
    BaseAgent,
    ManagerAgent,
    WorkerAgent,
    AgentHierarchy,
    Task,
    TaskResult,
    ExecutionContext,
    ProgressUpdate,
    AssistanceRequest,
    Escalation,
    Resolution,
    DelegationStrategy,
    CoordinationProtocol,
)


@pytest.fixture
def sample_capabilities():
    """Create sample capabilities for testing."""
    return {
        "data_analysis": AgentCapability(
            capability_type=CapabilityType.DATA_ANALYSIS,
            name="Data Analysis",
            description="Analyze data"
        ),
        "visualization": AgentCapability(
            capability_type=CapabilityType.VISUALIZATION,
            name="Visualization",
            description="Create visualizations"
        ),
        "coordination": AgentCapability(
            capability_type=CapabilityType.COORDINATION,
            name="Coordination",
            description="Coordinate tasks"
        ),
        "code_generation": AgentCapability(
            capability_type=CapabilityType.CODE_GENERATION,
            name="Code Generation",
            description="Generate code"
        )
    }


@pytest.fixture
def sample_agents(sample_capabilities):
    """Create sample agents for testing."""
    # Manager agent
    manager_config = AgentConfiguration(
        name="Test Manager",
        role=AgentRole.MANAGER,
        capabilities=[sample_capabilities["coordination"]],
        priority_level=1
    )
    manager = ManagerAgent(manager_config)
    
    # Data worker
    data_worker_config = AgentConfiguration(
        name="Data Worker",
        role=AgentRole.WORKER,
        capabilities=[sample_capabilities["data_analysis"]],
        specialization="analytics",
        priority_level=2
    )
    data_worker = WorkerAgent(data_worker_config)
    
    # Viz worker
    viz_worker_config = AgentConfiguration(
        name="Viz Worker",
        role=AgentRole.WORKER,
        capabilities=[sample_capabilities["visualization"]],
        specialization="visualization",
        priority_level=3
    )
    viz_worker = WorkerAgent(viz_worker_config)
    
    # Code worker
    code_worker_config = AgentConfiguration(
        name="Code Worker",
        role=AgentRole.WORKER,
        capabilities=[sample_capabilities["code_generation"]],
        specialization="development",
        priority_level=2
    )
    code_worker = WorkerAgent(code_worker_config)
    
    return {
        "manager": manager,
        "data_worker": data_worker,
        "viz_worker": viz_worker,
        "code_worker": code_worker
    }


@pytest.fixture
def hierarchy_with_agents(sample_agents):
    """Create hierarchy with sample agents."""
    hierarchy = AgentHierarchy()
    
    # Add manager as root
    hierarchy.add_agent(sample_agents["manager"])
    
    # Add workers under manager
    hierarchy.add_agent(sample_agents["data_worker"], parent_id=sample_agents["manager"].agent_id)
    hierarchy.add_agent(sample_agents["viz_worker"], parent_id=sample_agents["manager"].agent_id)
    hierarchy.add_agent(sample_agents["code_worker"], parent_id=sample_agents["manager"].agent_id)
    
    return hierarchy


class TestAgentHierarchy:
    """Test AgentHierarchy class functionality."""
    
    def test_hierarchy_creation(self):
        """Test basic hierarchy creation."""
        hierarchy = AgentHierarchy()
        assert len(hierarchy.agents) == 0
        assert len(hierarchy.root_agents) == 0
        assert hierarchy.get_hierarchy_depth() == 0
    
    def test_add_root_agent(self, sample_agents):
        """Test adding root agent to hierarchy."""
        hierarchy = AgentHierarchy()
        manager = sample_agents["manager"]
        
        hierarchy.add_agent(manager)
        
        assert len(hierarchy.agents) == 1
        assert len(hierarchy.root_agents) == 1
        assert manager.agent_id in hierarchy.root_agents
        assert hierarchy.get_hierarchy_depth() == 1
    
    def test_add_child_agent(self, sample_agents):
        """Test adding child agent to hierarchy."""
        hierarchy = AgentHierarchy()
        manager = sample_agents["manager"]
        worker = sample_agents["data_worker"]
        
        hierarchy.add_agent(manager)
        hierarchy.add_agent(worker, parent_id=manager.agent_id)
        
        assert len(hierarchy.agents) == 2
        assert len(hierarchy.root_agents) == 1
        assert hierarchy.get_hierarchy_depth() == 2
        
        # Test relationships
        children = hierarchy.get_children(manager.agent_id)
        assert len(children) == 1
        assert children[0].agent_id == worker.agent_id
        
        parent = hierarchy.get_parent(worker.agent_id)
        assert parent is not None
        assert parent.agent_id == manager.agent_id
    
    def test_add_duplicate_agent(self, sample_agents):
        """Test adding duplicate agent raises error."""
        hierarchy = AgentHierarchy()
        manager = sample_agents["manager"]
        
        hierarchy.add_agent(manager)
        
        with pytest.raises(ValueError, match="already exists"):
            hierarchy.add_agent(manager)
    
    def test_add_agent_with_nonexistent_parent(self, sample_agents):
        """Test adding agent with nonexistent parent raises error."""
        hierarchy = AgentHierarchy()
        worker = sample_agents["data_worker"]
        
        with pytest.raises(ValueError, match="not found"):
            hierarchy.add_agent(worker, parent_id="nonexistent")
    
    def test_remove_agent(self, hierarchy_with_agents, sample_agents):
        """Test removing agent from hierarchy."""
        hierarchy = hierarchy_with_agents
        worker = sample_agents["data_worker"]
        
        # Remove worker
        hierarchy.remove_agent(worker.agent_id)
        
        assert worker.agent_id not in hierarchy.agents
        assert len(hierarchy.agents) == 3  # Manager + 2 remaining workers
        
        # Check parent's children updated
        manager = sample_agents["manager"]
        children = hierarchy.get_children(manager.agent_id)
        assert len(children) == 2
        assert worker.agent_id not in [c.agent_id for c in children]
    
    def test_remove_agent_with_children(self, hierarchy_with_agents, sample_agents):
        """Test removing agent with children raises error."""
        hierarchy = hierarchy_with_agents
        manager = sample_agents["manager"]
        
        with pytest.raises(ValueError, match="with children"):
            hierarchy.remove_agent(manager.agent_id)
    
    def test_remove_nonexistent_agent(self, hierarchy_with_agents):
        """Test removing nonexistent agent raises error."""
        hierarchy = hierarchy_with_agents
        
        with pytest.raises(ValueError, match="not found"):
            hierarchy.remove_agent("nonexistent")
    
    def test_find_agents_by_capability(self, hierarchy_with_agents):
        """Test finding agents by capability."""
        hierarchy = hierarchy_with_agents
        
        # Find data analysis agents
        data_agents = hierarchy.find_agents_by_capability("data_analysis")
        assert len(data_agents) == 1
        assert data_agents[0].name == "Data Worker"
        
        # Find visualization agents
        viz_agents = hierarchy.find_agents_by_capability("visualization")
        assert len(viz_agents) == 1
        assert viz_agents[0].name == "Viz Worker"
        
        # Find nonexistent capability
        nonexistent_agents = hierarchy.find_agents_by_capability("nonexistent")
        assert len(nonexistent_agents) == 0
    
    def test_find_available_agents_by_capability(self, hierarchy_with_agents, sample_agents):
        """Test finding available agents by capability."""
        hierarchy = hierarchy_with_agents
        
        # All agents should be available initially
        data_agents = hierarchy.find_available_agents_by_capability("data_analysis")
        assert len(data_agents) == 1
        assert data_agents[0].is_available
        
        # Make agent unavailable by exceeding max concurrent tasks
        data_worker = sample_agents["data_worker"]
        data_worker._status = ExecutionStatus.RUNNING
        # Add tasks to exceed max_concurrent_tasks (which is 1 by default)
        data_worker._active_tasks["test1"] = Task(name="test1", description="test")
        data_worker._active_tasks["test2"] = Task(name="test2", description="test")
        
        # Should not find agent as it's now unavailable (exceeds max concurrent tasks)
        data_agents = hierarchy.find_available_agents_by_capability("data_analysis")
        assert len(data_agents) == 0  # No longer available as it exceeds max concurrent tasks
    
    def test_capability_based_routing(self, hierarchy_with_agents):
        """Test capability-based task routing."""
        hierarchy = hierarchy_with_agents
        
        # Create task requiring data analysis
        task = Task(
            name="Analysis Task",
            description="Analyze data",
            required_capabilities=["data_analysis"]
        )
        
        best_agent = hierarchy.find_best_agent_for_task(task, "capability_based")
        assert best_agent is not None
        assert best_agent.name == "Data Worker"
        assert best_agent.has_capability("data_analysis")
    
    def test_load_balanced_routing(self, hierarchy_with_agents, sample_agents):
        """Test load-balanced task routing."""
        hierarchy = hierarchy_with_agents
        
        # Create multiple agents with same capability
        data_capability = AgentCapability(
            capability_type=CapabilityType.DATA_ANALYSIS,
            name="Data Analysis",
            description="Analyze data"
        )
        
        # Add second data worker
        data_worker2_config = AgentConfiguration(
            name="Data Worker 2",
            role=AgentRole.WORKER,
            capabilities=[data_capability]
        )
        data_worker2 = WorkerAgent(data_worker2_config)
        hierarchy.add_agent(data_worker2, parent_id=sample_agents["manager"].agent_id)
        
        # Give first worker some load
        data_worker1 = sample_agents["data_worker"]
        data_worker1._active_tasks["task1"] = Task(name="task1", description="test")
        data_worker1._active_tasks["task2"] = Task(name="task2", description="test")
        
        # Create task
        task = Task(
            name="Analysis Task",
            description="Analyze data",
            required_capabilities=["data_analysis"]
        )
        
        # Load balanced routing should pick the less loaded agent
        best_agent = hierarchy.find_best_agent_for_task(task, "load_balanced")
        assert best_agent is not None
        assert best_agent.name == "Data Worker 2"
        assert best_agent.active_task_count == 0
    
    def test_priority_based_routing(self, hierarchy_with_agents):
        """Test priority-based task routing."""
        hierarchy = hierarchy_with_agents
        
        # Create task that multiple agents can handle
        # Both data_worker (priority 2) and code_worker (priority 2) could handle if they had same capability
        # Let's add data analysis capability to code worker for this test
        code_worker = None
        for agent in hierarchy.agents.values():
            if agent.name == "Code Worker":
                code_worker = agent
                break
        
        # Add data analysis capability to code worker
        data_capability = AgentCapability(
            capability_type=CapabilityType.DATA_ANALYSIS,
            name="Data Analysis",
            description="Analyze data"
        )
        code_worker.capabilities["data_analysis"] = data_capability
        
        task = Task(
            name="Analysis Task",
            description="Analyze data",
            required_capabilities=["data_analysis"]
        )
        
        best_agent = hierarchy.find_best_agent_for_task(task, "priority_based")
        assert best_agent is not None
        # Should pick one of the agents with priority 2 (both data_worker and code_worker)
        assert best_agent.priority_level == 2
    
    def test_round_robin_routing(self, hierarchy_with_agents, sample_agents):
        """Test round-robin task routing."""
        hierarchy = hierarchy_with_agents
        
        # Add second data worker
        data_capability = AgentCapability(
            capability_type=CapabilityType.DATA_ANALYSIS,
            name="Data Analysis",
            description="Analyze data"
        )
        
        data_worker2_config = AgentConfiguration(
            name="Data Worker 2",
            role=AgentRole.WORKER,
            capabilities=[data_capability]
        )
        data_worker2 = WorkerAgent(data_worker2_config)
        hierarchy.add_agent(data_worker2, parent_id=sample_agents["manager"].agent_id)
        
        # Create tasks with different IDs
        task1 = Task(
            task_id="task1",
            name="Analysis Task 1",
            description="Analyze data",
            required_capabilities=["data_analysis"]
        )
        
        task2 = Task(
            task_id="task2",
            name="Analysis Task 2",
            description="Analyze data",
            required_capabilities=["data_analysis"]
        )
        
        # Round robin should distribute tasks
        agent1 = hierarchy.find_best_agent_for_task(task1, "round_robin")
        agent2 = hierarchy.find_best_agent_for_task(task2, "round_robin")
        
        assert agent1 is not None
        assert agent2 is not None
        # Agents might be the same or different depending on hash, but both should be capable
        assert agent1.has_capability("data_analysis")
        assert agent2.has_capability("data_analysis")
    
    def test_discover_agents_by_pattern(self, hierarchy_with_agents):
        """Test discovering agents by complex patterns."""
        hierarchy = hierarchy_with_agents
        
        # Find available workers
        pattern = {
            "role": AgentRole.WORKER,
            "available_only": True
        }
        workers = hierarchy.discover_agents_by_pattern(pattern)
        assert len(workers) == 3  # All workers should be available
        
        # Find agents with specific capability
        pattern = {
            "capabilities": ["data_analysis"],
            "available_only": True
        }
        data_agents = hierarchy.discover_agents_by_pattern(pattern)
        assert len(data_agents) == 1
        assert data_agents[0].name == "Data Worker"
        
        # Find agents with specialization
        pattern = {
            "specialization": "analytics"
        }
        analytics_agents = hierarchy.discover_agents_by_pattern(pattern)
        assert len(analytics_agents) == 1
        assert analytics_agents[0].name == "Data Worker"
        
        # Find high priority agents
        pattern = {
            "min_priority": 2  # Priority 1 and 2
        }
        high_priority_agents = hierarchy.discover_agents_by_pattern(pattern)
        assert len(high_priority_agents) >= 1  # Manager has priority 1
    
    def test_get_capability_coverage_report(self, hierarchy_with_agents):
        """Test capability coverage reporting."""
        hierarchy = hierarchy_with_agents
        
        report = hierarchy.get_capability_coverage_report()
        
        assert "capabilities" in report
        assert "total_capabilities" in report
        assert "total_agents" in report
        assert "available_agents" in report
        
        # Check specific capabilities
        capabilities = report["capabilities"]
        assert "data_analysis" in capabilities
        assert "visualization" in capabilities
        assert "coordination" in capabilities
        assert "code_generation" in capabilities
        
        # Check data analysis coverage
        data_analysis_coverage = capabilities["data_analysis"]
        assert data_analysis_coverage["total_agents"] == 1
        assert data_analysis_coverage["available_agents"] == 1
        assert data_analysis_coverage["coverage_percentage"] == 100.0
    
    def test_hierarchy_validation(self, hierarchy_with_agents):
        """Test hierarchy validation."""
        hierarchy = hierarchy_with_agents
        
        is_valid, errors = hierarchy.validate_hierarchy()
        assert is_valid is True
        assert len(errors) == 0
    
    def test_hierarchy_validation_with_errors(self, sample_agents):
        """Test hierarchy validation with errors."""
        hierarchy = AgentHierarchy()
        
        # Create invalid hierarchy - worker with children
        manager_config = AgentConfiguration(
            name="Fake Manager",
            role=AgentRole.WORKER,  # Wrong role for having children
            capabilities=[]
        )
        fake_manager = WorkerAgent(manager_config)
        
        worker = sample_agents["data_worker"]
        
        hierarchy.add_agent(fake_manager)
        hierarchy.add_agent(worker, parent_id=fake_manager.agent_id)
        
        is_valid, errors = hierarchy.validate_hierarchy()
        assert is_valid is False
        assert len(errors) > 0
        assert any("has children but is not a manager" in error for error in errors)
    
    def test_get_hierarchy_summary(self, hierarchy_with_agents):
        """Test hierarchy summary generation."""
        hierarchy = hierarchy_with_agents
        
        summary = hierarchy.get_hierarchy_summary()
        
        assert summary["total_agents"] == 4
        assert summary["root_agents"] == 1
        assert summary["max_depth"] == 2
        assert summary["manager_count"] == 1
        assert summary["worker_count"] == 3
        assert summary["is_valid"] is True
        assert len(summary["validation_errors"]) == 0
        assert "agents_by_depth" in summary
    
    def test_agent_routing_recommendations(self, hierarchy_with_agents):
        """Test agent routing recommendations."""
        hierarchy = hierarchy_with_agents
        
        task = Task(
            name="Analysis Task",
            description="Analyze data",
            required_capabilities=["data_analysis"]
        )
        
        recommendations = hierarchy.get_agent_routing_recommendations(task)
        
        assert "task_id" in recommendations
        assert "required_capabilities" in recommendations
        assert "routing_options" in recommendations
        assert "best_recommendation" in recommendations
        assert "analysis" in recommendations
        
        # Should have routing options
        assert len(recommendations["routing_options"]) > 0
        
        # Should have best recommendation
        assert recommendations["best_recommendation"] is not None
        
        # Analysis should contain useful information
        analysis = recommendations["analysis"]
        assert "total_capable_agents" in analysis
        assert "available_capable_agents" in analysis
        assert "capability_coverage" in analysis
        assert "load_distribution" in analysis
    
    def test_get_agent_network_topology(self, hierarchy_with_agents):
        """Test agent network topology generation."""
        hierarchy = hierarchy_with_agents
        
        topology = hierarchy.get_agent_network_topology()
        
        assert "nodes" in topology
        assert "edges" in topology
        assert "clusters" in topology
        assert "metrics" in topology
        
        # Check nodes
        assert len(topology["nodes"]) == 4
        
        # Check edges (3 workers connected to 1 manager)
        assert len(topology["edges"]) == 3
        
        # Check clusters
        assert "capabilities" in topology["clusters"]
        assert "specializations" in topology["clusters"]
        
        # Check metrics
        metrics = topology["metrics"]
        assert metrics["total_nodes"] == 4
        assert metrics["total_edges"] == 3
        assert metrics["max_depth"] == 2


class TestManagerAgent:
    """Test ManagerAgent functionality."""
    
    @pytest.fixture
    def manager_with_subordinates(self, sample_agents):
        """Create manager with subordinates."""
        manager = sample_agents["manager"]
        data_worker = sample_agents["data_worker"]
        viz_worker = sample_agents["viz_worker"]
        
        manager.add_subordinate(data_worker)
        manager.add_subordinate(viz_worker)
        
        return manager
    
    def test_manager_creation(self, sample_agents):
        """Test ManagerAgent creation."""
        manager = sample_agents["manager"]
        
        assert manager.name == "Test Manager"
        assert manager.role == AgentRole.MANAGER
        assert len(manager.subordinates) == 0
        assert manager.delegation_strategy == DelegationStrategy.CAPABILITY_BASED
        assert manager.coordination_protocol == CoordinationProtocol.ADAPTIVE
    
    def test_add_subordinate(self, sample_agents):
        """Test adding subordinate to manager."""
        manager = sample_agents["manager"]
        worker = sample_agents["data_worker"]
        
        manager.add_subordinate(worker)
        
        assert len(manager.subordinates) == 1
        assert worker in manager.subordinates
        assert worker.manager == manager
    
    def test_remove_subordinate(self, manager_with_subordinates, sample_agents):
        """Test removing subordinate from manager."""
        manager = manager_with_subordinates
        worker = sample_agents["data_worker"]
        
        manager.remove_subordinate(worker.agent_id)
        
        assert len(manager.subordinates) == 1  # Only viz_worker remains
        assert worker not in manager.subordinates
    
    def test_find_capable_subordinates(self, manager_with_subordinates):
        """Test finding capable subordinates."""
        manager = manager_with_subordinates
        
        # Find subordinates with data analysis capability
        capable = manager.find_capable_subordinates(["data_analysis"])
        assert len(capable) == 1
        assert capable[0].name == "Data Worker"
        
        # Find subordinates with visualization capability
        capable = manager.find_capable_subordinates(["visualization"])
        assert len(capable) == 1
        assert capable[0].name == "Viz Worker"
        
        # Find subordinates with nonexistent capability
        capable = manager.find_capable_subordinates(["nonexistent"])
        assert len(capable) == 0
    
    @pytest.mark.asyncio
    async def test_delegate_task(self, manager_with_subordinates):
        """Test task delegation."""
        manager = manager_with_subordinates
        
        task = Task(
            name="Analysis Task",
            description="Analyze data",
            required_capabilities=["data_analysis"]
        )
        
        assignment = await manager.delegate_task(task)
        
        assert assignment is not None
        assert assignment.task_id == task.task_id
        assert assignment.agent_id in [s.agent_id for s in manager.subordinates]
        
        # Should be assigned to data worker
        data_worker = next(s for s in manager.subordinates if s.name == "Data Worker")
        assert assignment.agent_id == data_worker.agent_id
    
    @pytest.mark.asyncio
    async def test_delegate_task_no_capable_subordinates(self, manager_with_subordinates):
        """Test task delegation when no capable subordinates."""
        manager = manager_with_subordinates
        
        task = Task(
            name="Impossible Task",
            description="Do something impossible",
            required_capabilities=["nonexistent_capability"]
        )
        
        with pytest.raises(ValueError, match="No capable subordinates"):
            await manager.delegate_task(task)
    
    @pytest.mark.asyncio
    async def test_receive_progress_update(self, manager_with_subordinates, sample_agents):
        """Test receiving progress updates from subordinates."""
        manager = manager_with_subordinates
        worker = sample_agents["data_worker"]
        
        progress = ProgressUpdate(
            agent_id=worker.agent_id,
            task_id="test_task",
            progress_percentage=50.0,
            status_message="Half complete"
        )
        
        await manager.receive_progress_update(progress)
        
        # Check that subordinate status was updated
        assert worker.agent_id in manager._subordinate_status
        status = manager._subordinate_status[worker.agent_id]
        assert status["last_progress"] == 50.0
        assert status["last_status"] == "Half complete"
    
    def test_get_management_summary(self, manager_with_subordinates):
        """Test management summary generation."""
        manager = manager_with_subordinates
        
        summary = manager.get_management_summary()
        
        assert "subordinate_count" in summary
        assert "available_subordinates" in summary
        assert "active_assignments" in summary
        assert "total_assignments" in summary
        assert "escalations_handled" in summary
        assert "delegation_strategy" in summary
        assert "coordination_protocol" in summary
        
        assert summary["subordinate_count"] == 2
        assert summary["available_subordinates"] == 2


class TestWorkerAgent:
    """Test WorkerAgent functionality."""
    
    @pytest.mark.asyncio
    async def test_worker_execute_task(self, sample_agents):
        """Test worker task execution."""
        worker = sample_agents["data_worker"]
        
        task = Task(
            name="Analysis Task",
            description="Analyze data",
            required_capabilities=["data_analysis"]
        )
        
        context = ExecutionContext(
            session_id="test_session",
            workflow_id="test_workflow"
        )
        
        result = await worker.execute_task(task, context)
        
        assert result is not None
        assert result.task_id == task.task_id
        assert result.agent_id == worker.agent_id
        assert result.status == ExecutionStatus.COMPLETED
        assert "data_retrieved" in result.outputs or "analysis_completed" in result.outputs
    
    @pytest.mark.asyncio
    async def test_worker_execute_task_without_capability(self, sample_agents):
        """Test worker task execution without required capability."""
        worker = sample_agents["data_worker"]
        
        task = Task(
            name="Visualization Task",
            description="Create visualization",
            required_capabilities=["visualization"]  # Worker doesn't have this
        )
        
        context = ExecutionContext(
            session_id="test_session",
            workflow_id="test_workflow"
        )
        
        result = await worker.execute_task(task, context)
        
        # Should fail due to missing capability
        assert result.status == ExecutionStatus.FAILED
        assert "cannot handle task" in result.error_message
    
    def test_worker_performance_summary(self, sample_agents):
        """Test worker performance summary."""
        worker = sample_agents["data_worker"]
        
        summary = worker.get_performance_summary()
        
        assert "performance_metrics" in summary
        assert "execution_history_count" in summary
        assert "assistance_requests_count" in summary
        assert "has_manager" in summary
        assert "manager_id" in summary
        
        # Initial state
        assert summary["performance_metrics"]["tasks_completed"] == 0
        assert summary["performance_metrics"]["tasks_failed"] == 0
        assert summary["performance_metrics"]["success_rate"] == 1.0
        assert summary["has_manager"] is False


class TestCommunicationProtocols:
    """Test agent communication protocols."""
    
    @pytest.mark.asyncio
    async def test_progress_reporting(self, sample_agents):
        """Test progress reporting between agents."""
        manager = sample_agents["manager"]
        worker = sample_agents["data_worker"]
        
        # Set up relationship
        manager.add_subordinate(worker)
        
        # Mock the manager's receive_progress_update method
        manager.receive_progress_update = AsyncMock()
        
        # Worker reports progress
        progress = ProgressUpdate(
            agent_id=worker.agent_id,
            task_id="test_task",
            progress_percentage=75.0,
            status_message="Almost done"
        )
        
        await worker.report_progress(progress)
        
        # Verify manager received the update
        manager.receive_progress_update.assert_called_once_with(progress)
    
    @pytest.mark.asyncio
    async def test_assistance_request(self, sample_agents):
        """Test assistance request mechanism."""
        worker = sample_agents["data_worker"]
        
        assistance_request = AssistanceRequest(
            requesting_agent_id=worker.agent_id,
            task_id="difficult_task",
            assistance_type="capability_gap",
            description="Need help with visualization",
            required_capabilities=["visualization"]
        )
        
        await worker.request_assistance(assistance_request)
        
        # Verify request was stored
        assert len(worker._assistance_requests) == 1
        assert worker._assistance_requests[0] == assistance_request
    
    @pytest.mark.asyncio
    async def test_escalation_handling(self, sample_agents):
        """Test escalation handling by manager."""
        manager = sample_agents["manager"]
        
        escalation = Escalation(
            from_agent_id="worker_id",
            task_id="failed_task",
            escalation_type="resource_shortage",
            reason="Insufficient memory",
            severity="high"
        )
        
        resolution = await manager.handle_escalation(escalation)
        
        assert resolution is not None
        assert resolution.escalation_id == escalation.escalation_id
        assert resolution.resolution_type in ["resource_reallocation", "capability_substitution", "quality_review", "manual_review"]
        assert escalation.escalation_id in manager._escalations


if __name__ == "__main__":
    pytest.main([__file__])