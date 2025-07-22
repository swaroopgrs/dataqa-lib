"""Tests for orchestration agents."""

import pytest
from unittest.mock import AsyncMock

from src.dataqa.orchestration.models import (
    AgentCapability,
    AgentConfiguration,
    AgentRole,
    CapabilityType,
)
from src.dataqa.orchestration.agents import (
    BaseAgent,
    ManagerAgent,
    WorkerAgent,
    AgentHierarchy,
    Task,
    ExecutionContext,
)


class TestOrchestrationAgents:
    """Test orchestration agent classes."""
    
    def test_worker_agent_creation(self):
        """Test WorkerAgent creation and basic functionality."""
        capability = AgentCapability(
            capability_type=CapabilityType.DATA_ANALYSIS,
            name="Data Analysis",
            description="Analyze data"
        )
        
        config = AgentConfiguration(
            name="Test Worker",
            role=AgentRole.WORKER,
            capabilities=[capability]
        )
        
        worker = WorkerAgent(config)
        
        assert worker.name == "Test Worker"
        assert worker.role == AgentRole.WORKER
        assert worker.is_available is True
        assert worker.active_task_count == 0
        assert worker.has_capability("data_analysis")
    
    def test_manager_agent_creation(self):
        """Test ManagerAgent creation and basic functionality."""
        capability = AgentCapability(
            capability_type=CapabilityType.COORDINATION,
            name="Coordination",
            description="Coordinate tasks"
        )
        
        config = AgentConfiguration(
            name="Test Manager",
            role=AgentRole.MANAGER,
            capabilities=[capability]
        )
        
        manager = ManagerAgent(config)
        
        assert manager.name == "Test Manager"
        assert manager.role == AgentRole.MANAGER
        assert len(manager.subordinates) == 0
        assert manager.is_available is True
    
    def test_agent_hierarchy_creation(self):
        """Test AgentHierarchy creation and management."""
        # Create manager
        manager_capability = AgentCapability(
            capability_type=CapabilityType.COORDINATION,
            name="Coordination",
            description="Coordinate tasks"
        )
        
        manager_config = AgentConfiguration(
            name="Manager",
            role=AgentRole.MANAGER,
            capabilities=[manager_capability]
        )
        
        manager = ManagerAgent(manager_config)
        
        # Create worker
        worker_capability = AgentCapability(
            capability_type=CapabilityType.DATA_ANALYSIS,
            name="Data Analysis",
            description="Analyze data"
        )
        
        worker_config = AgentConfiguration(
            name="Worker",
            role=AgentRole.WORKER,
            capabilities=[worker_capability]
        )
        
        worker = WorkerAgent(worker_config)
        
        # Create hierarchy
        hierarchy = AgentHierarchy()
        hierarchy.add_agent(manager)
        hierarchy.add_agent(worker, parent_id=manager.agent_id)
        
        # Test hierarchy structure
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
    
    def test_hierarchy_validation(self):
        """Test hierarchy validation."""
        hierarchy = AgentHierarchy()
        
        # Empty hierarchy should be valid
        is_valid, errors = hierarchy.validate_hierarchy()
        assert is_valid is True
        assert len(errors) == 0
        
        # Add valid hierarchy
        manager_config = AgentConfiguration(
            name="Manager",
            role=AgentRole.MANAGER,
            capabilities=[]
        )
        manager = ManagerAgent(manager_config)
        
        worker_config = AgentConfiguration(
            name="Worker",
            role=AgentRole.WORKER,
            capabilities=[]
        )
        worker = WorkerAgent(worker_config)
        
        hierarchy.add_agent(manager)
        hierarchy.add_agent(worker, parent_id=manager.agent_id)
        
        is_valid, errors = hierarchy.validate_hierarchy()
        assert is_valid is True
        assert len(errors) == 0
    
    def test_capability_based_queries(self):
        """Test capability-based agent queries."""
        hierarchy = AgentHierarchy()
        
        # Create agents with different capabilities
        data_capability = AgentCapability(
            capability_type=CapabilityType.DATA_ANALYSIS,
            name="Data Analysis",
            description="Analyze data"
        )
        
        viz_capability = AgentCapability(
            capability_type=CapabilityType.VISUALIZATION,
            name="Visualization",
            description="Create visualizations"
        )
        
        data_worker_config = AgentConfiguration(
            name="Data Worker",
            role=AgentRole.WORKER,
            capabilities=[data_capability]
        )
        
        viz_worker_config = AgentConfiguration(
            name="Viz Worker",
            role=AgentRole.WORKER,
            capabilities=[viz_capability]
        )
        
        data_worker = WorkerAgent(data_worker_config)
        viz_worker = WorkerAgent(viz_worker_config)
        
        hierarchy.add_agent(data_worker)
        hierarchy.add_agent(viz_worker)
        
        # Test capability queries
        data_agents = hierarchy.find_agents_by_capability("data_analysis")
        assert len(data_agents) == 1
        assert data_agents[0].name == "Data Worker"
        
        viz_agents = hierarchy.find_agents_by_capability("visualization")
        assert len(viz_agents) == 1
        assert viz_agents[0].name == "Viz Worker"
        
        # Test available agents query
        available_data_agents = hierarchy.find_available_agents_by_capability("data_analysis")
        assert len(available_data_agents) == 1
        assert available_data_agents[0].is_available is True
    
    def test_task_capability_matching(self):
        """Test task capability matching."""
        capability = AgentCapability(
            capability_type=CapabilityType.DATA_ANALYSIS,
            name="Data Analysis",
            description="Analyze data"
        )
        
        config = AgentConfiguration(
            name="Test Worker",
            role=AgentRole.WORKER,
            capabilities=[capability]
        )
        
        worker = WorkerAgent(config)
        
        # Create task requiring data analysis
        task = Task(
            name="Analysis Task",
            description="Analyze some data",
            required_capabilities=["data_analysis"]
        )
        
        assert worker.can_handle_task(task) is True
        
        # Create task requiring different capability
        task_viz = Task(
            name="Visualization Task",
            description="Create a chart",
            required_capabilities=["visualization"]
        )
        
        assert worker.can_handle_task(task_viz) is False