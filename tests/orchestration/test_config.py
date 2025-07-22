"""Tests for orchestration configuration schemas."""

import pytest
from datetime import datetime

from src.dataqa.orchestration.config import (
    CapabilityConfigSchema,
    AgentConfigSchema,
    WorkflowConfigSchema,
    MultiAgentWorkflowConfig,
    WorkflowTrigger,
    WorkflowStatus,
)
from src.dataqa.orchestration.models import (
    AgentRole,
    CapabilityType,
)


class TestOrchestrationConfig:
    """Test orchestration configuration schemas."""
    
    def test_capability_config_schema(self):
        """Test CapabilityConfigSchema creation and validation."""
        config = CapabilityConfigSchema(
            capability_type=CapabilityType.DATA_ANALYSIS,
            name="Advanced Analytics",
            description="Perform complex data analysis",
            cpu_cores=2,
            memory_mb=1024,
            execution_timeout_seconds=600
        )
        
        assert config.capability_type == CapabilityType.DATA_ANALYSIS
        assert config.name == "Advanced Analytics"
        assert config.cpu_cores == 2
        assert config.memory_mb == 1024
        assert config.execution_timeout_seconds == 600
        assert config.version == "1.0.0"
    
    def test_agent_config_schema(self):
        """Test AgentConfigSchema creation and validation."""
        capability_config = CapabilityConfigSchema(
            capability_type=CapabilityType.DATA_RETRIEVAL,
            name="Data Retrieval",
            description="Retrieve data from sources"
        )
        
        config = AgentConfigSchema(
            name="Data Agent",
            role=AgentRole.WORKER,
            agent_type="worker",
            capabilities=[capability_config],
            max_concurrent_tasks=3,
            specialization="finance"
        )
        
        assert config.name == "Data Agent"
        assert config.role == AgentRole.WORKER
        assert config.agent_type == "worker"
        assert len(config.capabilities) == 1
        assert config.max_concurrent_tasks == 3
        assert config.specialization == "finance"
    
    def test_workflow_config_schema(self):
        """Test WorkflowConfigSchema creation and validation."""
        capability_config = CapabilityConfigSchema(
            capability_type=CapabilityType.VISUALIZATION,
            name="Visualization",
            description="Create charts and graphs"
        )
        
        agent_config = AgentConfigSchema(
            name="Viz Agent",
            role=AgentRole.WORKER,
            capabilities=[capability_config]
        )
        
        workflow_config = WorkflowConfigSchema(
            name="Analytics Workflow",
            description="Complete analytics pipeline",
            workflow_type="data_analysis",
            trigger=WorkflowTrigger.MANUAL,
            status=WorkflowStatus.ACTIVE,
            agents=[agent_config],
            max_execution_time_minutes=120
        )
        
        assert workflow_config.name == "Analytics Workflow"
        assert workflow_config.trigger == WorkflowTrigger.MANUAL
        assert workflow_config.status == WorkflowStatus.ACTIVE
        assert len(workflow_config.agents) == 1
        assert workflow_config.max_execution_time_minutes == 120
        assert isinstance(workflow_config.created_at, datetime)
    
    def test_multi_agent_workflow_config(self):
        """Test MultiAgentWorkflowConfig creation and runtime conversion."""
        capability_config = CapabilityConfigSchema(
            capability_type=CapabilityType.CODE_GENERATION,
            name="Code Generation",
            description="Generate code solutions"
        )
        
        agent_config = AgentConfigSchema(
            name="Code Agent",
            role=AgentRole.WORKER,
            capabilities=[capability_config]
        )
        
        workflow_config = WorkflowConfigSchema(
            name="Code Generation Workflow",
            description="Automated code generation pipeline",
            agents=[agent_config]
        )
        
        complete_config = MultiAgentWorkflowConfig(
            workflow=workflow_config
        )
        
        assert complete_config.workflow.name == "Code Generation Workflow"
        assert complete_config.config_version == "1.0"
        assert isinstance(complete_config.config_id, str)
        assert isinstance(complete_config.created_at, datetime)
        
        # Test runtime config conversion
        runtime_config = complete_config.to_runtime_config()
        assert "workflow_id" in runtime_config
        assert "workflow" in runtime_config
        assert "config_metadata" in runtime_config
        assert runtime_config["workflow"]["name"] == "Code Generation Workflow"
    
    def test_agent_hierarchy_validation(self):
        """Test agent hierarchy validation in workflow config."""
        # Create valid hierarchy
        manager_config = AgentConfigSchema(
            name="Manager",
            role=AgentRole.MANAGER,
            agent_type="manager"
        )
        
        worker_config = AgentConfigSchema(
            name="Worker",
            role=AgentRole.WORKER,
            agent_type="worker",
            parent_agent="Manager"
        )
        
        # This should validate successfully
        workflow_config = WorkflowConfigSchema(
            name="Hierarchical Workflow",
            description="Workflow with agent hierarchy",
            agents=[manager_config, worker_config]
        )
        
        assert len(workflow_config.agents) == 2
        
        # Test invalid hierarchy (non-existent parent)
        invalid_worker_config = AgentConfigSchema(
            name="Invalid Worker",
            role=AgentRole.WORKER,
            parent_agent="NonExistentManager"
        )
        
        with pytest.raises(ValueError, match="references non-existent parent"):
            WorkflowConfigSchema(
                name="Invalid Workflow",
                description="Workflow with invalid hierarchy",
                agents=[invalid_worker_config]
            )
    
    def test_role_consistency_validation(self):
        """Test role consistency validation in agent config."""
        # Valid manager configuration
        manager_config = AgentConfigSchema(
            name="Manager",
            role=AgentRole.MANAGER,
            agent_type="manager"
        )
        assert manager_config.role == AgentRole.MANAGER
        
        # Valid worker configuration
        worker_config = AgentConfigSchema(
            name="Worker",
            role=AgentRole.WORKER,
            agent_type="worker"
        )
        assert worker_config.role == AgentRole.WORKER
        
        # Invalid configuration - manager type with worker role
        with pytest.raises(ValueError, match="Manager agent_type must have MANAGER role"):
            AgentConfigSchema(
                name="Invalid Manager",
                role=AgentRole.WORKER,
                agent_type="manager"
            )
        
        # Invalid configuration - worker type with manager role
        with pytest.raises(ValueError, match="Worker/specialist agent_type cannot have MANAGER role"):
            AgentConfigSchema(
                name="Invalid Worker",
                role=AgentRole.MANAGER,
                agent_type="worker"
            )