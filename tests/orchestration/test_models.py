"""Tests for orchestration models."""

import pytest
from datetime import datetime
from uuid import uuid4

from src.dataqa.orchestration.models import (
    AgentCapability,
    AgentConfiguration,
    AgentRole,
    CapabilityType,
    ExecutionState,
    ExecutionStatus,
    MultiAgentWorkflow,
    DomainContext,
    BusinessRule,
)


class TestOrchestrationModels:
    """Test orchestration Pydantic models."""
    
    def test_agent_capability_creation(self):
        """Test AgentCapability model creation."""
        capability = AgentCapability(
            capability_type=CapabilityType.DATA_ANALYSIS,
            name="Test Capability",
            description="Test capability for validation"
        )
        
        assert capability.capability_type == CapabilityType.DATA_ANALYSIS
        assert capability.name == "Test Capability"
        assert capability.description == "Test capability for validation"
        assert capability.version == "1.0.0"
        assert isinstance(capability.capability_id, str)
    
    def test_agent_configuration_creation(self):
        """Test AgentConfiguration model creation."""
        capability = AgentCapability(
            capability_type=CapabilityType.DATA_RETRIEVAL,
            name="Data Retrieval",
            description="Retrieve data from sources"
        )
        
        config = AgentConfiguration(
            name="Test Agent",
            role=AgentRole.WORKER,
            capabilities=[capability],
            specialization="finance"
        )
        
        assert config.name == "Test Agent"
        assert config.role == AgentRole.WORKER
        assert len(config.capabilities) == 1
        assert config.specialization == "finance"
        assert config.max_concurrent_tasks == 1
        assert config.enabled is True
    
    def test_execution_state_creation(self):
        """Test ExecutionState model creation."""
        state = ExecutionState()
        
        assert isinstance(state.session_id, str)
        assert state.status == ExecutionStatus.PENDING
        assert len(state.completed_steps) == 0
        assert len(state.intermediate_results) == 0
        assert isinstance(state.updated_at, datetime)
    
    def test_multi_agent_workflow_creation(self):
        """Test MultiAgentWorkflow model creation."""
        capability = AgentCapability(
            capability_type=CapabilityType.VISUALIZATION,
            name="Visualization",
            description="Create visualizations"
        )
        
        agent_config = AgentConfiguration(
            name="Viz Agent",
            role=AgentRole.WORKER,
            capabilities=[capability]
        )
        
        workflow = MultiAgentWorkflow(
            name="Test Workflow",
            description="Test workflow for validation",
            agents=[agent_config]
        )
        
        assert workflow.name == "Test Workflow"
        assert workflow.description == "Test workflow for validation"
        assert len(workflow.agents) == 1
        assert workflow.version == "1.0.0"
        assert isinstance(workflow.workflow_id, str)
        assert isinstance(workflow.created_at, datetime)
    
    def test_domain_context_creation(self):
        """Test DomainContext model creation."""
        rule = BusinessRule(
            name="Test Rule",
            description="Test business rule",
            rule_type="validation",
            condition="amount > 0",
            action="approve"
        )
        
        context = DomainContext(
            domain_name="finance",
            applicable_rules=[rule]
        )
        
        assert context.domain_name == "finance"
        assert len(context.applicable_rules) == 1
        assert context.applicable_rules[0].name == "Test Rule"
    
    def test_enum_values(self):
        """Test that enum values are properly handled."""
        # Test CapabilityType enum
        assert CapabilityType.DATA_ANALYSIS.value == "data_analysis"
        assert CapabilityType.VISUALIZATION.value == "visualization"
        
        # Test AgentRole enum
        assert AgentRole.MANAGER.value == "manager"
        assert AgentRole.WORKER.value == "worker"
        
        # Test ExecutionStatus enum
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.COMPLETED.value == "completed"