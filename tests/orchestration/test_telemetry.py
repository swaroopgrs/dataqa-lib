"""
Tests for telemetry collection system.
"""

import pytest
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from src.dataqa.orchestration.monitoring.telemetry import (
    TelemetryCollector,
    ExecutionTelemetry,
    AgentMetrics,
    DecisionPoint,
    ResourceUsage,
    ExecutionStep
)
from src.dataqa.orchestration.models import AgentType, ExecutionPhase


class TestTelemetryCollector:
    """Test telemetry collection functionality."""
    
    @pytest.fixture
    def collector(self):
        """Create a telemetry collector for testing."""
        return TelemetryCollector()
    
    def test_start_execution(self, collector):
        """Test starting execution tracking."""
        execution_id = "test-exec-1"
        workflow_id = "test-workflow"
        correlation_id = "test-corr-1"
        
        collector.start_execution(execution_id, workflow_id, correlation_id)
        
        # Check execution was registered
        assert execution_id in collector._active_executions
        
        telemetry = collector._active_executions[execution_id]
        assert telemetry.execution_id == execution_id
        assert telemetry.workflow_id == workflow_id
        assert telemetry.correlation_id == correlation_id
        assert telemetry.status == "running"
        assert telemetry.start_time is not None
    
    def test_end_execution(self, collector):
        """Test ending execution tracking."""
        execution_id = "test-exec-1"
        workflow_id = "test-workflow"
        correlation_id = "test-corr-1"
        
        # Start execution
        collector.start_execution(execution_id, workflow_id, correlation_id)
        
        # Add some agents
        collector._active_executions[execution_id].participating_agents = ["agent1", "agent2"]
        
        # End execution
        result = collector.end_execution(execution_id, "completed")
        
        # Check result
        assert result is not None
        assert result.status == "completed"
        assert result.end_time is not None
        assert result.duration_ms is not None
        assert result.duration_ms > 0
        
        # Check execution removed from active
        assert execution_id not in collector._active_executions
        
        # Check agent metrics updated
        assert "agent1" in collector._agent_metrics
        assert "agent2" in collector._agent_metrics
    
    def test_start_step(self, collector):
        """Test starting step tracking."""
        execution_id = "test-exec-1"
        collector.start_execution(execution_id, "workflow", "corr")
        
        step_id = "step-1"
        agent_id = "agent-1"
        step_type = "analysis"
        phase = ExecutionPhase.EXECUTION
        
        collector.start_step(execution_id, step_id, agent_id, step_type, phase)
        
        telemetry = collector._active_executions[execution_id]
        
        # Check agent added to participating agents
        assert agent_id in telemetry.participating_agents
        
        # Check step created
        assert len(telemetry.execution_steps) == 1
        step = telemetry.execution_steps[0]
        assert step.step_id == step_id
        assert step.agent_id == agent_id
        assert step.step_type == step_type
        assert step.phase == phase
        assert step.start_time is not None
    
    def test_end_step(self, collector):
        """Test ending step tracking."""
        execution_id = "test-exec-1"
        collector.start_execution(execution_id, "workflow", "corr")
        
        step_id = "step-1"
        collector.start_step(execution_id, step_id, "agent-1", "analysis", ExecutionPhase.EXECUTION)
        
        # End step successfully
        collector.end_step(execution_id, step_id, success=True)
        
        telemetry = collector._active_executions[execution_id]
        step = telemetry.execution_steps[0]
        
        assert step.success is True
        assert step.end_time is not None
        assert step.duration_ms is not None
        assert step.duration_ms > 0
        assert step.error_message is None
    
    def test_end_step_with_error(self, collector):
        """Test ending step with error."""
        execution_id = "test-exec-1"
        collector.start_execution(execution_id, "workflow", "corr")
        
        step_id = "step-1"
        collector.start_step(execution_id, step_id, "agent-1", "analysis", ExecutionPhase.EXECUTION)
        
        error_message = "Test error"
        collector.end_step(execution_id, step_id, success=False, error_message=error_message)
        
        telemetry = collector._active_executions[execution_id]
        step = telemetry.execution_steps[0]
        
        assert step.success is False
        assert step.error_message == error_message
        assert telemetry.error_count == 1
    
    def test_record_decision(self, collector):
        """Test recording decision points."""
        execution_id = "test-exec-1"
        collector.start_execution(execution_id, "workflow", "corr")
        
        step_id = "step-1"
        collector.start_step(execution_id, step_id, "agent-1", "analysis", ExecutionPhase.EXECUTION)
        
        decision = DecisionPoint(
            decision_id="decision-1",
            agent_id="agent-1",
            decision_type="algorithm_choice",
            context={"data_size": 1000},
            options_considered=["linear", "tree", "neural"],
            chosen_option="tree",
            reasoning="Tree algorithm best for this data size",
            confidence_score=0.85
        )
        
        collector.record_decision(execution_id, step_id, decision)
        
        telemetry = collector._active_executions[execution_id]
        step = telemetry.execution_steps[0]
        
        assert len(step.decision_points) == 1
        assert step.decision_points[0] == decision
        assert telemetry.total_decision_points == 1
    
    def test_track_step_context_manager(self, collector):
        """Test step tracking context manager."""
        execution_id = "test-exec-1"
        collector.start_execution(execution_id, "workflow", "corr")
        
        step_id = "step-1"
        agent_id = "agent-1"
        
        # Test successful execution
        with collector.track_step(execution_id, step_id, agent_id, "analysis", ExecutionPhase.EXECUTION):
            time.sleep(0.01)  # Small delay to ensure duration > 0
        
        telemetry = collector._active_executions[execution_id]
        step = telemetry.execution_steps[0]
        
        assert step.success is True
        assert step.duration_ms > 0
    
    def test_track_step_context_manager_with_exception(self, collector):
        """Test step tracking context manager with exception."""
        execution_id = "test-exec-1"
        collector.start_execution(execution_id, "workflow", "corr")
        
        step_id = "step-1"
        agent_id = "agent-1"
        
        # Test exception handling
        with pytest.raises(ValueError):
            with collector.track_step(execution_id, step_id, agent_id, "analysis", ExecutionPhase.EXECUTION):
                raise ValueError("Test error")
        
        telemetry = collector._active_executions[execution_id]
        step = telemetry.execution_steps[0]
        
        assert step.success is False
        assert step.error_message == "Test error"
    
    @patch('psutil.Process')
    def test_resource_usage_collection(self, mock_process, collector):
        """Test resource usage collection."""
        # Mock psutil process
        mock_proc = Mock()
        mock_proc.cpu_percent.return_value = 25.5
        mock_proc.memory_info.return_value = Mock(rss=1024*1024*100)  # 100MB
        mock_proc.memory_percent.return_value = 15.2
        mock_proc.io_counters.return_value = Mock(
            read_bytes=1024*1024*50,   # 50MB
            write_bytes=1024*1024*25   # 25MB
        )
        mock_process.return_value = mock_proc
        
        resource_usage = collector._collect_resource_usage()
        
        assert resource_usage.cpu_percent == 25.5
        assert resource_usage.memory_mb == 100.0
        assert resource_usage.memory_percent == 15.2
        assert resource_usage.disk_io_read_mb == 50.0
        assert resource_usage.disk_io_write_mb == 25.0
    
    def test_get_execution_telemetry(self, collector):
        """Test getting execution telemetry."""
        execution_id = "test-exec-1"
        collector.start_execution(execution_id, "workflow", "corr")
        
        telemetry = collector.get_execution_telemetry(execution_id)
        assert telemetry is not None
        assert telemetry.execution_id == execution_id
        
        # Test non-existent execution
        assert collector.get_execution_telemetry("non-existent") is None
    
    def test_get_agent_metrics(self, collector):
        """Test getting agent metrics."""
        execution_id = "test-exec-1"
        collector.start_execution(execution_id, "workflow", "corr")
        
        # Add agent to execution
        collector._active_executions[execution_id].participating_agents = ["agent-1"]
        
        # End execution to update metrics
        collector.end_execution(execution_id, "completed")
        
        metrics = collector.get_agent_metrics("agent-1")
        assert metrics is not None
        assert metrics.agent_id == "agent-1"
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
    
    def test_get_active_executions(self, collector):
        """Test getting active executions."""
        assert collector.get_active_executions() == []
        
        collector.start_execution("exec-1", "workflow", "corr")
        collector.start_execution("exec-2", "workflow", "corr")
        
        active = collector.get_active_executions()
        assert len(active) == 2
        assert "exec-1" in active
        assert "exec-2" in active
    
    def test_agent_metrics_update(self, collector):
        """Test agent metrics are properly updated."""
        execution_id = "test-exec-1"
        collector.start_execution(execution_id, "workflow", "corr")
        
        # Create execution with agent
        telemetry = collector._active_executions[execution_id]
        telemetry.participating_agents = ["agent-1"]
        telemetry.total_decision_points = 3
        
        # End execution
        collector.end_execution(execution_id, "completed")
        
        metrics = collector._agent_metrics["agent-1"]
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        assert metrics.failed_executions == 0
        assert metrics.total_execution_time_ms > 0  # Should have some execution time
        assert metrics.average_execution_time_ms > 0  # Should have some average time
        assert metrics.decision_count == 3
        assert metrics.last_execution is not None
    
    def test_thread_safety(self, collector):
        """Test thread safety of collector operations."""
        import threading
        import time
        
        def worker(thread_id):
            execution_id = f"exec-{thread_id}"
            collector.start_execution(execution_id, "workflow", f"corr-{thread_id}")
            
            for i in range(5):
                step_id = f"step-{thread_id}-{i}"
                collector.start_step(execution_id, step_id, f"agent-{thread_id}", "work", ExecutionPhase.EXECUTION)
                time.sleep(0.001)  # Small delay
                collector.end_step(execution_id, step_id, success=True)
            
            collector.end_execution(execution_id, "completed")
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all executions completed
        assert len(collector._agent_metrics) == 5
        for i in range(5):
            agent_id = f"agent-{i}"
            assert agent_id in collector._agent_metrics
            metrics = collector._agent_metrics[agent_id]
            assert metrics.total_executions == 1
            assert metrics.successful_executions == 1


class TestResourceUsage:
    """Test ResourceUsage model."""
    
    def test_resource_usage_creation(self):
        """Test creating ResourceUsage instance."""
        usage = ResourceUsage(
            cpu_percent=25.5,
            memory_mb=100.0,
            memory_percent=15.2,
            disk_io_read_mb=50.0,
            disk_io_write_mb=25.0,
            network_sent_mb=10.0,
            network_recv_mb=5.0
        )
        
        assert usage.cpu_percent == 25.5
        assert usage.memory_mb == 100.0
        assert usage.timestamp is not None


class TestDecisionPoint:
    """Test DecisionPoint model."""
    
    def test_decision_point_creation(self):
        """Test creating DecisionPoint instance."""
        decision = DecisionPoint(
            decision_id="decision-1",
            agent_id="agent-1",
            decision_type="algorithm_choice",
            context={"data_size": 1000},
            options_considered=["linear", "tree"],
            chosen_option="tree",
            reasoning="Best for data size",
            confidence_score=0.85
        )
        
        assert decision.decision_id == "decision-1"
        assert decision.agent_id == "agent-1"
        assert decision.confidence_score == 0.85
        assert decision.timestamp is not None


class TestExecutionTelemetry:
    """Test ExecutionTelemetry model."""
    
    def test_execution_telemetry_creation(self):
        """Test creating ExecutionTelemetry instance."""
        telemetry = ExecutionTelemetry(
            execution_id="exec-1",
            workflow_id="workflow-1",
            correlation_id="corr-1",
            start_time=datetime.now(timezone.utc)
        )
        
        assert telemetry.execution_id == "exec-1"
        assert telemetry.workflow_id == "workflow-1"
        assert telemetry.status == "running"
        assert telemetry.participating_agents == []
        assert telemetry.execution_steps == []


class TestAgentMetrics:
    """Test AgentMetrics model."""
    
    def test_agent_metrics_creation(self):
        """Test creating AgentMetrics instance."""
        metrics = AgentMetrics(
            agent_id="agent-1",
            agent_type=AgentType.WORKER
        )
        
        assert metrics.agent_id == "agent-1"
        assert metrics.agent_type == AgentType.WORKER
        assert metrics.total_executions == 0
        assert metrics.successful_executions == 0
        assert metrics.failed_executions == 0
        assert metrics.capabilities_used == set()