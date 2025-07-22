"""
Tests for the ExecutionStateManager.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

from src.dataqa.orchestration.models import ExecutionState, ExecutionStatus
from src.dataqa.orchestration.models import ExecutionStep as ModelsExecutionStep
from src.dataqa.orchestration.planning.models import (
    ExecutionStep as PlanningExecutionStep,
    IntermediateResult,
    Plan,
    ReplanningEvent,
    ReplanningTriggerType,
)
from src.dataqa.orchestration.planning.execution_state import ExecutionStateManager


class TestExecutionStateManager:
    """Test ExecutionStateManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ExecutionStateManager()
    
    @pytest.mark.asyncio
    async def test_create_execution_state(self):
        """Test creating a new execution state."""
        plan = Plan(
            name="Test Plan",
            description="Test plan",
            steps=[
                PlanningExecutionStep(name="Step 1", description="First step"),
                PlanningExecutionStep(name="Step 2", description="Second step")
            ]
        )
        
        execution_state = await self.manager.create_execution_state("session-123", plan)
        
        assert execution_state.session_id == "session-123"
        assert execution_state.current_plan_id == plan.plan_id
        assert execution_state.status == ExecutionStatus.PENDING
        assert execution_state.execution_metrics.total_steps == 2
        assert execution_state.started_at is not None
    
    @pytest.mark.asyncio
    async def test_update_step_status_new_step(self):
        """Test updating status of a new step."""
        execution_state = ExecutionState(session_id="test-session")
        
        await self.manager.update_step_status(
            execution_state,
            "step-123",
            ExecutionStatus.RUNNING
        )
        
        assert len(execution_state.completed_steps) == 1
        step = execution_state.completed_steps[0]
        assert step.step_id == "step-123"
        assert step.status == ExecutionStatus.RUNNING
        assert step.started_at is not None
    
    @pytest.mark.asyncio
    async def test_update_step_status_existing_step(self):
        """Test updating status of an existing step."""
        existing_step = ModelsExecutionStep(
            step_id="step-123",
            name="Test Step",
            description="Test",
            status=ExecutionStatus.RUNNING
        )
        execution_state = ExecutionState(
            session_id="test-session",
            completed_steps=[existing_step]
        )
        
        await self.manager.update_step_status(
            execution_state,
            "step-123",
            ExecutionStatus.COMPLETED,
            outputs={"result": "success"}
        )
        
        step = execution_state.completed_steps[0]
        assert step.status == ExecutionStatus.COMPLETED
        assert step.completed_at is not None
        assert step.outputs == {"result": "success"}
    
    @pytest.mark.asyncio
    async def test_update_step_status_with_error(self):
        """Test updating step status with error message."""
        execution_state = ExecutionState(session_id="test-session")
        
        await self.manager.update_step_status(
            execution_state,
            "step-123",
            ExecutionStatus.FAILED,
            error_message="Step failed due to timeout"
        )
        
        step = execution_state.completed_steps[0]
        assert step.status == ExecutionStatus.FAILED
        assert step.error_message == "Step failed due to timeout"
        assert step.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_add_intermediate_result(self):
        """Test adding an intermediate result."""
        execution_state = ExecutionState(session_id="test-session")
        
        result = await self.manager.add_intermediate_result(
            execution_state,
            "step-123",
            "analysis_output",
            {"data": "test result"},
            quality_score=0.85,
            metadata={"source": "test"}
        )
        
        assert result.step_id == "step-123"
        assert result.result_type == "analysis_output"
        assert result.data == {"data": "test result"}
        assert result.quality_score == 0.85
        assert result.metadata == {"source": "test"}
        
        # Check it's stored in execution state
        assert result.result_id in execution_state.intermediate_results
        stored_result = execution_state.intermediate_results[result.result_id]
        assert stored_result["step_id"] == "step-123"
        assert stored_result["result_type"] == "analysis_output"
        
        # Check quality metrics updated
        assert execution_state.execution_metrics.quality_scores["analysis_output"] == 0.85
    
    @pytest.mark.asyncio
    async def test_get_intermediate_results_no_filter(self):
        """Test getting all intermediate results."""
        execution_state = ExecutionState(session_id="test-session")
        
        # Add multiple results
        result1 = await self.manager.add_intermediate_result(
            execution_state, "step-1", "type-a", "data-1"
        )
        result2 = await self.manager.add_intermediate_result(
            execution_state, "step-2", "type-b", "data-2"
        )
        
        results = await self.manager.get_intermediate_results(execution_state)
        
        assert len(results) == 2
        result_ids = [r.result_id for r in results]
        assert result1.result_id in result_ids
        assert result2.result_id in result_ids
    
    @pytest.mark.asyncio
    async def test_get_intermediate_results_filter_by_step(self):
        """Test getting intermediate results filtered by step ID."""
        execution_state = ExecutionState(session_id="test-session")
        
        # Add results for different steps
        await self.manager.add_intermediate_result(
            execution_state, "step-1", "type-a", "data-1"
        )
        result2 = await self.manager.add_intermediate_result(
            execution_state, "step-2", "type-b", "data-2"
        )
        
        results = await self.manager.get_intermediate_results(
            execution_state, step_id="step-2"
        )
        
        assert len(results) == 1
        assert results[0].result_id == result2.result_id
        assert results[0].step_id == "step-2"
    
    @pytest.mark.asyncio
    async def test_get_intermediate_results_filter_by_type(self):
        """Test getting intermediate results filtered by result type."""
        execution_state = ExecutionState(session_id="test-session")
        
        # Add results of different types
        result1 = await self.manager.add_intermediate_result(
            execution_state, "step-1", "analysis", "data-1"
        )
        await self.manager.add_intermediate_result(
            execution_state, "step-2", "visualization", "data-2"
        )
        
        results = await self.manager.get_intermediate_results(
            execution_state, result_type="analysis"
        )
        
        assert len(results) == 1
        assert results[0].result_id == result1.result_id
        assert results[0].result_type == "analysis"
    
    @pytest.mark.asyncio
    async def test_record_replanning_event(self):
        """Test recording a replanning event."""
        execution_state = ExecutionState(session_id="test-session")
        event = ReplanningEvent(
            trigger_type=ReplanningTriggerType.STEP_FAILURE,
            trigger_description="Step failed",
            new_plan_id="new-plan-123"
        )
        
        await self.manager.record_replanning_event(execution_state, event)
        
        assert len(execution_state.replanning_history) == 1
        assert execution_state.replanning_history[0] == event
        assert execution_state.updated_at is not None
    
    @pytest.mark.asyncio
    async def test_get_execution_summary(self):
        """Test getting execution summary."""
        execution_state = ExecutionState(session_id="test-session")
        execution_state.execution_metrics.total_steps = 5
        execution_state.execution_metrics.completed_steps = 3
        execution_state.execution_metrics.failed_steps = 1
        execution_state.execution_metrics.total_execution_time_seconds = 120.0
        execution_state.execution_metrics.quality_scores = {
            "analysis": 0.9,
            "visualization": 0.8
        }
        
        # Add some intermediate results
        await self.manager.add_intermediate_result(
            execution_state, "step-1", "result-1", "data-1", quality_score=0.9
        )
        await self.manager.add_intermediate_result(
            execution_state, "step-2", "result-2", "data-2", quality_score=0.8
        )
        
        # Add replanning event
        event = ReplanningEvent(
            trigger_type=ReplanningTriggerType.QUALITY_THRESHOLD,
            trigger_description="Quality too low"
        )
        execution_state.replanning_history.append(event)
        
        summary = await self.manager.get_execution_summary(execution_state)
        
        assert summary["session_id"] == "test-session"
        assert summary["completion_percentage"] == 60.0  # 3/5 * 100
        assert summary["total_steps"] == 5
        assert summary["completed_steps"] == 3
        assert summary["failed_steps"] == 1
        assert abs(summary["average_quality_score"] - 0.85) < 0.001  # Handle floating point precision
        assert summary["total_execution_time"] == 120.0
        assert summary["replanning_count"] == 1
        assert summary["escalation_count"] == 0
        assert len(summary["recent_results"]) == 2
    
    @pytest.mark.asyncio
    async def test_update_execution_metrics_completed_step(self):
        """Test updating execution metrics when step completes."""
        execution_state = ExecutionState(session_id="test-session")
        execution_state.execution_metrics.total_steps = 2
        
        # Create a step that completes
        step = ModelsExecutionStep(
            step_id="step-123",
            name="Test Step",
            description="Test",
            status=ExecutionStatus.COMPLETED,  # Changed to COMPLETED since it has completed_at
            started_at=datetime.utcnow() - timedelta(seconds=30),
            completed_at=datetime.utcnow()
        )
        
        await self.manager._update_execution_metrics(
            execution_state, step, ExecutionStatus.RUNNING
        )
        
        metrics = execution_state.execution_metrics
        assert metrics.completed_steps == 1
        assert abs(metrics.total_execution_time_seconds - 30.0) < 0.1  # Allow small timing variance
        assert abs(metrics.average_step_time_seconds - 30.0) < 0.1  # Allow small timing variance
    
    @pytest.mark.asyncio
    async def test_update_execution_metrics_failed_step(self):
        """Test updating execution metrics when step fails."""
        execution_state = ExecutionState(session_id="test-session")
        
        step = ModelsExecutionStep(
            step_id="step-123",
            name="Test Step",
            description="Test",
            status=ExecutionStatus.FAILED
        )
        
        await self.manager._update_execution_metrics(
            execution_state, step, ExecutionStatus.RUNNING
        )
        
        assert execution_state.execution_metrics.failed_steps == 1
    
    @pytest.mark.asyncio
    async def test_update_overall_status_pending(self):
        """Test updating overall status when no steps completed."""
        execution_state = ExecutionState(session_id="test-session")
        execution_state.execution_metrics.total_steps = 3
        
        await self.manager._update_overall_status(execution_state)
        
        assert execution_state.status == ExecutionStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_update_overall_status_running(self):
        """Test updating overall status when some steps completed."""
        execution_state = ExecutionState(session_id="test-session")
        execution_state.execution_metrics.total_steps = 3
        execution_state.execution_metrics.completed_steps = 1
        
        await self.manager._update_overall_status(execution_state)
        
        assert execution_state.status == ExecutionStatus.RUNNING
    
    @pytest.mark.asyncio
    async def test_update_overall_status_completed(self):
        """Test updating overall status when all steps completed."""
        execution_state = ExecutionState(session_id="test-session")
        execution_state.execution_metrics.total_steps = 3
        execution_state.execution_metrics.completed_steps = 3
        
        await self.manager._update_overall_status(execution_state)
        
        assert execution_state.status == ExecutionStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_update_overall_status_failed(self):
        """Test updating overall status when only failures occurred."""
        execution_state = ExecutionState(session_id="test-session")
        execution_state.execution_metrics.total_steps = 3
        execution_state.execution_metrics.failed_steps = 2
        execution_state.execution_metrics.completed_steps = 0
        
        await self.manager._update_overall_status(execution_state)
        
        assert execution_state.status == ExecutionStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_cleanup_old_results_by_count(self):
        """Test cleaning up old results by count limit."""
        execution_state = ExecutionState(session_id="test-session")
        
        # Add more results than the limit
        for i in range(5):
            await self.manager.add_intermediate_result(
                execution_state, f"step-{i}", f"type-{i}", f"data-{i}"
            )
            # Make some results non-preserved
            if i < 3:
                result_id = list(execution_state.intermediate_results.keys())[-1]
                execution_state.intermediate_results[result_id]["preserved"] = False
        
        # Clean up with max_results=3
        cleaned_count = await self.manager.cleanup_old_results(
            execution_state, max_results=3
        )
        
        # Should clean up non-preserved results
        assert cleaned_count > 0
        assert len(execution_state.intermediate_results) <= 5  # Preserved results remain
    
    @pytest.mark.asyncio
    async def test_cleanup_old_results_by_age(self):
        """Test cleaning up old results by age limit."""
        execution_state = ExecutionState(session_id="test-session")
        
        # Add old result
        old_time = datetime.utcnow() - timedelta(hours=25)
        result = await self.manager.add_intermediate_result(
            execution_state, "step-old", "type-old", "data-old"
        )
        # Manually set old timestamp and make non-preserved
        result_data = execution_state.intermediate_results[result.result_id]
        result_data["created_at"] = old_time.isoformat()
        result_data["preserved"] = False
        
        # Add recent result
        await self.manager.add_intermediate_result(
            execution_state, "step-new", "type-new", "data-new"
        )
        
        # Clean up with max_age_hours=24
        cleaned_count = await self.manager.cleanup_old_results(
            execution_state, max_age_hours=24
        )
        
        # Should clean up old non-preserved result
        assert cleaned_count == 1
        assert len(execution_state.intermediate_results) == 1
    
    @pytest.mark.asyncio
    async def test_persist_state(self):
        """Test persisting execution state."""
        execution_state = ExecutionState(session_id="test-session")
        
        result = await self.manager.persist_state(execution_state)
        
        assert result is True
        assert execution_state.updated_at is not None
    
    @pytest.mark.asyncio
    async def test_restore_state(self):
        """Test restoring execution state."""
        # Currently returns None (placeholder implementation)
        result = await self.manager.restore_state("test-session")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_step_status_update_flow(self):
        """Test complete step status update flow."""
        execution_state = ExecutionState(session_id="test-session")
        execution_state.execution_metrics.total_steps = 1
        
        # Start step
        await self.manager.update_step_status(
            execution_state, "step-123", ExecutionStatus.RUNNING
        )
        assert execution_state.status == ExecutionStatus.RUNNING
        
        # Complete step
        await self.manager.update_step_status(
            execution_state, "step-123", ExecutionStatus.COMPLETED,
            outputs={"result": "success"}
        )
        assert execution_state.status == ExecutionStatus.COMPLETED
        assert execution_state.execution_metrics.completed_steps == 1
    
    @pytest.mark.asyncio
    async def test_intermediate_results_sorting(self):
        """Test that intermediate results are sorted by creation time."""
        execution_state = ExecutionState(session_id="test-session")
        
        # Add results with different timestamps
        result1 = await self.manager.add_intermediate_result(
            execution_state, "step-1", "type-1", "data-1"
        )
        
        # Simulate time passing
        import time
        time.sleep(0.01)
        
        result2 = await self.manager.add_intermediate_result(
            execution_state, "step-2", "type-2", "data-2"
        )
        
        results = await self.manager.get_intermediate_results(execution_state)
        
        # Should be sorted by creation time (oldest first)
        assert len(results) == 2
        assert results[0].result_id == result1.result_id
        assert results[1].result_id == result2.result_id
        assert results[0].created_at <= results[1].created_at