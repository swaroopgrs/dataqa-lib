"""
Execution state management with intermediate result tracking.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import ExecutionState, ExecutionStatus, ExecutionMetrics
from .models import ExecutionStep, IntermediateResult, Plan, ReplanningEvent

logger = logging.getLogger(__name__)


class ExecutionStateManager:
    """
    Manages execution state with intermediate result tracking.
    
    Provides centralized management of execution state, including step tracking,
    intermediate result storage, metrics collection, and state persistence.
    """
    
    def __init__(self):
        """Initialize the execution state manager."""
        self.logger = logging.getLogger(__name__)
    
    async def create_execution_state(
        self,
        session_id: str,
        plan: Plan
    ) -> ExecutionState:
        """
        Create a new execution state for a plan.
        
        Args:
            session_id: Unique session identifier
            plan: Execution plan to track
            
        Returns:
            New execution state
        """
        execution_state = ExecutionState(
            session_id=session_id,
            current_plan_id=plan.plan_id,
            status=ExecutionStatus.PENDING,
            started_at=datetime.utcnow()
        )
        
        # Initialize metrics based on plan
        execution_state.execution_metrics.total_steps = len(plan.steps)
        
        self.logger.info(f"Created execution state for session {session_id}")
        return execution_state
    
    async def update_step_status(
        self,
        execution_state: ExecutionState,
        step_id: str,
        status: ExecutionStatus,
        error_message: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the status of an execution step.
        
        Args:
            execution_state: Current execution state
            step_id: ID of the step to update
            status: New status
            error_message: Error message if failed
            outputs: Step outputs if completed
        """
        # Find existing step or create new one
        step = None
        for existing_step in execution_state.completed_steps:
            if existing_step.step_id == step_id:
                step = existing_step
                break
        
        if not step:
            # Create new step record
            step = ExecutionStep(
                step_id=step_id,
                name=f"Step {step_id}",
                description=f"Execution step {step_id}",
                status=status
            )
            execution_state.completed_steps.append(step)
        
        # Update step details
        previous_status = step.status
        step.status = status
        
        if status == ExecutionStatus.RUNNING and not step.started_at:
            step.started_at = datetime.utcnow()
        elif status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]:
            step.completed_at = datetime.utcnow()
            if outputs:
                step.outputs.update(outputs)
            if error_message:
                step.error_message = error_message
        
        # Update execution metrics
        await self._update_execution_metrics(execution_state, step, previous_status)
        
        # Update overall execution state status
        await self._update_overall_status(execution_state)
        
        self.logger.info(f"Updated step {step_id} status: {previous_status} -> {status}")
    
    async def add_intermediate_result(
        self,
        execution_state: ExecutionState,
        step_id: str,
        result_type: str,
        data: Any,
        quality_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IntermediateResult:
        """
        Add an intermediate result to the execution state.
        
        Args:
            execution_state: Current execution state
            step_id: ID of the step that produced the result
            result_type: Type of result
            data: Result data
            quality_score: Quality score (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            Created intermediate result
        """
        result = IntermediateResult(
            step_id=step_id,
            result_type=result_type,
            data=data,
            quality_score=quality_score,
            metadata=metadata or {}
        )
        
        # Store in execution state
        execution_state.intermediate_results[result.result_id] = {
            "result_id": result.result_id,
            "step_id": step_id,
            "result_type": result_type,
            "data": data,
            "quality_score": quality_score,
            "metadata": metadata or {},
            "created_at": result.created_at.isoformat(),
            "preserved": result.preserved
        }
        
        # Update quality metrics
        if quality_score is not None:
            execution_state.execution_metrics.quality_scores[result_type] = quality_score
        
        self.logger.info(f"Added intermediate result {result.result_id} for step {step_id}")
        return result
    
    async def get_intermediate_results(
        self,
        execution_state: ExecutionState,
        step_id: Optional[str] = None,
        result_type: Optional[str] = None
    ) -> List[IntermediateResult]:
        """
        Get intermediate results with optional filtering.
        
        Args:
            execution_state: Current execution state
            step_id: Filter by step ID
            result_type: Filter by result type
            
        Returns:
            List of matching intermediate results
        """
        results = []
        
        for result_data in execution_state.intermediate_results.values():
            # Apply filters
            if step_id and result_data.get("step_id") != step_id:
                continue
            if result_type and result_data.get("result_type") != result_type:
                continue
            
            # Convert back to IntermediateResult object
            result = IntermediateResult(
                result_id=result_data["result_id"],
                step_id=result_data["step_id"],
                result_type=result_data["result_type"],
                data=result_data["data"],
                quality_score=result_data.get("quality_score"),
                metadata=result_data.get("metadata", {}),
                created_at=datetime.fromisoformat(result_data["created_at"]),
                preserved=result_data.get("preserved", True)
            )
            results.append(result)
        
        return sorted(results, key=lambda r: r.created_at)
    
    async def record_replanning_event(
        self,
        execution_state: ExecutionState,
        event: ReplanningEvent
    ) -> None:
        """
        Record a replanning event in the execution state.
        
        Args:
            execution_state: Current execution state
            event: Replanning event to record
        """
        execution_state.replanning_history.append(event)
        execution_state.updated_at = datetime.utcnow()
        
        self.logger.info(f"Recorded replanning event {event.event_id}")
    
    async def get_execution_summary(
        self,
        execution_state: ExecutionState
    ) -> Dict[str, Any]:
        """
        Get a summary of the current execution state.
        
        Args:
            execution_state: Current execution state
            
        Returns:
            Execution summary
        """
        metrics = execution_state.execution_metrics
        
        # Calculate completion percentage
        completion_percentage = 0.0
        if metrics.total_steps > 0:
            completion_percentage = (metrics.completed_steps / metrics.total_steps) * 100
        
        # Calculate average quality score
        avg_quality = 0.0
        if metrics.quality_scores:
            avg_quality = sum(metrics.quality_scores.values()) / len(metrics.quality_scores)
        
        # Get recent intermediate results
        recent_results = await self.get_intermediate_results(execution_state)
        recent_results = recent_results[-5:]  # Last 5 results
        
        summary = {
            "session_id": execution_state.session_id,
            "status": execution_state.status.value,
            "completion_percentage": completion_percentage,
            "total_steps": metrics.total_steps,
            "completed_steps": metrics.completed_steps,
            "failed_steps": metrics.failed_steps,
            "average_quality_score": avg_quality,
            "total_execution_time": metrics.total_execution_time_seconds,
            "replanning_count": len(execution_state.replanning_history),
            "escalation_count": len(execution_state.escalation_points),
            "recent_results": [
                {
                    "result_type": r.result_type,
                    "quality_score": r.quality_score,
                    "created_at": r.created_at.isoformat()
                }
                for r in recent_results
            ],
            "started_at": execution_state.started_at.isoformat() if execution_state.started_at else None,
            "updated_at": execution_state.updated_at.isoformat()
        }
        
        return summary
    
    async def _update_execution_metrics(
        self,
        execution_state: ExecutionState,
        step: ExecutionStep,
        previous_status: ExecutionStatus
    ) -> None:
        """Update execution metrics based on step status change."""
        metrics = execution_state.execution_metrics
        
        # Update step counts
        if previous_status != ExecutionStatus.COMPLETED and step.status == ExecutionStatus.COMPLETED:
            metrics.completed_steps += 1
        elif previous_status != ExecutionStatus.FAILED and step.status == ExecutionStatus.FAILED:
            metrics.failed_steps += 1
        
        # Update timing metrics
        if step.started_at and step.completed_at:
            step_duration = (step.completed_at - step.started_at).total_seconds()
            metrics.total_execution_time_seconds += step_duration
            
            # Update average step time
            completed_steps = metrics.completed_steps + metrics.failed_steps
            if completed_steps > 0:
                metrics.average_step_time_seconds = (
                    metrics.total_execution_time_seconds / completed_steps
                )
        
        # Update resource utilization (placeholder - would be populated by actual monitoring)
        if step.outputs:
            if "cpu_usage" in step.outputs:
                metrics.resource_utilization["cpu"] = step.outputs["cpu_usage"]
            if "memory_usage" in step.outputs:
                metrics.resource_utilization["memory"] = step.outputs["memory_usage"]
    
    async def _update_overall_status(self, execution_state: ExecutionState) -> None:
        """Update the overall execution status based on step statuses."""
        metrics = execution_state.execution_metrics
        
        # Check if there are any running steps
        running_steps = sum(1 for step in execution_state.completed_steps 
                          if step.status == ExecutionStatus.RUNNING)
        
        # Determine overall status
        if metrics.failed_steps > 0 and metrics.completed_steps == 0 and running_steps == 0:
            execution_state.status = ExecutionStatus.FAILED
        elif metrics.completed_steps == metrics.total_steps:
            execution_state.status = ExecutionStatus.COMPLETED
        elif metrics.completed_steps > 0 or metrics.failed_steps > 0 or running_steps > 0:
            execution_state.status = ExecutionStatus.RUNNING
        else:
            execution_state.status = ExecutionStatus.PENDING
        
        execution_state.updated_at = datetime.utcnow()
    
    async def cleanup_old_results(
        self,
        execution_state: ExecutionState,
        max_results: int = 100,
        max_age_hours: int = 24
    ) -> int:
        """
        Clean up old intermediate results to manage memory usage.
        
        Args:
            execution_state: Current execution state
            max_results: Maximum number of results to keep
            max_age_hours: Maximum age of results in hours
            
        Returns:
            Number of results cleaned up
        """
        current_time = datetime.utcnow()
        results_to_remove = []
        
        # Convert to list for processing
        all_results = list(execution_state.intermediate_results.items())
        
        # Sort by creation time (oldest first)
        all_results.sort(key=lambda x: x[1].get("created_at", ""))
        
        # Remove old results
        for result_id, result_data in all_results:
            created_at_str = result_data.get("created_at")
            if created_at_str:
                created_at = datetime.fromisoformat(created_at_str)
                age_hours = (current_time - created_at).total_seconds() / 3600
                
                # Remove if too old or if we have too many results
                if (age_hours > max_age_hours or 
                    len(all_results) - len(results_to_remove) > max_results):
                    # Don't remove preserved results
                    if not result_data.get("preserved", True):
                        results_to_remove.append(result_id)
        
        # Remove identified results
        for result_id in results_to_remove:
            del execution_state.intermediate_results[result_id]
        
        if results_to_remove:
            self.logger.info(f"Cleaned up {len(results_to_remove)} old intermediate results")
        
        return len(results_to_remove)
    
    async def persist_state(
        self,
        execution_state: ExecutionState,
        storage_backend: Optional[Any] = None
    ) -> bool:
        """
        Persist execution state to storage backend.
        
        Args:
            execution_state: State to persist
            storage_backend: Storage backend (placeholder for future implementation)
            
        Returns:
            True if successful, False otherwise
        """
        # Placeholder for persistence implementation
        # In production, this would save to database, file system, etc.
        
        execution_state.updated_at = datetime.utcnow()
        self.logger.info(f"Persisted execution state for session {execution_state.session_id}")
        return True
    
    async def restore_state(
        self,
        session_id: str,
        storage_backend: Optional[Any] = None
    ) -> Optional[ExecutionState]:
        """
        Restore execution state from storage backend.
        
        Args:
            session_id: Session ID to restore
            storage_backend: Storage backend (placeholder for future implementation)
            
        Returns:
            Restored execution state or None if not found
        """
        # Placeholder for restoration implementation
        # In production, this would load from database, file system, etc.
        
        self.logger.info(f"Attempted to restore execution state for session {session_id}")
        return None