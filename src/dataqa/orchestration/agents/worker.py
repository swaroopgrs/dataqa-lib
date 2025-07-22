"""
Worker agent implementation for hierarchical multi-agent orchestration.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field

from .base import BaseAgent, Task, TaskResult, ExecutionContext, ProgressUpdate, AssistanceRequest
from ..models import ExecutionStatus

if TYPE_CHECKING:
    from .manager import ManagerAgent


class WorkerAgent(BaseAgent):
    """
    Worker agent that executes specific tasks within its capabilities.
    
    Responsibilities:
    - Execute assigned tasks using available capabilities
    - Report progress to manager agents
    - Request assistance when encountering issues
    - Maintain execution history and performance metrics
    """
    
    def __init__(self, config):
        """
        Initialize worker agent.
        
        Args:
            config: Agent configuration
        """
        super().__init__(config)
        self.manager: Optional['ManagerAgent'] = None
        
        # Worker-specific state
        self._execution_history: List[Dict[str, Any]] = []
        self._performance_metrics: Dict[str, float] = {
            "average_execution_time": 0.0,
            "success_rate": 1.0,
            "tasks_completed": 0,
            "tasks_failed": 0
        }
        self._assistance_requests: List[AssistanceRequest] = []
    
    def set_manager(self, manager: 'ManagerAgent') -> None:
        """Set the manager agent for this worker."""
        self.manager = manager
    
    async def execute_task(self, task: Task, context: ExecutionContext) -> TaskResult:
        """
        Execute a task using the worker's capabilities.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            Task execution result
        """
        start_time = datetime.utcnow()
        
        try:
            # Start task execution
            await self.start_task(task, context)
            
            # Report initial progress
            await self.report_progress(ProgressUpdate(
                agent_id=self.agent_id,
                task_id=task.task_id,
                progress_percentage=0.0,
                status_message="Task execution started"
            ))
            
            # Execute task based on required capabilities
            result = await self._execute_task_by_capability(task, context)
            
            # Calculate execution time
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            result.execution_time_seconds = execution_time
            
            # Update performance metrics
            self._update_performance_metrics(result, execution_time)
            
            # Record execution history
            self._execution_history.append({
                "task_id": task.task_id,
                "task_name": task.name,
                "status": result.status if isinstance(result.status, str) else result.status.value,
                "execution_time": execution_time,
                "timestamp": end_time.isoformat(),
                "capabilities_used": task.required_capabilities
            })
            
            # Report final progress
            await self.report_progress(ProgressUpdate(
                agent_id=self.agent_id,
                task_id=task.task_id,
                progress_percentage=100.0,
                status_message=f"Task completed with status: {result.status if isinstance(result.status, str) else result.status.value}",
                intermediate_results=result.outputs
            ))
            
            # Complete task
            await self.complete_task(task.task_id, result)
            return result
            
        except Exception as e:
            # Handle execution error
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            result = TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=ExecutionStatus.FAILED,
                error_message=str(e),
                execution_time_seconds=execution_time
            )
            
            # Update performance metrics
            self._update_performance_metrics(result, execution_time)
            
            # Record execution history
            self._execution_history.append({
                "task_id": task.task_id,
                "task_name": task.name,
                "status": "failed",
                "error": str(e),
                "execution_time": execution_time,
                "timestamp": end_time.isoformat(),
                "capabilities_used": task.required_capabilities
            })
            
            # Complete task with error
            await self.complete_task(task.task_id, result)
            return result
    
    async def _execute_task_by_capability(self, task: Task, context: ExecutionContext) -> TaskResult:
        """
        Execute task based on the primary required capability.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            Task execution result
        """
        # Determine primary capability needed
        primary_capability = task.required_capabilities[0] if task.required_capabilities else None
        
        if not primary_capability:
            raise ValueError(f"Task {task.task_id} has no required capabilities specified")
        
        if not self.has_capability(primary_capability):
            raise ValueError(f"Agent {self.agent_id} does not have required capability: {primary_capability}")
        
        # Execute based on capability type
        if primary_capability == "data_retrieval":
            return await self._execute_data_retrieval_task(task, context)
        elif primary_capability == "data_analysis":
            return await self._execute_data_analysis_task(task, context)
        elif primary_capability == "visualization":
            return await self._execute_visualization_task(task, context)
        elif primary_capability == "code_generation":
            return await self._execute_code_generation_task(task, context)
        elif primary_capability == "domain_expertise":
            return await self._execute_domain_expertise_task(task, context)
        else:
            return await self._execute_generic_task(task, context)
    
    async def _execute_data_retrieval_task(self, task: Task, context: ExecutionContext) -> TaskResult:
        """Execute a data retrieval task."""
        # Simulate data retrieval execution
        await self._simulate_work_progress(task.task_id, "Retrieving data")
        
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=ExecutionStatus.COMPLETED,
            outputs={
                "data_retrieved": True,
                "records_count": 1000,
                "data_source": task.inputs.get("source", "unknown"),
                "retrieval_method": "simulated"
            },
            metadata={"capability_used": "data_retrieval"}
        )
    
    async def _execute_data_analysis_task(self, task: Task, context: ExecutionContext) -> TaskResult:
        """Execute a data analysis task."""
        # Simulate data analysis execution
        await self._simulate_work_progress(task.task_id, "Analyzing data")
        
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=ExecutionStatus.COMPLETED,
            outputs={
                "analysis_completed": True,
                "insights_generated": 5,
                "analysis_type": task.inputs.get("analysis_type", "descriptive"),
                "confidence_score": 0.85
            },
            metadata={"capability_used": "data_analysis"}
        )
    
    async def _execute_visualization_task(self, task: Task, context: ExecutionContext) -> TaskResult:
        """Execute a visualization task."""
        # Simulate visualization creation
        await self._simulate_work_progress(task.task_id, "Creating visualization")
        
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=ExecutionStatus.COMPLETED,
            outputs={
                "visualization_created": True,
                "chart_type": task.inputs.get("chart_type", "bar"),
                "data_points": 50,
                "format": "png"
            },
            metadata={"capability_used": "visualization"}
        )
    
    async def _execute_code_generation_task(self, task: Task, context: ExecutionContext) -> TaskResult:
        """Execute a code generation task."""
        # Simulate code generation
        await self._simulate_work_progress(task.task_id, "Generating code")
        
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=ExecutionStatus.COMPLETED,
            outputs={
                "code_generated": True,
                "language": task.inputs.get("language", "python"),
                "lines_of_code": 150,
                "functions_created": 3
            },
            metadata={"capability_used": "code_generation"}
        )
    
    async def _execute_domain_expertise_task(self, task: Task, context: ExecutionContext) -> TaskResult:
        """Execute a domain expertise task."""
        # Simulate domain-specific analysis
        await self._simulate_work_progress(task.task_id, "Applying domain expertise")
        
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=ExecutionStatus.COMPLETED,
            outputs={
                "expertise_applied": True,
                "domain": self.specialization or "general",
                "recommendations": 3,
                "compliance_checked": True
            },
            metadata={"capability_used": "domain_expertise"}
        )
    
    async def _execute_generic_task(self, task: Task, context: ExecutionContext) -> TaskResult:
        """Execute a generic task."""
        # Simulate generic task execution
        await self._simulate_work_progress(task.task_id, "Executing task")
        
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=ExecutionStatus.COMPLETED,
            outputs={
                "task_completed": True,
                "processing_time": 2.5,
                "result_quality": "good"
            },
            metadata={"capability_used": "generic"}
        )
    
    async def _simulate_work_progress(self, task_id: str, work_description: str) -> None:
        """Simulate work progress with periodic updates."""
        # Simulate progress reporting
        progress_steps = [25.0, 50.0, 75.0]
        
        for progress in progress_steps:
            await self.report_progress(ProgressUpdate(
                agent_id=self.agent_id,
                task_id=task_id,
                progress_percentage=progress,
                status_message=f"{work_description} - {progress}% complete"
            ))
    
    def _update_performance_metrics(self, result: TaskResult, execution_time: float) -> None:
        """Update performance metrics based on task result."""
        # Update task counts
        if result.status == ExecutionStatus.COMPLETED:
            self._performance_metrics["tasks_completed"] += 1
        else:
            self._performance_metrics["tasks_failed"] += 1
        
        # Update success rate
        total_tasks = self._performance_metrics["tasks_completed"] + self._performance_metrics["tasks_failed"]
        self._performance_metrics["success_rate"] = self._performance_metrics["tasks_completed"] / total_tasks
        
        # Update average execution time
        current_avg = self._performance_metrics["average_execution_time"]
        self._performance_metrics["average_execution_time"] = (
            (current_avg * (total_tasks - 1) + execution_time) / total_tasks
        )
    
    async def report_progress(self, progress: ProgressUpdate) -> None:
        """
        Report progress to manager agent.
        
        Args:
            progress: Progress update information
        """
        # Update last heartbeat
        await super().report_progress(progress)
        
        # If we have a manager, report progress to them
        if self.manager:
            await self.manager.receive_progress_update(progress)
    
    async def request_assistance(self, assistance_request: AssistanceRequest) -> None:
        """
        Request assistance from manager or peer agents.
        
        Args:
            assistance_request: Details of assistance needed
        """
        self._assistance_requests.append(assistance_request)
        
        # If we have a manager, escalate the assistance request
        if self.manager:
            # In a real implementation, this would create an escalation to the manager
            pass
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of worker performance."""
        return {
            **self.get_status_summary(),
            "performance_metrics": self._performance_metrics.copy(),
            "execution_history_count": len(self._execution_history),
            "assistance_requests_count": len(self._assistance_requests),
            "has_manager": self.manager is not None,
            "manager_id": self.manager.agent_id if self.manager else None
        }