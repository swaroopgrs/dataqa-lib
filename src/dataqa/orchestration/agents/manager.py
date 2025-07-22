"""
Manager agent implementation for hierarchical multi-agent orchestration.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from .base import BaseAgent, Task, TaskResult, ExecutionContext, ProgressUpdate, AssistanceRequest
from .worker import WorkerAgent
from ..models import ExecutionStatus, TaskAssignment


class DelegationStrategy(str, Enum):
    """Strategies for task delegation."""
    ROUND_ROBIN = "round_robin"
    CAPABILITY_BASED = "capability_based"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_BASED = "priority_based"


class CoordinationProtocol(str, Enum):
    """Protocols for coordinating subordinate agents."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    ADAPTIVE = "adaptive"


class Escalation(BaseModel):
    """Escalation from a subordinate agent."""
    escalation_id: str = Field(default_factory=lambda: str(uuid4()))
    from_agent_id: str
    task_id: str
    escalation_type: str
    reason: str
    context: Dict[str, Any] = Field(default_factory=dict)
    severity: str = "medium"  # "low", "medium", "high", "critical"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Resolution(BaseModel):
    """Resolution for an escalation."""
    escalation_id: str
    resolution_type: str
    action_taken: str
    new_assignment: Optional[TaskAssignment] = None
    additional_resources: Dict[str, Any] = Field(default_factory=dict)
    resolved_at: datetime = Field(default_factory=datetime.utcnow)


class ManagerAgent(BaseAgent):
    """
    Manager agent that coordinates and delegates tasks to subordinate worker agents.
    
    Responsibilities:
    - Task delegation based on agent capabilities and availability
    - Coordination of multi-agent workflows
    - Escalation handling and conflict resolution
    - Resource allocation and load balancing
    """
    
    def __init__(self, config, subordinates: Optional[List[WorkerAgent]] = None):
        """
        Initialize manager agent.
        
        Args:
            config: Agent configuration
            subordinates: List of subordinate worker agents
        """
        super().__init__(config)
        self.subordinates: List[WorkerAgent] = subordinates or []
        self.delegation_strategy = DelegationStrategy.CAPABILITY_BASED
        self.coordination_protocol = CoordinationProtocol.ADAPTIVE
        
        # Manager-specific state
        self._task_assignments: Dict[str, TaskAssignment] = {}
        self._escalations: Dict[str, Escalation] = {}
        self._subordinate_status: Dict[str, Dict[str, Any]] = {}
        self._delegation_history: List[Dict[str, Any]] = []
    
    def add_subordinate(self, worker: WorkerAgent) -> None:
        """Add a subordinate worker agent."""
        if worker not in self.subordinates:
            self.subordinates.append(worker)
            worker.set_manager(self)
            self._subordinate_status[worker.agent_id] = worker.get_status_summary()
    
    def remove_subordinate(self, worker_id: str) -> None:
        """Remove a subordinate worker agent."""
        self.subordinates = [w for w in self.subordinates if w.agent_id != worker_id]
        if worker_id in self._subordinate_status:
            del self._subordinate_status[worker_id]
    
    def get_available_subordinates(self) -> List[WorkerAgent]:
        """Get list of available subordinate agents."""
        return [worker for worker in self.subordinates if worker.is_available]
    
    def find_capable_subordinates(self, required_capabilities: List[str]) -> List[WorkerAgent]:
        """
        Find subordinates that have the required capabilities.
        
        Args:
            required_capabilities: List of required capability types
            
        Returns:
            List of capable subordinate agents
        """
        capable_agents = []
        for worker in self.subordinates:
            if worker.is_available and all(worker.has_capability(cap) for cap in required_capabilities):
                capable_agents.append(worker)
        return capable_agents
    
    async def delegate_task(self, task: Task) -> TaskAssignment:
        """
        Delegate a task to an appropriate subordinate agent.
        
        Args:
            task: Task to delegate
            
        Returns:
            Task assignment details
            
        Raises:
            ValueError: If no suitable subordinate is available
        """
        # Find capable subordinates
        capable_agents = self.find_capable_subordinates(task.required_capabilities)
        
        if not capable_agents:
            raise ValueError(f"No capable subordinates available for task {task.task_id}")
        
        # Select agent based on delegation strategy
        selected_agent = self._select_agent_for_delegation(capable_agents, task)
        
        # Create task assignment
        assignment = TaskAssignment(
            task_id=task.task_id,
            agent_id=selected_agent.agent_id,
            priority=task.priority,
            deadline=task.deadline,
            context=task.context
        )
        
        # Store assignment
        self._task_assignments[task.task_id] = assignment
        
        # Record delegation history
        self._delegation_history.append({
            "task_id": task.task_id,
            "assigned_to": selected_agent.agent_id,
            "strategy": self.delegation_strategy.value,
            "timestamp": datetime.utcnow().isoformat(),
            "available_agents": len(capable_agents)
        })
        
        return assignment
    
    def _select_agent_for_delegation(self, capable_agents: List[WorkerAgent], task: Task) -> WorkerAgent:
        """
        Select the best agent for task delegation based on strategy.
        
        Args:
            capable_agents: List of capable agents
            task: Task to assign
            
        Returns:
            Selected worker agent
        """
        if self.delegation_strategy == DelegationStrategy.ROUND_ROBIN:
            # Simple round-robin selection
            return capable_agents[len(self._delegation_history) % len(capable_agents)]
        
        elif self.delegation_strategy == DelegationStrategy.LOAD_BALANCED:
            # Select agent with lowest current load
            return min(capable_agents, key=lambda a: a.active_task_count)
        
        elif self.delegation_strategy == DelegationStrategy.PRIORITY_BASED:
            # Select agent with highest priority level
            return min(capable_agents, key=lambda a: a.priority_level)
        
        else:  # CAPABILITY_BASED (default)
            # Select agent with most specific capabilities for this task
            best_agent = capable_agents[0]
            best_score = 0
            
            for agent in capable_agents:
                # Score based on capability match and specialization
                score = 0
                for req_cap in task.required_capabilities:
                    if agent.has_capability(req_cap):
                        score += 1
                
                # Bonus for specialization match
                if agent.specialization and task.context.get("domain") == agent.specialization:
                    score += 2
                
                # Penalty for high load
                score -= agent.active_task_count * 0.5
                
                if score > best_score:
                    best_score = score
                    best_agent = agent
            
            return best_agent
    
    async def coordinate_execution(self, assignments: List[TaskAssignment]) -> List[TaskResult]:
        """
        Coordinate execution of multiple task assignments.
        
        Args:
            assignments: List of task assignments to coordinate
            
        Returns:
            List of task execution results
        """
        results = []
        
        if self.coordination_protocol == CoordinationProtocol.SEQUENTIAL:
            # Execute tasks sequentially
            for assignment in assignments:
                result = await self._execute_assignment(assignment)
                results.append(result)
        
        elif self.coordination_protocol == CoordinationProtocol.PARALLEL:
            # Execute tasks in parallel (simplified - would use asyncio.gather in real implementation)
            import asyncio
            tasks = [self._execute_assignment(assignment) for assignment in assignments]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to failed TaskResults
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    results[i] = TaskResult(
                        task_id=assignments[i].task_id,
                        agent_id=assignments[i].agent_id,
                        status=ExecutionStatus.FAILED,
                        error_message=str(result)
                    )
        
        else:  # ADAPTIVE or PIPELINE
            # Adaptive coordination based on task dependencies and agent availability
            remaining_assignments = assignments.copy()
            
            while remaining_assignments:
                # Find assignments that can be executed now
                ready_assignments = [a for a in remaining_assignments if self._is_assignment_ready(a, results)]
                
                if not ready_assignments:
                    # No assignments ready - might indicate dependency cycle or resource constraints
                    break
                
                # Execute ready assignments in parallel if possible
                if len(ready_assignments) > 1:
                    import asyncio
                    batch_tasks = [self._execute_assignment(assignment) for assignment in ready_assignments]
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Process batch results
                    for i, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            result = TaskResult(
                                task_id=ready_assignments[i].task_id,
                                agent_id=ready_assignments[i].agent_id,
                                status=ExecutionStatus.FAILED,
                                error_message=str(result)
                            )
                        results.append(result)
                        remaining_assignments.remove(ready_assignments[i])
                else:
                    # Execute single assignment
                    result = await self._execute_assignment(ready_assignments[0])
                    results.append(result)
                    remaining_assignments.remove(ready_assignments[0])
        
        return results
    
    async def _execute_assignment(self, assignment: TaskAssignment) -> TaskResult:
        """Execute a single task assignment."""
        # Find the assigned agent
        assigned_agent = next((a for a in self.subordinates if a.agent_id == assignment.agent_id), None)
        
        if not assigned_agent:
            return TaskResult(
                task_id=assignment.task_id,
                agent_id=assignment.agent_id,
                status=ExecutionStatus.FAILED,
                error_message=f"Assigned agent {assignment.agent_id} not found"
            )
        
        # Create task from assignment
        task = Task(
            task_id=assignment.task_id,
            name=f"Delegated task {assignment.task_id}",
            description="Task delegated by manager",
            priority=assignment.priority,
            deadline=assignment.deadline,
            context=assignment.context
        )
        
        # Create execution context
        context = ExecutionContext(
            session_id=assignment.context.get("session_id", "unknown"),
            workflow_id=assignment.context.get("workflow_id", "unknown"),
            domain_context=assignment.context.get("domain_context"),
            metadata=assignment.context
        )
        
        # Execute task through subordinate
        try:
            result = await assigned_agent.execute_task(task, context)
            assignment.status = result.status
            return result
        except Exception as e:
            assignment.status = ExecutionStatus.FAILED
            return TaskResult(
                task_id=assignment.task_id,
                agent_id=assignment.agent_id,
                status=ExecutionStatus.FAILED,
                error_message=str(e)
            )
    
    def _is_assignment_ready(self, assignment: TaskAssignment, completed_results: List[TaskResult]) -> bool:
        """Check if an assignment is ready to execute based on dependencies."""
        # Simplified dependency checking - would be more sophisticated in real implementation
        return True
    
    async def handle_escalation(self, escalation: Escalation) -> Resolution:
        """
        Handle escalation from a subordinate agent.
        
        Args:
            escalation: Escalation details
            
        Returns:
            Resolution for the escalation
        """
        self._escalations[escalation.escalation_id] = escalation
        
        # Determine resolution strategy based on escalation type and severity
        if escalation.escalation_type == "resource_shortage":
            return await self._handle_resource_escalation(escalation)
        elif escalation.escalation_type == "capability_gap":
            return await self._handle_capability_escalation(escalation)
        elif escalation.escalation_type == "quality_issue":
            return await self._handle_quality_escalation(escalation)
        else:
            return await self._handle_generic_escalation(escalation)
    
    async def _handle_resource_escalation(self, escalation: Escalation) -> Resolution:
        """Handle resource shortage escalation."""
        # Try to reallocate resources or reassign task
        return Resolution(
            escalation_id=escalation.escalation_id,
            resolution_type="resource_reallocation",
            action_taken="Attempted resource reallocation"
        )
    
    async def _handle_capability_escalation(self, escalation: Escalation) -> Resolution:
        """Handle capability gap escalation."""
        # Try to find alternative agent or request human intervention
        return Resolution(
            escalation_id=escalation.escalation_id,
            resolution_type="capability_substitution",
            action_taken="Searched for alternative capable agent"
        )
    
    async def _handle_quality_escalation(self, escalation: Escalation) -> Resolution:
        """Handle quality issue escalation."""
        # Review and potentially retry with different parameters
        return Resolution(
            escalation_id=escalation.escalation_id,
            resolution_type="quality_review",
            action_taken="Initiated quality review process"
        )
    
    async def _handle_generic_escalation(self, escalation: Escalation) -> Resolution:
        """Handle generic escalation."""
        return Resolution(
            escalation_id=escalation.escalation_id,
            resolution_type="manual_review",
            action_taken="Escalated to human oversight"
        )
    
    async def execute_task(self, task: Task, context: ExecutionContext) -> TaskResult:
        """
        Execute a task by delegating to subordinates and coordinating execution.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            Task execution result
        """
        try:
            # Start task execution
            await self.start_task(task, context)
            
            # Delegate task to subordinate
            assignment = await self.delegate_task(task)
            
            # Coordinate execution
            results = await self.coordinate_execution([assignment])
            
            # Aggregate results
            if results and results[0].status == ExecutionStatus.COMPLETED:
                result = TaskResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status=ExecutionStatus.COMPLETED,
                    outputs=results[0].outputs,
                    execution_time_seconds=results[0].execution_time_seconds,
                    metadata={"delegated_to": assignment.agent_id, "delegation_results": results}
                )
            else:
                result = TaskResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status=ExecutionStatus.FAILED,
                    error_message="Delegation failed or subordinate execution failed",
                    metadata={"delegation_results": results}
                )
            
            # Complete task
            await self.complete_task(task.task_id, result)
            return result
            
        except Exception as e:
            result = TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=ExecutionStatus.FAILED,
                error_message=str(e)
            )
            await self.complete_task(task.task_id, result)
            return result
    
    async def receive_progress_update(self, progress: ProgressUpdate) -> None:
        """
        Receive progress update from a subordinate agent.
        
        Args:
            progress: Progress update from subordinate
        """
        # Update subordinate status
        if progress.agent_id in self._subordinate_status:
            self._subordinate_status[progress.agent_id].update({
                "last_progress": progress.progress_percentage,
                "last_status": progress.status_message,
                "last_update": progress.timestamp.isoformat()
            })
        
        # Update task assignment status if applicable
        if progress.task_id in self._task_assignments:
            assignment = self._task_assignments[progress.task_id]
            if progress.progress_percentage >= 100.0:
                assignment.status = ExecutionStatus.COMPLETED
            elif progress.progress_percentage > 0:
                assignment.status = ExecutionStatus.RUNNING
    
    def get_management_summary(self) -> Dict[str, Any]:
        """Get summary of management activities."""
        return {
            **self.get_status_summary(),
            "subordinate_count": len(self.subordinates),
            "available_subordinates": len(self.get_available_subordinates()),
            "active_assignments": len([a for a in self._task_assignments.values() if a.status == ExecutionStatus.RUNNING]),
            "total_assignments": len(self._task_assignments),
            "escalations_handled": len(self._escalations),
            "delegation_strategy": self.delegation_strategy.value if hasattr(self.delegation_strategy, 'value') else self.delegation_strategy,
            "coordination_protocol": self.coordination_protocol.value if hasattr(self.coordination_protocol, 'value') else self.coordination_protocol
        }