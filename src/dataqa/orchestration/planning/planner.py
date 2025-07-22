"""
Adaptive planner for dynamic multi-agent workflow planning.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from ..models import CapabilityType, ExecutionStatus
from .models import (
    ExecutionContext,
    ExecutionStep,
    IntermediateResult,
    Plan,
    ReplanningEvent,
    ReplanningTriggerType,
)

logger = logging.getLogger(__name__)


class PlanningStrategy:
    """Base class for planning strategies."""
    
    def estimate_step_duration(self, step: ExecutionStep, context: ExecutionContext) -> int:
        """Estimate duration for a step in minutes."""
        # Default estimation based on capability type
        duration_map = {
            CapabilityType.DATA_RETRIEVAL: 2,
            CapabilityType.DATA_ANALYSIS: 5,
            CapabilityType.VISUALIZATION: 3,
            CapabilityType.CODE_GENERATION: 4,
            CapabilityType.DOMAIN_EXPERTISE: 3,
            CapabilityType.COORDINATION: 1,
            CapabilityType.APPROVAL: 10,  # Human approval takes longer
        }
        return duration_map.get(step.capability_required, 3)
    
    def validate_dependencies(self, steps: List[ExecutionStep], dependencies: Dict[str, List[str]]) -> bool:
        """Validate that dependencies form a valid DAG."""
        step_ids = {step.step_id for step in steps}
        
        # Check all dependency references exist
        for step_id, deps in dependencies.items():
            if step_id not in step_ids:
                return False
            for dep in deps:
                if dep not in step_ids:
                    return False
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependencies.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for step_id in step_ids:
            if step_id not in visited:
                if has_cycle(step_id):
                    return False
        
        return True


class AdaptivePlanner:
    """
    Adaptive planner for generating and modifying execution plans based on agent capabilities.
    
    Generates initial plans by analyzing user queries, available agent capabilities,
    and domain constraints to create optimal execution workflows.
    """
    
    def __init__(self, strategy: Optional[PlanningStrategy] = None):
        """Initialize the adaptive planner."""
        self.strategy = strategy or PlanningStrategy()
        self.logger = logging.getLogger(__name__)
    
    async def generate_plan(
        self,
        query: str,
        context: ExecutionContext,
        preserve_results: Optional[List[IntermediateResult]] = None
    ) -> Plan:
        """
        Generate an initial execution plan based on query and available capabilities.
        
        Args:
            query: User query to execute
            context: Execution context with available agents and constraints
            preserve_results: Previous results to preserve during replanning
            
        Returns:
            Generated execution plan
        """
        self.logger.info(f"Generating plan for query: {query}")
        
        # Analyze query to determine required capabilities
        required_capabilities = await self._analyze_query_requirements(query, context)
        
        # Generate execution steps based on capabilities
        steps = await self._generate_execution_steps(required_capabilities, context, preserve_results)
        
        # Build dependency graph
        dependencies = await self._build_dependency_graph(steps, context)
        
        # Validate the plan
        if not self.strategy.validate_dependencies(steps, dependencies):
            raise ValueError("Generated plan contains invalid dependencies")
        
        # Calculate estimated duration
        total_duration = sum(
            self.strategy.estimate_step_duration(step, context) for step in steps
        )
        
        plan = Plan(
            name=f"Plan for: {query[:50]}...",
            description=f"Execution plan for query: {query}",
            steps=steps,
            dependencies=dependencies,
            estimated_duration_minutes=total_duration,
            metadata={
                "query": query,
                "required_capabilities": [cap.value for cap in required_capabilities],
                "agent_count": len(context.available_agents),
                "preserved_results": len(preserve_results) if preserve_results else 0,
            }
        )
        
        self.logger.info(f"Generated plan {plan.plan_id} with {len(steps)} steps")
        return plan
    
    async def _analyze_query_requirements(
        self, 
        query: str, 
        context: ExecutionContext
    ) -> List[CapabilityType]:
        """Analyze query to determine required capabilities."""
        required_capabilities = []
        query_lower = query.lower()
        
        # Simple keyword-based analysis (in production, use NLP/LLM)
        capability_keywords = {
            CapabilityType.DATA_RETRIEVAL: ["get", "fetch", "retrieve", "load", "find", "search", "data"],
            CapabilityType.DATA_ANALYSIS: ["analyze", "calculate", "compute", "aggregate", "summarize", "trends"],
            CapabilityType.VISUALIZATION: ["plot", "chart", "graph", "visualize", "show", "create", "visualization"],
            CapabilityType.CODE_GENERATION: ["generate", "write", "code"],
            CapabilityType.DOMAIN_EXPERTISE: ["explain", "interpret", "validate", "recommend"],
        }
        
        for capability, keywords in capability_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                required_capabilities.append(capability)
        
        # Ensure we have at least data retrieval for most queries
        if not required_capabilities:
            required_capabilities.append(CapabilityType.DATA_RETRIEVAL)
        
        # Add coordination if multiple capabilities needed
        if len(required_capabilities) > 1:
            required_capabilities.append(CapabilityType.COORDINATION)
        
        return required_capabilities
    
    async def _generate_execution_steps(
        self,
        required_capabilities: List[CapabilityType],
        context: ExecutionContext,
        preserve_results: Optional[List[IntermediateResult]] = None
    ) -> List[ExecutionStep]:
        """Generate execution steps for required capabilities."""
        steps = []
        
        # Add preserved results as completed steps if replanning
        if preserve_results:
            for result in preserve_results:
                if result.preserved:
                    step = ExecutionStep(
                        step_id=result.step_id,
                        name=f"Preserved: {result.result_type}",
                        description=f"Preserved result from previous execution",
                        status=ExecutionStatus.COMPLETED,
                        completed_at=result.created_at,
                        outputs={"preserved_data": result.data}
                    )
                    steps.append(step)
        
        # Generate new steps for required capabilities
        for i, capability in enumerate(required_capabilities):
            # Find suitable agent for this capability
            suitable_agent = await self._find_suitable_agent(capability, context)
            
            step = ExecutionStep(
                name=f"{capability.value.replace('_', ' ').title()} Step",
                description=f"Execute {capability.value} using {suitable_agent or 'available agent'}",
                agent_id=suitable_agent,
                capability_required=capability,
                success_criteria=await self._generate_success_criteria(capability),
                quality_threshold=context.quality_requirements.get(capability.value, 0.8),
                timeout_seconds=context.timeout_seconds or 300,
            )
            steps.append(step)
        
        return steps
    
    async def _find_suitable_agent(
        self, 
        capability: CapabilityType, 
        context: ExecutionContext
    ) -> Optional[str]:
        """Find the most suitable agent for a capability."""
        suitable_agents = []
        
        for agent_id in context.available_agents:
            agent_capabilities = context.agent_capabilities.get(agent_id, [])
            if capability in agent_capabilities:
                suitable_agents.append(agent_id)
        
        # Return first suitable agent (in production, use more sophisticated selection)
        return suitable_agents[0] if suitable_agents else None
    
    async def _generate_success_criteria(self, capability: CapabilityType) -> List[str]:
        """Generate success criteria for a capability."""
        criteria_map = {
            CapabilityType.DATA_RETRIEVAL: [
                "Data successfully retrieved",
                "No data corruption detected",
                "Response time within limits"
            ],
            CapabilityType.DATA_ANALYSIS: [
                "Analysis completed without errors",
                "Results meet quality threshold",
                "Statistical validity confirmed"
            ],
            CapabilityType.VISUALIZATION: [
                "Visualization generated successfully",
                "Chart displays correctly",
                "Data accurately represented"
            ],
            CapabilityType.CODE_GENERATION: [
                "Code generated without syntax errors",
                "Code passes basic validation",
                "Generated code is executable"
            ],
            CapabilityType.DOMAIN_EXPERTISE: [
                "Domain rules applied correctly",
                "Recommendations are valid",
                "Compliance requirements met"
            ],
            CapabilityType.COORDINATION: [
                "All sub-tasks coordinated",
                "Communication successful",
                "Dependencies resolved"
            ],
            CapabilityType.APPROVAL: [
                "Approval request submitted",
                "Response received within timeout",
                "Decision properly recorded"
            ],
        }
        
        return criteria_map.get(capability, ["Task completed successfully"])
    
    async def _build_dependency_graph(
        self,
        steps: List[ExecutionStep],
        context: ExecutionContext
    ) -> Dict[str, List[str]]:
        """Build dependency graph between execution steps."""
        dependencies = {}
        
        # Simple sequential dependencies for now
        # In production, analyze actual data flow dependencies
        for i, step in enumerate(steps):
            if i > 0:
                # Each step depends on the previous one
                dependencies[step.step_id] = [steps[i-1].step_id]
            else:
                dependencies[step.step_id] = []
        
        return dependencies
    
    async def update_plan_with_results(
        self,
        plan: Plan,
        completed_steps: List[ExecutionStep],
        intermediate_results: List[IntermediateResult]
    ) -> Plan:
        """Update plan with completed steps and intermediate results."""
        # Update step statuses
        completed_step_ids = {step.step_id for step in completed_steps}
        
        for step in plan.steps:
            if step.step_id in completed_step_ids:
                completed_step = next(s for s in completed_steps if s.step_id == step.step_id)
                step.status = completed_step.status
                step.started_at = completed_step.started_at
                step.completed_at = completed_step.completed_at
                step.error_message = completed_step.error_message
                step.outputs = completed_step.outputs
        
        # Add intermediate results to metadata
        plan.metadata["intermediate_results"] = [
            {
                "result_id": result.result_id,
                "step_id": result.step_id,
                "result_type": result.result_type,
                "quality_score": result.quality_score,
            }
            for result in intermediate_results
        ]
        
        return plan