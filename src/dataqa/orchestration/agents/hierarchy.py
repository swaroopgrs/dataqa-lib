"""
Agent hierarchy management for multi-agent orchestration.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from .base import BaseAgent
from .manager import ManagerAgent
from .worker import WorkerAgent
from ..models import AgentConfiguration, AgentRole


class HierarchyNode(BaseModel):
    """Node in the agent hierarchy."""
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    depth: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentHierarchy:
    """
    Manages hierarchical relationships between agents in a multi-agent system.
    
    Provides functionality for:
    - Building and validating agent hierarchies
    - Managing parent-child relationships
    - Capability-based routing and delegation
    - Hierarchy traversal and querying
    """
    
    def __init__(self):
        """Initialize empty agent hierarchy."""
        self.agents: Dict[str, BaseAgent] = {}
        self.hierarchy_nodes: Dict[str, HierarchyNode] = {}
        self.root_agents: Set[str] = set()
        self._hierarchy_valid = True
        self._validation_errors: List[str] = []
    
    def add_agent(self, agent: BaseAgent, parent_id: Optional[str] = None) -> None:
        """
        Add an agent to the hierarchy.
        
        Args:
            agent: Agent to add
            parent_id: ID of parent agent (None for root agents)
            
        Raises:
            ValueError: If agent already exists or parent doesn't exist
        """
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent {agent.agent_id} already exists in hierarchy")
        
        if parent_id and parent_id not in self.agents:
            raise ValueError(f"Parent agent {parent_id} not found in hierarchy")
        
        # Add agent
        self.agents[agent.agent_id] = agent
        
        # Create hierarchy node
        depth = 0
        if parent_id:
            parent_node = self.hierarchy_nodes[parent_id]
            depth = parent_node.depth + 1
            parent_node.children_ids.append(agent.agent_id)
        else:
            self.root_agents.add(agent.agent_id)
        
        node = HierarchyNode(
            agent_id=agent.agent_id,
            parent_id=parent_id,
            depth=depth
        )
        self.hierarchy_nodes[agent.agent_id] = node
        
        # Set up agent relationships
        if parent_id and isinstance(agent, WorkerAgent):
            parent_agent = self.agents[parent_id]
            if isinstance(parent_agent, ManagerAgent):
                parent_agent.add_subordinate(agent)
        
        # Invalidate hierarchy validation
        self._hierarchy_valid = False
    
    def remove_agent(self, agent_id: str) -> None:
        """
        Remove an agent from the hierarchy.
        
        Args:
            agent_id: ID of agent to remove
            
        Raises:
            ValueError: If agent doesn't exist or has children
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found in hierarchy")
        
        node = self.hierarchy_nodes[agent_id]
        
        if node.children_ids:
            raise ValueError(f"Cannot remove agent {agent_id} with children. Remove children first.")
        
        # Remove from parent's children
        if node.parent_id:
            parent_node = self.hierarchy_nodes[node.parent_id]
            parent_node.children_ids.remove(agent_id)
            
            # Remove from manager's subordinates
            parent_agent = self.agents[node.parent_id]
            if isinstance(parent_agent, ManagerAgent):
                parent_agent.remove_subordinate(agent_id)
        else:
            self.root_agents.discard(agent_id)
        
        # Remove agent and node
        del self.agents[agent_id]
        del self.hierarchy_nodes[agent_id]
        
        # Invalidate hierarchy validation
        self._hierarchy_valid = False
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)
    
    def get_parent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get parent agent of the specified agent."""
        if agent_id not in self.hierarchy_nodes:
            return None
        
        parent_id = self.hierarchy_nodes[agent_id].parent_id
        return self.agents.get(parent_id) if parent_id else None
    
    def get_children(self, agent_id: str) -> List[BaseAgent]:
        """Get direct children of the specified agent."""
        if agent_id not in self.hierarchy_nodes:
            return []
        
        children_ids = self.hierarchy_nodes[agent_id].children_ids
        return [self.agents[child_id] for child_id in children_ids if child_id in self.agents]
    
    def get_descendants(self, agent_id: str) -> List[BaseAgent]:
        """Get all descendants (children, grandchildren, etc.) of the specified agent."""
        descendants = []
        children = self.get_children(agent_id)
        
        for child in children:
            descendants.append(child)
            descendants.extend(self.get_descendants(child.agent_id))
        
        return descendants
    
    def get_ancestors(self, agent_id: str) -> List[BaseAgent]:
        """Get all ancestors (parent, grandparent, etc.) of the specified agent."""
        ancestors = []
        current_id = agent_id
        
        while current_id in self.hierarchy_nodes:
            parent_id = self.hierarchy_nodes[current_id].parent_id
            if not parent_id:
                break
            
            parent_agent = self.agents.get(parent_id)
            if parent_agent:
                ancestors.append(parent_agent)
            
            current_id = parent_id
        
        return ancestors
    
    def get_siblings(self, agent_id: str) -> List[BaseAgent]:
        """Get sibling agents (same parent) of the specified agent."""
        if agent_id not in self.hierarchy_nodes:
            return []
        
        parent_id = self.hierarchy_nodes[agent_id].parent_id
        if not parent_id:
            # Root agent - siblings are other root agents
            return [self.agents[root_id] for root_id in self.root_agents if root_id != agent_id]
        
        # Get parent's children excluding this agent
        parent_children = self.get_children(parent_id)
        return [child for child in parent_children if child.agent_id != agent_id]
    
    def find_agents_by_capability(self, capability_type: str) -> List[BaseAgent]:
        """
        Find all agents that have a specific capability.
        
        Args:
            capability_type: Type of capability to search for
            
        Returns:
            List of agents with the specified capability
        """
        return [agent for agent in self.agents.values() if agent.has_capability(capability_type)]
    
    def find_available_agents_by_capability(self, capability_type: str) -> List[BaseAgent]:
        """
        Find available agents that have a specific capability.
        
        Args:
            capability_type: Type of capability to search for
            
        Returns:
            List of available agents with the specified capability
        """
        return [
            agent for agent in self.agents.values() 
            if agent.has_capability(capability_type) and agent.is_available
        ]
    
    def find_best_agent_for_task(self, task: 'Task', routing_strategy: str = "capability_based") -> Optional[BaseAgent]:
        """
        Find the best available agent for a specific task using advanced routing.
        
        Args:
            task: Task to find agent for
            routing_strategy: Strategy for agent selection ("capability_based", "load_balanced", "priority_based", "round_robin")
            
        Returns:
            Best available agent or None if no suitable agent found
        """
        from .base import Task  # Import here to avoid circular imports
        
        if not task.required_capabilities:
            return None
        
        # Find agents with all required capabilities
        capable_agents = []
        for agent in self.agents.values():
            if agent.is_available and all(agent.has_capability(cap) for cap in task.required_capabilities):
                capable_agents.append(agent)
        
        if not capable_agents:
            return None
        
        # Apply routing strategy
        if routing_strategy == "load_balanced":
            return min(capable_agents, key=lambda a: a.active_task_count)
        elif routing_strategy == "priority_based":
            return min(capable_agents, key=lambda a: a.priority_level)
        elif routing_strategy == "round_robin":
            # Simple round-robin based on agent ID hash
            return capable_agents[hash(task.task_id) % len(capable_agents)]
        else:  # capability_based (default)
            # Score agents based on multiple factors
            scored_agents = []
            for agent in capable_agents:
                score = self._calculate_agent_score(agent, task)
                scored_agents.append((agent, score))
            
            # Return agent with highest score
            scored_agents.sort(key=lambda x: x[1], reverse=True)
            return scored_agents[0][0]
    
    def _calculate_agent_score(self, agent: BaseAgent, task: 'Task') -> float:
        """
        Calculate a score for how well an agent matches a task.
        
        Args:
            agent: Agent to score
            task: Task to match against
            
        Returns:
            Score (higher is better)
        """
        score = 0.0
        
        # Base score for having required capabilities
        score += len(task.required_capabilities) * 10
        
        # Bonus for specialization match
        if agent.specialization and task.context.get("domain") == agent.specialization:
            score += 20
        
        # Penalty for high current load
        load_penalty = agent.active_task_count * 5
        score -= load_penalty
        
        # Bonus for higher priority agents
        priority_bonus = (11 - agent.priority_level) * 2  # Higher priority = lower number
        score += priority_bonus
        
        # Bonus for task priority match
        if hasattr(task, 'priority'):
            if task.priority <= 3 and agent.priority_level <= 3:  # High priority task + high priority agent
                score += 15
        
        # Penalty for agents with recent failures (if we track this)
        if hasattr(agent, '_task_history'):
            recent_failures = sum(1 for result in agent._task_history[-5:] if result.status.value == 'failed')
            score -= recent_failures * 3
        
        return score
    
    def discover_agents_by_pattern(self, pattern: Dict[str, Any]) -> List[BaseAgent]:
        """
        Discover agents matching a complex pattern.
        
        Args:
            pattern: Dictionary with search criteria
                - capabilities: List of required capabilities
                - role: Required agent role
                - specialization: Required specialization
                - available_only: Whether to include only available agents
                - max_load: Maximum current task load
                - min_priority: Minimum priority level (lower number = higher priority)
                
        Returns:
            List of matching agents
        """
        matching_agents = []
        
        for agent in self.agents.values():
            # Check availability if required
            if pattern.get('available_only', False) and not agent.is_available:
                continue
            
            # Check capabilities
            required_caps = pattern.get('capabilities', [])
            if required_caps and not all(agent.has_capability(cap) for cap in required_caps):
                continue
            
            # Check role
            required_role = pattern.get('role')
            if required_role and agent.role != required_role:
                continue
            
            # Check specialization
            required_spec = pattern.get('specialization')
            if required_spec and agent.specialization != required_spec:
                continue
            
            # Check maximum load
            max_load = pattern.get('max_load')
            if max_load is not None and agent.active_task_count > max_load:
                continue
            
            # Check minimum priority
            min_priority = pattern.get('min_priority')
            if min_priority is not None and agent.priority_level > min_priority:
                continue
            
            matching_agents.append(agent)
        
        return matching_agents
    
    def get_load_balanced_agent(self, capability_type: str) -> Optional[BaseAgent]:
        """
        Get the least loaded available agent with a specific capability.
        
        Args:
            capability_type: Required capability type
            
        Returns:
            Least loaded agent or None if no suitable agent found
        """
        available_agents = self.find_available_agents_by_capability(capability_type)
        
        if not available_agents:
            return None
        
        # Return agent with lowest current load
        return min(available_agents, key=lambda a: a.active_task_count)
    
    def get_capability_coverage_report(self) -> Dict[str, Any]:
        """
        Generate a report on capability coverage across the hierarchy.
        
        Returns:
            Dictionary with capability coverage information
        """
        capability_coverage = {}
        
        # Collect all capabilities
        all_capabilities = set()
        for agent in self.agents.values():
            all_capabilities.update(agent.capability_types)
        
        # Analyze coverage for each capability
        for capability in all_capabilities:
            agents_with_cap = self.find_agents_by_capability(capability)
            available_agents_with_cap = self.find_available_agents_by_capability(capability)
            
            capability_coverage[capability] = {
                "total_agents": len(agents_with_cap),
                "available_agents": len(available_agents_with_cap),
                "agent_ids": [a.agent_id for a in agents_with_cap],
                "available_agent_ids": [a.agent_id for a in available_agents_with_cap],
                "coverage_percentage": (len(available_agents_with_cap) / max(len(agents_with_cap), 1)) * 100
            }
        
        return {
            "capabilities": capability_coverage,
            "total_capabilities": len(all_capabilities),
            "total_agents": len(self.agents),
            "available_agents": len([a for a in self.agents.values() if a.is_available])
        }
    
    def find_agents_by_role(self, role: AgentRole) -> List[BaseAgent]:
        """
        Find all agents with a specific role.
        
        Args:
            role: Agent role to search for
            
        Returns:
            List of agents with the specified role
        """
        return [agent for agent in self.agents.values() if agent.role == role]
    
    def find_agents_by_specialization(self, specialization: str) -> List[BaseAgent]:
        """
        Find all agents with a specific specialization.
        
        Args:
            specialization: Specialization to search for
            
        Returns:
            List of agents with the specified specialization
        """
        return [
            agent for agent in self.agents.values() 
            if agent.specialization == specialization
        ]
    
    def get_hierarchy_depth(self) -> int:
        """Get the maximum depth of the hierarchy."""
        if not self.hierarchy_nodes:
            return 0
        return max(node.depth for node in self.hierarchy_nodes.values()) + 1
    
    def get_agents_at_depth(self, depth: int) -> List[BaseAgent]:
        """
        Get all agents at a specific depth in the hierarchy.
        
        Args:
            depth: Depth level (0 = root level)
            
        Returns:
            List of agents at the specified depth
        """
        agent_ids = [
            node.agent_id for node in self.hierarchy_nodes.values() 
            if node.depth == depth
        ]
        return [self.agents[agent_id] for agent_id in agent_ids]
    
    def validate_hierarchy(self) -> Tuple[bool, List[str]]:
        """
        Validate the hierarchy structure.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if self._hierarchy_valid:
            return True, []
        
        errors = []
        
        # Check for circular dependencies
        for agent_id in self.agents:
            if self._has_circular_dependency(agent_id):
                errors.append(f"Circular dependency detected involving agent {agent_id}")
        
        # Check for orphaned nodes
        for agent_id, node in self.hierarchy_nodes.items():
            if node.parent_id and node.parent_id not in self.agents:
                errors.append(f"Agent {agent_id} has non-existent parent {node.parent_id}")
        
        # Check role consistency
        for agent_id, agent in self.agents.items():
            children = self.get_children(agent_id)
            if children and agent.role != AgentRole.MANAGER:
                errors.append(f"Agent {agent_id} has children but is not a manager")
            
            if agent.role == AgentRole.MANAGER and not isinstance(agent, ManagerAgent):
                errors.append(f"Agent {agent_id} has manager role but is not a ManagerAgent instance")
            
            if agent.role == AgentRole.WORKER and not isinstance(agent, WorkerAgent):
                errors.append(f"Agent {agent_id} has worker role but is not a WorkerAgent instance")
        
        self._hierarchy_valid = len(errors) == 0
        self._validation_errors = errors
        
        return self._hierarchy_valid, errors
    
    def _has_circular_dependency(self, agent_id: str, visited: Optional[Set[str]] = None) -> bool:
        """Check if an agent has circular dependencies in its ancestry."""
        if visited is None:
            visited = set()
        
        if agent_id in visited:
            return True
        
        visited.add(agent_id)
        
        if agent_id not in self.hierarchy_nodes:
            return False
        
        parent_id = self.hierarchy_nodes[agent_id].parent_id
        if parent_id:
            return self._has_circular_dependency(parent_id, visited.copy())
        
        return False
    
    def get_hierarchy_summary(self) -> Dict[str, Any]:
        """Get a summary of the hierarchy structure."""
        is_valid, errors = self.validate_hierarchy()
        
        return {
            "total_agents": len(self.agents),
            "root_agents": len(self.root_agents),
            "max_depth": self.get_hierarchy_depth(),
            "manager_count": len(self.find_agents_by_role(AgentRole.MANAGER)),
            "worker_count": len(self.find_agents_by_role(AgentRole.WORKER)),
            "is_valid": is_valid,
            "validation_errors": errors,
            "agents_by_depth": {
                depth: len(self.get_agents_at_depth(depth))
                for depth in range(self.get_hierarchy_depth())
            }
        }
    
    def discover_agents_with_multiple_capabilities(self, capabilities: List[str], match_all: bool = True) -> List[BaseAgent]:
        """
        Discover agents that have multiple capabilities.
        
        Args:
            capabilities: List of capability types to search for
            match_all: If True, agent must have ALL capabilities. If False, agent must have ANY capability.
            
        Returns:
            List of agents matching the capability criteria
        """
        matching_agents = []
        
        for agent in self.agents.values():
            if match_all:
                # Agent must have all capabilities
                if all(agent.has_capability(cap) for cap in capabilities):
                    matching_agents.append(agent)
            else:
                # Agent must have at least one capability
                if any(agent.has_capability(cap) for cap in capabilities):
                    matching_agents.append(agent)
        
        return matching_agents
    
    def get_agent_routing_recommendations(self, task: 'Task') -> Dict[str, Any]:
        """
        Get routing recommendations for a task with detailed analysis.
        
        Args:
            task: Task to analyze
            
        Returns:
            Dictionary with routing recommendations and analysis
        """
        from .base import Task  # Import here to avoid circular imports
        
        recommendations = {
            "task_id": task.task_id,
            "required_capabilities": task.required_capabilities,
            "routing_options": [],
            "best_recommendation": None,
            "analysis": {}
        }
        
        # Find all capable agents
        capable_agents = []
        for agent in self.agents.values():
            if all(agent.has_capability(cap) for cap in task.required_capabilities):
                capable_agents.append(agent)
        
        if not capable_agents:
            recommendations["analysis"]["no_capable_agents"] = True
            return recommendations
        
        # Analyze each routing strategy
        strategies = ["capability_based", "load_balanced", "priority_based", "round_robin"]
        
        for strategy in strategies:
            best_agent = self.find_best_agent_for_task(task, strategy)
            if best_agent:
                option = {
                    "strategy": strategy,
                    "agent_id": best_agent.agent_id,
                    "agent_name": best_agent.name,
                    "current_load": best_agent.active_task_count,
                    "priority_level": best_agent.priority_level,
                    "specialization": best_agent.specialization,
                    "is_available": best_agent.is_available
                }
                
                if strategy == "capability_based":
                    option["score"] = self._calculate_agent_score(best_agent, task)
                
                recommendations["routing_options"].append(option)
        
        # Set best recommendation (capability_based by default)
        if recommendations["routing_options"]:
            recommendations["best_recommendation"] = recommendations["routing_options"][0]
        
        # Add analysis
        recommendations["analysis"] = {
            "total_capable_agents": len(capable_agents),
            "available_capable_agents": len([a for a in capable_agents if a.is_available]),
            "capability_coverage": {
                cap: len(self.find_agents_by_capability(cap))
                for cap in task.required_capabilities
            },
            "load_distribution": {
                "min_load": min(a.active_task_count for a in capable_agents),
                "max_load": max(a.active_task_count for a in capable_agents),
                "avg_load": sum(a.active_task_count for a in capable_agents) / len(capable_agents)
            }
        }
        
        return recommendations
    
    def get_agent_network_topology(self) -> Dict[str, Any]:
        """
        Get network topology information for the agent hierarchy.
        
        Returns:
            Dictionary with topology information
        """
        topology = {
            "nodes": [],
            "edges": [],
            "clusters": {},
            "metrics": {}
        }
        
        # Build nodes
        for agent_id, agent in self.agents.items():
            node = {
                "id": agent_id,
                "name": agent.name,
                "role": agent.role.value if hasattr(agent.role, 'value') else agent.role,
                "capabilities": list(agent.capability_types),
                "specialization": agent.specialization,
                "depth": self.hierarchy_nodes[agent_id].depth,
                "is_available": agent.is_available,
                "active_tasks": agent.active_task_count
            }
            topology["nodes"].append(node)
        
        # Build edges (parent-child relationships)
        for agent_id, node in self.hierarchy_nodes.items():
            if node.parent_id:
                edge = {
                    "source": node.parent_id,
                    "target": agent_id,
                    "type": "hierarchy"
                }
                topology["edges"].append(edge)
        
        # Build capability clusters
        capability_clusters = {}
        for agent in self.agents.values():
            for cap in agent.capability_types:
                if cap not in capability_clusters:
                    capability_clusters[cap] = []
                capability_clusters[cap].append(agent.agent_id)
        
        topology["clusters"]["capabilities"] = capability_clusters
        
        # Build specialization clusters
        specialization_clusters = {}
        for agent in self.agents.values():
            if agent.specialization:
                if agent.specialization not in specialization_clusters:
                    specialization_clusters[agent.specialization] = []
                specialization_clusters[agent.specialization].append(agent.agent_id)
        
        topology["clusters"]["specializations"] = specialization_clusters
        
        # Calculate metrics
        topology["metrics"] = {
            "total_nodes": len(self.agents),
            "total_edges": len(topology["edges"]),
            "max_depth": self.get_hierarchy_depth(),
            "branching_factor": len(topology["edges"]) / max(len(self.root_agents), 1),
            "capability_diversity": len(capability_clusters),
            "specialization_diversity": len(specialization_clusters)
        }
        
        return topology
    
    def find_optimal_delegation_path(self, task: 'Task', start_agent_id: str) -> Optional[List[str]]:
        """
        Find optimal delegation path from a starting agent to execute a task.
        
        Args:
            task: Task to execute
            start_agent_id: Starting agent ID
            
        Returns:
            List of agent IDs representing the delegation path, or None if no path found
        """
        from .base import Task  # Import here to avoid circular imports
        
        if start_agent_id not in self.agents:
            return None
        
        start_agent = self.agents[start_agent_id]
        
        # If start agent can handle the task directly
        if start_agent.can_handle_task(task):
            return [start_agent_id]
        
        # If start agent is a manager, try to delegate to subordinates
        if isinstance(start_agent, ManagerAgent):
            # Find capable subordinates
            capable_subordinates = []
            for subordinate in start_agent.subordinates:
                if subordinate.can_handle_task(task):
                    capable_subordinates.append(subordinate)
            
            if capable_subordinates:
                # Select best subordinate using scoring
                best_subordinate = max(
                    capable_subordinates,
                    key=lambda a: self._calculate_agent_score(a, task)
                )
                return [start_agent_id, best_subordinate.agent_id]
            
            # Try deeper delegation
            for subordinate in start_agent.subordinates:
                if isinstance(subordinate, ManagerAgent):
                    sub_path = self.find_optimal_delegation_path(task, subordinate.agent_id)
                    if sub_path:
                        return [start_agent_id] + sub_path
        
        # Try escalation to parent
        parent = self.get_parent(start_agent_id)
        if parent and isinstance(parent, ManagerAgent):
            parent_path = self.find_optimal_delegation_path(task, parent.agent_id)
            if parent_path and parent_path[0] != start_agent_id:  # Avoid cycles
                return parent_path
        
        return None
    
    def get_agent_collaboration_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Generate a collaboration matrix showing potential collaboration scores between agents.
        
        Returns:
            Matrix of collaboration scores between agents
        """
        matrix = {}
        
        for agent1_id, agent1 in self.agents.items():
            matrix[agent1_id] = {}
            
            for agent2_id, agent2 in self.agents.items():
                if agent1_id == agent2_id:
                    matrix[agent1_id][agent2_id] = 0.0
                    continue
                
                # Calculate collaboration score
                score = 0.0
                
                # Capability complementarity
                agent1_caps = agent1.capability_types
                agent2_caps = agent2.capability_types
                
                # Bonus for complementary capabilities
                unique_caps = len(agent1_caps.union(agent2_caps))
                common_caps = len(agent1_caps.intersection(agent2_caps))
                if unique_caps > 0:
                    score += (unique_caps - common_caps) / unique_caps * 50
                
                # Bonus for same specialization
                if agent1.specialization and agent1.specialization == agent2.specialization:
                    score += 20
                
                # Penalty for hierarchy distance
                if self._are_agents_related(agent1_id, agent2_id):
                    distance = self._calculate_hierarchy_distance(agent1_id, agent2_id)
                    score -= distance * 5
                else:
                    score -= 10  # Penalty for unrelated agents
                
                # Load balancing consideration
                avg_load = (agent1.active_task_count + agent2.active_task_count) / 2
                if avg_load < 2:  # Both agents are lightly loaded
                    score += 10
                
                matrix[agent1_id][agent2_id] = max(0.0, score)
        
        return matrix
    
    def _are_agents_related(self, agent1_id: str, agent2_id: str) -> bool:
        """Check if two agents are related in the hierarchy."""
        # Check if one is ancestor of the other
        ancestors1 = [a.agent_id for a in self.get_ancestors(agent1_id)]
        ancestors2 = [a.agent_id for a in self.get_ancestors(agent2_id)]
        
        return (agent1_id in ancestors2 or 
                agent2_id in ancestors1 or 
                len(set(ancestors1).intersection(set(ancestors2))) > 0)
    
    def _calculate_hierarchy_distance(self, agent1_id: str, agent2_id: str) -> int:
        """Calculate the hierarchy distance between two agents."""
        if agent1_id not in self.hierarchy_nodes or agent2_id not in self.hierarchy_nodes:
            return float('inf')
        
        depth1 = self.hierarchy_nodes[agent1_id].depth
        depth2 = self.hierarchy_nodes[agent2_id].depth
        
        # Simple distance calculation based on depth difference
        # In a real implementation, this would use graph algorithms
        return abs(depth1 - depth2) + 1
    
    def print_hierarchy(self, agent_id: Optional[str] = None, indent: int = 0) -> None:
        """
        Print a visual representation of the hierarchy.
        
        Args:
            agent_id: Starting agent ID (None for all root agents)
            indent: Current indentation level
        """
        if agent_id is None:
            # Print all root agents
            for root_id in self.root_agents:
                self.print_hierarchy(root_id, 0)
            return
        
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        prefix = "  " * indent
        role_value = agent.role.value if hasattr(agent.role, 'value') else agent.role
        status = "✓" if agent.is_available else "✗"
        capabilities = ", ".join(list(agent.capability_types)[:3])  # Show first 3 capabilities
        if len(agent.capability_types) > 3:
            capabilities += "..."
        
        print(f"{prefix}{status} {agent.name} ({role_value}) [{capabilities}] - {agent.agent_id}")
        
        # Print children
        children = self.get_children(agent_id)
        for child in children:
            self.print_hierarchy(child.agent_id, indent + 1)