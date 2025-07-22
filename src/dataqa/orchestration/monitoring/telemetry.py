"""
Telemetry collection system for agent execution workflows.

Provides comprehensive data collection for execution times, resource usage,
decision points, and workflow performance metrics.
"""

import time
import psutil
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from contextlib import contextmanager
from pydantic import BaseModel, Field

from ..models import AgentType, ExecutionPhase


class ResourceUsage(BaseModel):
    """Resource usage metrics at a point in time."""
    
    cpu_percent: float = Field(description="CPU usage percentage")
    memory_mb: float = Field(description="Memory usage in MB")
    memory_percent: float = Field(description="Memory usage percentage")
    disk_io_read_mb: float = Field(description="Disk read in MB")
    disk_io_write_mb: float = Field(description="Disk write in MB")
    network_sent_mb: float = Field(description="Network sent in MB")
    network_recv_mb: float = Field(description="Network received in MB")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DecisionPoint(BaseModel):
    """Represents a decision point in agent execution."""
    
    decision_id: str = Field(description="Unique decision identifier")
    agent_id: str = Field(description="Agent making the decision")
    decision_type: str = Field(description="Type of decision")
    context: Dict[str, Any] = Field(description="Decision context")
    options_considered: List[str] = Field(description="Options that were considered")
    chosen_option: str = Field(description="Selected option")
    reasoning: str = Field(description="Reasoning for the decision")
    confidence_score: float = Field(description="Confidence in decision (0-1)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExecutionStep(BaseModel):
    """Individual step in agent execution."""
    
    step_id: str = Field(description="Unique step identifier")
    agent_id: str = Field(description="Agent executing the step")
    step_type: str = Field(description="Type of execution step")
    phase: ExecutionPhase = Field(description="Execution phase")
    start_time: datetime = Field(description="Step start time")
    end_time: Optional[datetime] = Field(None, description="Step end time")
    duration_ms: Optional[float] = Field(None, description="Step duration in milliseconds")
    success: Optional[bool] = Field(None, description="Whether step succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    resource_usage_start: Optional[ResourceUsage] = Field(None, description="Resource usage at start")
    resource_usage_end: Optional[ResourceUsage] = Field(None, description="Resource usage at end")
    decision_points: List[DecisionPoint] = Field(default_factory=list, description="Decisions made during step")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional step metadata")


class AgentMetrics(BaseModel):
    """Comprehensive metrics for an individual agent."""
    
    agent_id: str = Field(description="Unique agent identifier")
    agent_type: AgentType = Field(description="Type of agent")
    total_executions: int = Field(default=0, description="Total number of executions")
    successful_executions: int = Field(default=0, description="Number of successful executions")
    failed_executions: int = Field(default=0, description="Number of failed executions")
    average_execution_time_ms: float = Field(default=0.0, description="Average execution time")
    total_execution_time_ms: float = Field(default=0.0, description="Total execution time")
    peak_memory_usage_mb: float = Field(default=0.0, description="Peak memory usage")
    average_cpu_usage: float = Field(default=0.0, description="Average CPU usage")
    decision_count: int = Field(default=0, description="Total decisions made")
    escalation_count: int = Field(default=0, description="Number of escalations")
    last_execution: Optional[datetime] = Field(None, description="Last execution timestamp")
    capabilities_used: Set[str] = Field(default_factory=set, description="Capabilities that have been used")


class ExecutionTelemetry(BaseModel):
    """Complete telemetry data for a workflow execution."""
    
    execution_id: str = Field(description="Unique execution identifier")
    workflow_id: str = Field(description="Workflow identifier")
    correlation_id: str = Field(description="Correlation ID for tracing")
    start_time: datetime = Field(description="Execution start time")
    end_time: Optional[datetime] = Field(None, description="Execution end time")
    duration_ms: Optional[float] = Field(None, description="Total execution duration")
    status: str = Field(default="running", description="Execution status")
    participating_agents: List[str] = Field(default_factory=list, description="Agents involved")
    execution_steps: List[ExecutionStep] = Field(default_factory=list, description="Individual execution steps")
    total_decision_points: int = Field(default=0, description="Total decision points")
    resource_usage_peak: Optional[ResourceUsage] = Field(None, description="Peak resource usage")
    error_count: int = Field(default=0, description="Number of errors encountered")
    warning_count: int = Field(default=0, description="Number of warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional execution metadata")


@dataclass
class TelemetryCollector:
    """
    Collects comprehensive telemetry data for agent execution workflows.
    
    Provides thread-safe collection of execution metrics, resource usage,
    and decision points across multi-agent workflows.
    """
    
    _active_executions: Dict[str, ExecutionTelemetry] = field(default_factory=dict)
    _agent_metrics: Dict[str, AgentMetrics] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _resource_monitoring_enabled: bool = True
    
    def start_execution(self, execution_id: str, workflow_id: str, correlation_id: str) -> None:
        """Start tracking a new execution."""
        with self._lock:
            telemetry = ExecutionTelemetry(
                execution_id=execution_id,
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                start_time=datetime.now(timezone.utc)
            )
            self._active_executions[execution_id] = telemetry
    
    def end_execution(self, execution_id: str, status: str = "completed") -> Optional[ExecutionTelemetry]:
        """End tracking an execution and return final telemetry."""
        with self._lock:
            if execution_id not in self._active_executions:
                return None
            
            telemetry = self._active_executions[execution_id]
            telemetry.end_time = datetime.now(timezone.utc)
            telemetry.status = status
            
            if telemetry.start_time and telemetry.end_time:
                duration = telemetry.end_time - telemetry.start_time
                telemetry.duration_ms = duration.total_seconds() * 1000
            
            # Update agent metrics
            for agent_id in telemetry.participating_agents:
                self._update_agent_metrics(agent_id, telemetry)
            
            # Remove from active executions
            completed_telemetry = self._active_executions.pop(execution_id)
            return completed_telemetry
    
    def start_step(self, execution_id: str, step_id: str, agent_id: str, 
                   step_type: str, phase: ExecutionPhase) -> None:
        """Start tracking an execution step."""
        with self._lock:
            if execution_id not in self._active_executions:
                return
            
            telemetry = self._active_executions[execution_id]
            
            # Add agent to participating agents if not already present
            if agent_id not in telemetry.participating_agents:
                telemetry.participating_agents.append(agent_id)
            
            # Create execution step
            step = ExecutionStep(
                step_id=step_id,
                agent_id=agent_id,
                step_type=step_type,
                phase=phase,
                start_time=datetime.now(timezone.utc)
            )
            
            # Collect resource usage if enabled
            if self._resource_monitoring_enabled:
                step.resource_usage_start = self._collect_resource_usage()
            
            telemetry.execution_steps.append(step)
    
    def end_step(self, execution_id: str, step_id: str, success: bool = True, 
                 error_message: Optional[str] = None) -> None:
        """End tracking an execution step."""
        with self._lock:
            if execution_id not in self._active_executions:
                return
            
            telemetry = self._active_executions[execution_id]
            
            # Find the step
            step = None
            for s in telemetry.execution_steps:
                if s.step_id == step_id:
                    step = s
                    break
            
            if not step:
                return
            
            step.end_time = datetime.now(timezone.utc)
            step.success = success
            step.error_message = error_message
            
            if step.start_time and step.end_time:
                duration = step.end_time - step.start_time
                step.duration_ms = duration.total_seconds() * 1000
            
            # Collect final resource usage
            if self._resource_monitoring_enabled:
                step.resource_usage_end = self._collect_resource_usage()
            
            # Update execution-level counters
            if not success:
                telemetry.error_count += 1
    
    def record_decision(self, execution_id: str, step_id: str, decision: DecisionPoint) -> None:
        """Record a decision point during execution."""
        with self._lock:
            if execution_id not in self._active_executions:
                return
            
            telemetry = self._active_executions[execution_id]
            
            # Find the step and add decision
            for step in telemetry.execution_steps:
                if step.step_id == step_id:
                    step.decision_points.append(decision)
                    telemetry.total_decision_points += 1
                    break
    
    def get_execution_telemetry(self, execution_id: str) -> Optional[ExecutionTelemetry]:
        """Get current telemetry for an active execution."""
        with self._lock:
            return self._active_executions.get(execution_id)
    
    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get metrics for a specific agent."""
        with self._lock:
            return self._agent_metrics.get(agent_id)
    
    def get_all_agent_metrics(self) -> Dict[str, AgentMetrics]:
        """Get metrics for all agents."""
        with self._lock:
            return self._agent_metrics.copy()
    
    def get_active_executions(self) -> List[str]:
        """Get list of currently active execution IDs."""
        with self._lock:
            return list(self._active_executions.keys())
    
    @contextmanager
    def track_step(self, execution_id: str, step_id: str, agent_id: str, 
                   step_type: str, phase: ExecutionPhase):
        """Context manager for tracking execution steps."""
        self.start_step(execution_id, step_id, agent_id, step_type, phase)
        try:
            yield
            self.end_step(execution_id, step_id, success=True)
        except Exception as e:
            self.end_step(execution_id, step_id, success=False, error_message=str(e))
            raise
    
    def _collect_resource_usage(self) -> ResourceUsage:
        """Collect current system resource usage."""
        try:
            process = psutil.Process()
            
            # Get CPU and memory info
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Get I/O info
            io_counters = process.io_counters()
            
            return ResourceUsage(
                cpu_percent=cpu_percent,
                memory_mb=memory_info.rss / (1024 * 1024),  # Convert to MB
                memory_percent=memory_percent,
                disk_io_read_mb=io_counters.read_bytes / (1024 * 1024),
                disk_io_write_mb=io_counters.write_bytes / (1024 * 1024),
                network_sent_mb=0.0,  # Would need network monitoring
                network_recv_mb=0.0   # Would need network monitoring
            )
        except Exception:
            # Return zero values if resource collection fails
            return ResourceUsage(
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0
            )
    
    def _update_agent_metrics(self, agent_id: str, telemetry: ExecutionTelemetry) -> None:
        """Update metrics for an agent based on execution telemetry."""
        if agent_id not in self._agent_metrics:
            self._agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                agent_type=AgentType.WORKER  # Default, should be set properly
            )
        
        metrics = self._agent_metrics[agent_id]
        metrics.total_executions += 1
        metrics.last_execution = telemetry.end_time
        
        if telemetry.status == "completed":
            metrics.successful_executions += 1
        else:
            metrics.failed_executions += 1
        
        # Update timing metrics
        if telemetry.duration_ms:
            metrics.total_execution_time_ms += telemetry.duration_ms
            metrics.average_execution_time_ms = (
                metrics.total_execution_time_ms / metrics.total_executions
            )
        
        # Update resource metrics from peak usage
        if telemetry.resource_usage_peak:
            metrics.peak_memory_usage_mb = max(
                metrics.peak_memory_usage_mb,
                telemetry.resource_usage_peak.memory_mb
            )
        
        # Count decisions and escalations
        metrics.decision_count += telemetry.total_decision_points
        
        # Update escalation count based on error patterns
        if telemetry.error_count > 0:
            metrics.escalation_count += 1