"""
Health check system and status reporting.

Provides comprehensive health monitoring for multi-agent orchestration
with configurable health checks and status endpoints.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field

from .metrics import metrics_collector, MetricsSnapshot


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ComponentType(str, Enum):
    """Types of system components."""
    AGENT = "agent"
    DATABASE = "database"
    EXTERNAL_SERVICE = "external_service"
    QUEUE = "queue"
    CACHE = "cache"
    STORAGE = "storage"
    NETWORK = "network"
    SYSTEM = "system"


class HealthCheck(BaseModel):
    """Individual health check definition."""
    
    name: str = Field(description="Health check name")
    component_type: ComponentType = Field(description="Type of component being checked")
    description: str = Field(description="Health check description")
    timeout_seconds: float = Field(default=5.0, description="Check timeout")
    interval_seconds: float = Field(default=30.0, description="Check interval")
    critical: bool = Field(default=False, description="Whether this check is critical")
    enabled: bool = Field(default=True, description="Whether check is enabled")


class HealthCheckResult(BaseModel):
    """Result of a health check."""
    
    check_name: str = Field(description="Name of the health check")
    status: HealthStatus = Field(description="Health status")
    message: str = Field(description="Status message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    duration_ms: float = Field(description="Check duration in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = Field(None, description="Error message if check failed")


class ComponentStatus(BaseModel):
    """Status of a system component."""
    
    component_name: str = Field(description="Component name")
    component_type: ComponentType = Field(description="Component type")
    status: HealthStatus = Field(description="Overall component status")
    health_checks: List[HealthCheckResult] = Field(default_factory=list)
    last_healthy: Optional[datetime] = Field(None, description="Last time component was healthy")
    uptime_seconds: float = Field(default=0.0, description="Component uptime")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Component metadata")


class SystemStatus(BaseModel):
    """Overall system status."""
    
    status: HealthStatus = Field(description="Overall system status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    uptime_seconds: float = Field(description="System uptime")
    components: List[ComponentStatus] = Field(default_factory=list)
    active_agents: int = Field(default=0, description="Number of active agents")
    active_executions: int = Field(default=0, description="Number of active executions")
    total_executions: int = Field(default=0, description="Total executions")
    error_rate: float = Field(default=0.0, description="Current error rate")
    performance_score: float = Field(default=1.0, description="Performance score (0-1)")
    alerts: List[str] = Field(default_factory=list, description="Active alerts")
    version: str = Field(default="1.0.0", description="System version")


@dataclass
class HealthChecker:
    """
    Comprehensive health monitoring system.
    
    Provides configurable health checks, status aggregation,
    and automated monitoring for multi-agent orchestration.
    """
    
    _health_checks: Dict[str, HealthCheck] = field(default_factory=dict)
    _check_functions: Dict[str, Callable[[], Awaitable[HealthCheckResult]]] = field(default_factory=dict)
    _last_results: Dict[str, HealthCheckResult] = field(default_factory=dict)
    _component_status: Dict[str, ComponentStatus] = field(default_factory=dict)
    _start_time: float = field(default_factory=time.time)
    _monitoring_task: Optional[asyncio.Task] = None
    _monitoring_enabled: bool = True
    
    def register_health_check(self, check: HealthCheck, 
                             check_function: Callable[[], Awaitable[HealthCheckResult]]) -> None:
        """Register a health check with its implementation."""
        self._health_checks[check.name] = check
        self._check_functions[check.name] = check_function
    
    def register_component(self, component_name: str, component_type: ComponentType,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a system component for monitoring."""
        self._component_status[component_name] = ComponentStatus(
            component_name=component_name,
            component_type=component_type,
            status=HealthStatus.HEALTHY,
            metadata=metadata or {}
        )
    
    async def run_health_check(self, check_name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if check_name not in self._health_checks:
            return HealthCheckResult(
                check_name=check_name,
                status=HealthStatus.UNHEALTHY,
                message="Health check not found",
                duration_ms=0.0,
                error="Check not registered"
            )
        
        check = self._health_checks[check_name]
        check_function = self._check_functions[check_name]
        
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                check_function(),
                timeout=check.timeout_seconds
            )
            
            duration_ms = (time.time() - start_time) * 1000
            result.duration_ms = duration_ms
            
            # Store result
            self._last_results[check_name] = result
            
            return result
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                check_name=check_name,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
                duration_ms=duration_ms,
                error="Timeout"
            )
            self._last_results[check_name] = result
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                check_name=check_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                error=str(e)
            )
            self._last_results[check_name] = result
            return result
    
    async def run_all_health_checks(self) -> List[HealthCheckResult]:
        """Run all registered health checks."""
        results = []
        
        for check_name, check in self._health_checks.items():
            if check.enabled:
                result = await self.run_health_check(check_name)
                results.append(result)
        
        return results
    
    def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status."""
        uptime = time.time() - self._start_time
        
        # Update component statuses based on health checks
        self._update_component_statuses()
        
        # Calculate overall system status
        overall_status = self._calculate_overall_status()
        
        # Get metrics from collector
        active_executions = metrics_collector.get_metric_value("concurrent_executions") or 0
        total_executions = metrics_collector.get_metric_value("total_executions") or 0
        error_rate = metrics_collector.get_metric_value("error_rate") or 0.0
        
        # Calculate performance score
        performance_score = self._calculate_performance_score()
        
        # Get active alerts
        alerts = self._get_active_alerts()
        
        return SystemStatus(
            status=overall_status,
            uptime_seconds=uptime,
            components=list(self._component_status.values()),
            active_agents=len([c for c in self._component_status.values() 
                             if c.component_type == ComponentType.AGENT]),
            active_executions=int(active_executions),
            total_executions=int(total_executions),
            error_rate=error_rate,
            performance_score=performance_score,
            alerts=alerts
        )
    
    def get_component_status(self, component_name: str) -> Optional[ComponentStatus]:
        """Get status for a specific component."""
        return self._component_status.get(component_name)
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_enabled = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self._monitoring_enabled = False
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
    
    async def _monitoring_loop(self) -> None:
        """Continuous monitoring loop."""
        while self._monitoring_enabled:
            try:
                # Run all health checks
                await self.run_all_health_checks()
                
                # Wait for next interval (use minimum interval)
                min_interval = min(
                    (check.interval_seconds for check in self._health_checks.values()),
                    default=30.0
                )
                await asyncio.sleep(min_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error and continue
                print(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)
    
    def _update_component_statuses(self) -> None:
        """Update component statuses based on health check results."""
        # Group health check results by component
        component_results = {}
        
        for check_name, result in self._last_results.items():
            check = self._health_checks.get(check_name)
            if not check:
                continue
            
            # Map check to component (simplified - could be more sophisticated)
            component_name = check_name.split('_')[0]  # Assume check names start with component
            
            if component_name not in component_results:
                component_results[component_name] = []
            component_results[component_name].append(result)
        
        # Update component statuses
        for component_name, results in component_results.items():
            if component_name in self._component_status:
                component = self._component_status[component_name]
                component.health_checks = results
                
                # Determine component status
                statuses = [r.status for r in results]
                if HealthStatus.CRITICAL in statuses:
                    component.status = HealthStatus.CRITICAL
                elif HealthStatus.UNHEALTHY in statuses:
                    component.status = HealthStatus.UNHEALTHY
                elif HealthStatus.DEGRADED in statuses:
                    component.status = HealthStatus.DEGRADED
                else:
                    component.status = HealthStatus.HEALTHY
                    component.last_healthy = datetime.now(timezone.utc)
                
                # Update uptime
                component.uptime_seconds = time.time() - self._start_time
    
    def _calculate_overall_status(self) -> HealthStatus:
        """Calculate overall system status."""
        if not self._component_status:
            return HealthStatus.HEALTHY
        
        statuses = [c.status for c in self._component_status.values()]
        
        # Critical components make system critical
        critical_checks = [
            check for check in self._health_checks.values() 
            if check.critical
        ]
        
        for check in critical_checks:
            result = self._last_results.get(check.name)
            if result and result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                return HealthStatus.CRITICAL
        
        # Aggregate non-critical statuses
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score."""
        # Get key performance metrics
        error_rate = metrics_collector.get_metric_value("error_rate") or 0.0
        success_rate = metrics_collector.get_metric_value("success_rate") or 1.0
        
        # Get system metrics
        cpu_usage = metrics_collector.get_metric_value("cpu_usage_percent") or 0.0
        memory_usage = metrics_collector.get_metric_value("memory_usage_percent") or 0.0
        
        # Calculate composite score
        error_score = max(0, 1 - error_rate)
        success_score = success_rate
        resource_score = max(0, 1 - ((cpu_usage + memory_usage) / 200))
        
        # Weighted average
        return (error_score * 0.4 + success_score * 0.4 + resource_score * 0.2)
    
    def _get_active_alerts(self) -> List[str]:
        """Get list of active alerts."""
        alerts = []
        
        # Check for critical health check failures
        for check_name, result in self._last_results.items():
            check = self._health_checks.get(check_name)
            if check and check.critical and result.status != HealthStatus.HEALTHY:
                alerts.append(f"Critical health check failed: {check_name}")
        
        # Check for high error rates
        error_rate = metrics_collector.get_metric_value("error_rate") or 0.0
        if error_rate > 0.1:  # 10% error rate threshold
            alerts.append(f"High error rate: {error_rate:.1%}")
        
        # Check for resource usage
        cpu_usage = metrics_collector.get_metric_value("cpu_usage_percent") or 0.0
        memory_usage = metrics_collector.get_metric_value("memory_usage_percent") or 0.0
        
        if cpu_usage > 90:
            alerts.append(f"High CPU usage: {cpu_usage:.1f}%")
        
        if memory_usage > 90:
            alerts.append(f"High memory usage: {memory_usage:.1f}%")
        
        return alerts


@dataclass
class HealthEndpoint:
    """
    HTTP endpoint for health status reporting.
    
    Provides REST API endpoints for health checks and system status
    with support for different response formats.
    """
    
    health_checker: HealthChecker = field(default_factory=HealthChecker)
    
    async def health_check_endpoint(self) -> Dict[str, Any]:
        """Basic health check endpoint."""
        status = self.health_checker.get_system_status()
        
        return {
            "status": status.status.value,
            "timestamp": status.timestamp.isoformat(),
            "uptime_seconds": status.uptime_seconds
        }
    
    async def detailed_status_endpoint(self) -> Dict[str, Any]:
        """Detailed system status endpoint."""
        status = self.health_checker.get_system_status()
        return status.model_dump()
    
    async def component_status_endpoint(self, component_name: str) -> Dict[str, Any]:
        """Component-specific status endpoint."""
        component = self.health_checker.get_component_status(component_name)
        
        if not component:
            return {
                "error": "Component not found",
                "component_name": component_name
            }
        
        return component.model_dump()
    
    async def metrics_endpoint(self) -> Dict[str, Any]:
        """Metrics endpoint."""
        from .metrics import metrics_exporter
        return metrics_exporter.export_json_format()
    
    async def prometheus_endpoint(self) -> str:
        """Prometheus metrics endpoint."""
        from .metrics import metrics_exporter
        return metrics_exporter.export_prometheus_format()


# Global health checker instance
health_checker = HealthChecker()
health_endpoint = HealthEndpoint(health_checker=health_checker)