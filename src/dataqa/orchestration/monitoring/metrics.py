"""
Performance metrics collection and export system.

Provides comprehensive metrics collection for monitoring systems integration
with support for various export formats and monitoring platforms.
"""

import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pydantic import BaseModel, Field
from enum import Enum

from .telemetry import AgentMetrics, ExecutionTelemetry, ResourceUsage


class MetricType(str, Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricUnit(str, Enum):
    """Units for metrics."""
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    BYTES = "bytes"
    MEGABYTES = "megabytes"
    PERCENT = "percent"
    COUNT = "count"
    RATE = "rate"


class Metric(BaseModel):
    """Individual metric definition."""
    
    name: str = Field(description="Metric name")
    type: MetricType = Field(description="Metric type")
    unit: MetricUnit = Field(description="Metric unit")
    description: str = Field(description="Metric description")
    labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels")
    value: float = Field(description="Current metric value")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PerformanceMetrics(BaseModel):
    """Performance metrics for agent execution."""
    
    execution_count: int = Field(default=0, description="Total executions")
    success_rate: float = Field(default=0.0, description="Success rate (0-1)")
    average_duration_ms: float = Field(default=0.0, description="Average execution duration")
    p95_duration_ms: float = Field(default=0.0, description="95th percentile duration")
    p99_duration_ms: float = Field(default=0.0, description="99th percentile duration")
    error_rate: float = Field(default=0.0, description="Error rate (0-1)")
    throughput_per_minute: float = Field(default=0.0, description="Executions per minute")
    concurrent_executions: int = Field(default=0, description="Current concurrent executions")
    queue_depth: int = Field(default=0, description="Current queue depth")
    resource_utilization: float = Field(default=0.0, description="Resource utilization (0-1)")


class SystemMetrics(BaseModel):
    """System-level metrics."""
    
    cpu_usage_percent: float = Field(default=0.0, description="CPU usage percentage")
    memory_usage_mb: float = Field(default=0.0, description="Memory usage in MB")
    memory_usage_percent: float = Field(default=0.0, description="Memory usage percentage")
    disk_usage_percent: float = Field(default=0.0, description="Disk usage percentage")
    network_io_mb_per_sec: float = Field(default=0.0, description="Network I/O MB/sec")
    active_connections: int = Field(default=0, description="Active network connections")
    uptime_seconds: float = Field(default=0.0, description="System uptime in seconds")
    health_score: float = Field(default=1.0, description="Overall health score (0-1)")


class MetricsSnapshot(BaseModel):
    """Point-in-time snapshot of all metrics."""
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    system_metrics: SystemMetrics = Field(default_factory=SystemMetrics)
    performance_metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    agent_metrics: Dict[str, AgentMetrics] = Field(default_factory=dict)
    custom_metrics: Dict[str, Metric] = Field(default_factory=dict)


@dataclass
class MetricsCollector:
    """
    Collects and aggregates metrics from various sources.
    
    Provides thread-safe collection of performance, system, and custom metrics
    with support for time-series data and statistical aggregation.
    """
    
    _metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    _metric_metadata: Dict[str, Metric] = field(default_factory=dict)
    _time_series: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=1000)))
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _start_time: float = field(default_factory=time.time)
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None,
                     metric_type: MetricType = MetricType.GAUGE,
                     unit: MetricUnit = MetricUnit.COUNT,
                     description: str = "") -> None:
        """Record a metric value."""
        with self._lock:
            # Create metric key with labels
            metric_key = self._create_metric_key(name, labels or {})
            
            # Store metric metadata
            if metric_key not in self._metric_metadata:
                self._metric_metadata[metric_key] = Metric(
                    name=name,
                    type=metric_type,
                    unit=unit,
                    description=description,
                    labels=labels or {},
                    value=value
                )
            else:
                self._metric_metadata[metric_key].value = value
                self._metric_metadata[metric_key].timestamp = datetime.now(timezone.utc)
            
            # Store value for aggregation
            self._metrics[metric_key].append(value)
            
            # Store in time series
            self._time_series[metric_key].append({
                'timestamp': time.time(),
                'value': value
            })
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None,
                         description: str = "") -> None:
        """Increment a counter metric."""
        metric_key = self._create_metric_key(name, labels or {})
        current_value = self._get_current_value(metric_key)
        self.record_metric(
            name, current_value + 1, labels, 
            MetricType.COUNTER, MetricUnit.COUNT, description
        )
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None,
                  unit: MetricUnit = MetricUnit.COUNT, description: str = "") -> None:
        """Set a gauge metric value."""
        self.record_metric(name, value, labels, MetricType.GAUGE, unit, description)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None,
                        unit: MetricUnit = MetricUnit.MILLISECONDS, description: str = "") -> None:
        """Record a histogram value."""
        self.record_metric(name, value, labels, MetricType.HISTOGRAM, unit, description)
    
    def get_metric_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current value of a metric."""
        metric_key = self._create_metric_key(name, labels or {})
        return self._get_current_value(metric_key)
    
    def get_metric_statistics(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        metric_key = self._create_metric_key(name, labels or {})
        
        with self._lock:
            values = self._metrics.get(metric_key, [])
            
            if not values:
                return {}
            
            sorted_values = sorted(values)
            count = len(sorted_values)
            
            return {
                'count': count,
                'sum': sum(sorted_values),
                'min': sorted_values[0],
                'max': sorted_values[-1],
                'mean': sum(sorted_values) / count,
                'p50': sorted_values[int(count * 0.5)],
                'p95': sorted_values[int(count * 0.95)],
                'p99': sorted_values[int(count * 0.99)]
            }
    
    def get_time_series(self, name: str, labels: Optional[Dict[str, str]] = None,
                       duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get time series data for a metric."""
        metric_key = self._create_metric_key(name, labels or {})
        
        with self._lock:
            time_series = self._time_series.get(metric_key, deque())
            cutoff_time = time.time() - (duration_minutes * 60)
            
            return [
                point for point in time_series 
                if point['timestamp'] >= cutoff_time
            ]
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._metrics.clear()
            self._metric_metadata.clear()
            self._time_series.clear()
    
    def _create_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _get_current_value(self, metric_key: str) -> float:
        """Get current value of a metric."""
        with self._lock:
            values = self._metrics.get(metric_key, [])
            return values[-1] if values else 0.0


@dataclass
class MetricsExporter:
    """
    Exports metrics to various monitoring systems.
    
    Supports multiple export formats and monitoring platforms
    with configurable export intervals and filtering.
    """
    
    collector: MetricsCollector = field(default_factory=MetricsCollector)
    export_interval_seconds: int = 60
    _export_handlers: Dict[str, Callable] = field(default_factory=dict)
    _last_export: float = field(default_factory=time.time)
    
    def register_export_handler(self, name: str, handler: Callable[[MetricsSnapshot], None]) -> None:
        """Register a custom export handler."""
        self._export_handlers[name] = handler
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self.collector._lock:
            for metric_key, metric in self.collector._metric_metadata.items():
                # Add help and type comments
                lines.append(f"# HELP {metric.name} {metric.description}")
                lines.append(f"# TYPE {metric.name} {metric.type.value}")
                
                # Add metric value with labels
                if metric.labels:
                    label_str = ",".join(f'{k}="{v}"' for k, v in metric.labels.items())
                    lines.append(f"{metric.name}{{{label_str}}} {metric.value}")
                else:
                    lines.append(f"{metric.name} {metric.value}")
        
        return "\n".join(lines)
    
    def export_json_format(self) -> Dict[str, Any]:
        """Export metrics in JSON format."""
        snapshot = self.create_snapshot()
        return snapshot.model_dump()
    
    def create_snapshot(self) -> MetricsSnapshot:
        """Create a comprehensive metrics snapshot."""
        snapshot = MetricsSnapshot()
        
        # Collect system metrics
        snapshot.system_metrics = self._collect_system_metrics()
        
        # Collect performance metrics
        snapshot.performance_metrics = self._collect_performance_metrics()
        
        # Add custom metrics
        with self.collector._lock:
            for metric_key, metric in self.collector._metric_metadata.items():
                snapshot.custom_metrics[metric_key] = metric
        
        return snapshot
    
    def export_to_handlers(self) -> None:
        """Export metrics to all registered handlers."""
        if time.time() - self._last_export < self.export_interval_seconds:
            return
        
        snapshot = self.create_snapshot()
        
        for name, handler in self._export_handlers.items():
            try:
                handler(snapshot)
            except Exception as e:
                # Log error but continue with other handlers
                print(f"Error exporting to {name}: {e}")
        
        self._last_export = time.time()
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            import psutil
            
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O (simplified)
            net_io = psutil.net_io_counters()
            
            # Uptime
            uptime = time.time() - self.collector._start_time
            
            return SystemMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=memory.used / (1024 * 1024),
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_io_mb_per_sec=0.0,  # Would need rate calculation
                active_connections=len(psutil.net_connections()),
                uptime_seconds=uptime,
                health_score=self._calculate_health_score(cpu_percent, memory.percent, disk.percent)
            )
        except Exception:
            return SystemMetrics()
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect performance metrics from collector."""
        metrics = PerformanceMetrics()
        
        # Get execution statistics
        execution_stats = self.collector.get_metric_statistics("execution_duration")
        if execution_stats:
            metrics.execution_count = int(execution_stats.get('count', 0))
            metrics.average_duration_ms = execution_stats.get('mean', 0.0)
            metrics.p95_duration_ms = execution_stats.get('p95', 0.0)
            metrics.p99_duration_ms = execution_stats.get('p99', 0.0)
        
        # Get success rate
        success_rate = self.collector.get_metric_value("success_rate")
        if success_rate is not None:
            metrics.success_rate = success_rate
        
        # Get error rate
        error_rate = self.collector.get_metric_value("error_rate")
        if error_rate is not None:
            metrics.error_rate = error_rate
        
        # Get throughput
        throughput = self.collector.get_metric_value("throughput_per_minute")
        if throughput is not None:
            metrics.throughput_per_minute = throughput
        
        # Get concurrent executions
        concurrent = self.collector.get_metric_value("concurrent_executions")
        if concurrent is not None:
            metrics.concurrent_executions = int(concurrent)
        
        return metrics
    
    def _calculate_health_score(self, cpu_percent: float, memory_percent: float, 
                               disk_percent: float) -> float:
        """Calculate overall system health score."""
        # Simple health scoring based on resource usage
        cpu_score = max(0, 1 - (cpu_percent / 100))
        memory_score = max(0, 1 - (memory_percent / 100))
        disk_score = max(0, 1 - (disk_percent / 100))
        
        # Weighted average
        return (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)


# Global metrics instances
metrics_collector = MetricsCollector()
metrics_exporter = MetricsExporter(collector=metrics_collector)