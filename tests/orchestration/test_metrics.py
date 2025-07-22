"""
Tests for metrics collection and export system.
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.dataqa.orchestration.monitoring.metrics import (
    MetricsCollector,
    MetricsExporter,
    Metric,
    MetricType,
    MetricUnit,
    PerformanceMetrics,
    SystemMetrics,
    MetricsSnapshot
)


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    @pytest.fixture
    def collector(self):
        """Create a metrics collector for testing."""
        return MetricsCollector()
    
    def test_record_metric(self, collector):
        """Test recording a metric."""
        collector.record_metric(
            name="test_metric",
            value=42.5,
            labels={"component": "test"},
            metric_type=MetricType.GAUGE,
            unit=MetricUnit.COUNT,
            description="Test metric"
        )
        
        # Check metric was recorded
        metric_key = "test_metric{component=test}"
        assert metric_key in collector._metrics
        assert collector._metrics[metric_key] == [42.5]
        
        # Check metadata
        assert metric_key in collector._metric_metadata
        metadata = collector._metric_metadata[metric_key]
        assert metadata.name == "test_metric"
        assert metadata.type == MetricType.GAUGE
        assert metadata.unit == MetricUnit.COUNT
        assert metadata.value == 42.5
        assert metadata.labels == {"component": "test"}
    
    def test_record_metric_without_labels(self, collector):
        """Test recording metric without labels."""
        collector.record_metric("simple_metric", 10.0)
        
        assert "simple_metric" in collector._metrics
        assert collector._metrics["simple_metric"] == [10.0]
    
    def test_increment_counter(self, collector):
        """Test incrementing counter metrics."""
        collector.increment_counter("request_count", {"endpoint": "/api"})
        collector.increment_counter("request_count", {"endpoint": "/api"})
        collector.increment_counter("request_count", {"endpoint": "/api"})
        
        metric_key = "request_count{endpoint=/api}"
        values = collector._metrics[metric_key]
        assert values == [1.0, 2.0, 3.0]
        
        # Current value should be 3
        assert collector.get_metric_value("request_count", {"endpoint": "/api"}) == 3.0
    
    def test_set_gauge(self, collector):
        """Test setting gauge metrics."""
        collector.set_gauge("cpu_usage", 25.5, unit=MetricUnit.PERCENT)
        collector.set_gauge("cpu_usage", 30.2, unit=MetricUnit.PERCENT)
        
        # Should have both values recorded
        values = collector._metrics["cpu_usage"]
        assert values == [25.5, 30.2]
        
        # Current value should be latest
        assert collector.get_metric_value("cpu_usage") == 30.2
    
    def test_record_histogram(self, collector):
        """Test recording histogram values."""
        # Record multiple response times
        response_times = [100, 150, 200, 120, 180]
        for time_ms in response_times:
            collector.record_histogram("response_time", time_ms, unit=MetricUnit.MILLISECONDS)
        
        values = collector._metrics["response_time"]
        assert values == response_times
    
    def test_get_metric_statistics(self, collector):
        """Test getting metric statistics."""
        # Record some values
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for value in values:
            collector.record_metric("test_stat", value)
        
        stats = collector.get_metric_statistics("test_stat")
        
        assert stats["count"] == 10
        assert stats["sum"] == 550
        assert stats["min"] == 10
        assert stats["max"] == 100
        assert stats["mean"] == 55.0
        assert stats["p50"] == 50  # Median
        assert stats["p95"] == 90
        assert stats["p99"] == 90  # With only 10 values
    
    def test_get_metric_statistics_empty(self, collector):
        """Test getting statistics for non-existent metric."""
        stats = collector.get_metric_statistics("non_existent")
        assert stats == {}
    
    def test_get_time_series(self, collector):
        """Test getting time series data."""
        # Record values with time progression
        for i in range(5):
            collector.record_metric("time_series_test", i * 10)
            time.sleep(0.001)  # Small delay to ensure different timestamps
        
        time_series = collector.get_time_series("time_series_test", duration_minutes=1)
        
        assert len(time_series) == 5
        for i, point in enumerate(time_series):
            assert point["value"] == i * 10
            assert "timestamp" in point
    
    def test_get_time_series_with_duration_filter(self, collector):
        """Test time series filtering by duration."""
        # Record old value
        collector.record_metric("old_metric", 1.0)
        
        # Wait and record new value
        time.sleep(0.01)
        collector.record_metric("old_metric", 2.0)
        
        # Get very short duration (should only include recent)
        time_series = collector.get_time_series("old_metric", duration_minutes=0.001)
        
        # Should have at least the most recent value
        assert len(time_series) >= 1
        assert time_series[-1]["value"] == 2.0
    
    def test_clear_metrics(self, collector):
        """Test clearing all metrics."""
        collector.record_metric("test1", 1.0)
        collector.record_metric("test2", 2.0)
        
        assert len(collector._metrics) == 2
        assert len(collector._metric_metadata) == 2
        
        collector.clear_metrics()
        
        assert len(collector._metrics) == 0
        assert len(collector._metric_metadata) == 0
        assert len(collector._time_series) == 0
    
    def test_create_metric_key(self, collector):
        """Test metric key creation."""
        # Without labels
        key = collector._create_metric_key("simple", {})
        assert key == "simple"
        
        # With single label
        key = collector._create_metric_key("metric", {"key": "value"})
        assert key == "metric{key=value}"
        
        # With multiple labels (should be sorted)
        key = collector._create_metric_key("metric", {"b": "2", "a": "1"})
        assert key == "metric{a=1,b=2}"
    
    def test_thread_safety(self, collector):
        """Test thread safety of metrics collection."""
        import threading
        
        def worker(thread_id):
            for i in range(100):
                collector.record_metric(f"thread_metric_{thread_id}", i)
                collector.increment_counter("global_counter")
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        for i in range(5):
            metric_name = f"thread_metric_{i}"
            assert len(collector._metrics[metric_name]) == 100
        
        # Global counter should have been incremented 500 times
        global_count = collector.get_metric_value("global_counter")
        assert global_count == 500.0


class TestMetricsExporter:
    """Test metrics export functionality."""
    
    @pytest.fixture
    def exporter(self):
        """Create a metrics exporter for testing."""
        collector = MetricsCollector()
        return MetricsExporter(collector=collector)
    
    def test_export_prometheus_format(self, exporter):
        """Test Prometheus format export."""
        # Add some metrics
        exporter.collector.record_metric(
            "http_requests_total",
            100,
            labels={"method": "GET", "status": "200"},
            metric_type=MetricType.COUNTER,
            description="Total HTTP requests"
        )
        
        exporter.collector.set_gauge(
            "memory_usage_bytes",
            1024*1024*100,  # 100MB
            description="Memory usage in bytes"
        )
        
        prometheus_output = exporter.export_prometheus_format()
        
        # Check format
        lines = prometheus_output.split('\n')
        
        # Should contain help and type comments
        assert any("# HELP http_requests_total Total HTTP requests" in line for line in lines)
        assert any("# TYPE http_requests_total counter" in line for line in lines)
        assert any("# HELP memory_usage_bytes Memory usage in bytes" in line for line in lines)
        assert any("# TYPE memory_usage_bytes gauge" in line for line in lines)
        
        # Should contain metric values
        assert any('http_requests_total{method="GET",status="200"} 100' in line for line in lines)
        assert any("memory_usage_bytes 104857600" in line for line in lines)
    
    def test_export_json_format(self, exporter):
        """Test JSON format export."""
        # Add some metrics
        exporter.collector.record_metric("test_metric", 42.0)
        
        json_output = exporter.export_json_format()
        
        assert isinstance(json_output, dict)
        assert "timestamp" in json_output
        assert "system_metrics" in json_output
        assert "performance_metrics" in json_output
        assert "custom_metrics" in json_output
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    @patch('psutil.net_connections')
    def test_collect_system_metrics(self, mock_net_conn, mock_net_io, mock_disk, 
                                   mock_memory, mock_cpu, exporter):
        """Test system metrics collection."""
        # Mock psutil functions
        mock_cpu.return_value = 25.5
        mock_memory.return_value = Mock(used=1024*1024*100, percent=15.2)
        mock_disk.return_value = Mock(percent=45.8)
        mock_net_io.return_value = Mock()
        mock_net_conn.return_value = [Mock(), Mock(), Mock()]  # 3 connections
        
        system_metrics = exporter._collect_system_metrics()
        
        assert system_metrics.cpu_usage_percent == 25.5
        assert system_metrics.memory_usage_mb == 100.0
        assert system_metrics.memory_usage_percent == 15.2
        assert system_metrics.disk_usage_percent == 45.8
        assert system_metrics.active_connections == 3
        assert system_metrics.uptime_seconds > 0
        assert 0 <= system_metrics.health_score <= 1
    
    def test_collect_performance_metrics(self, exporter):
        """Test performance metrics collection."""
        # Set up some metrics in collector
        exporter.collector.record_metric("execution_duration", 100)
        exporter.collector.record_metric("execution_duration", 200)
        exporter.collector.record_metric("execution_duration", 150)
        
        exporter.collector.set_gauge("success_rate", 0.95)
        exporter.collector.set_gauge("error_rate", 0.05)
        exporter.collector.set_gauge("throughput_per_minute", 120.0)
        exporter.collector.set_gauge("concurrent_executions", 5)
        
        performance_metrics = exporter._collect_performance_metrics()
        
        assert performance_metrics.execution_count == 3
        assert performance_metrics.average_duration_ms == 150.0  # (100+200+150)/3
        assert performance_metrics.success_rate == 0.95
        assert performance_metrics.error_rate == 0.05
        assert performance_metrics.throughput_per_minute == 120.0
        assert performance_metrics.concurrent_executions == 5
    
    def test_calculate_health_score(self, exporter):
        """Test health score calculation."""
        # Test healthy system
        score = exporter._calculate_health_score(10.0, 20.0, 30.0)
        assert score > 0.5  # Should be healthy
        
        # Test unhealthy system
        score = exporter._calculate_health_score(95.0, 90.0, 85.0)
        assert score < 0.5  # Should be unhealthy
        
        # Test perfect system
        score = exporter._calculate_health_score(0.0, 0.0, 0.0)
        assert score == 1.0
    
    def test_create_snapshot(self, exporter):
        """Test creating metrics snapshot."""
        # Add some metrics
        exporter.collector.record_metric("test_metric", 42.0)
        
        snapshot = exporter.create_snapshot()
        
        assert isinstance(snapshot, MetricsSnapshot)
        assert snapshot.timestamp is not None
        assert isinstance(snapshot.system_metrics, SystemMetrics)
        assert isinstance(snapshot.performance_metrics, PerformanceMetrics)
        assert len(snapshot.custom_metrics) > 0
    
    def test_register_export_handler(self, exporter):
        """Test registering custom export handlers."""
        handler_called = False
        
        def custom_handler(snapshot):
            nonlocal handler_called
            handler_called = True
            assert isinstance(snapshot, MetricsSnapshot)
        
        exporter.register_export_handler("custom", custom_handler)
        
        # Export should call the handler
        exporter.export_to_handlers()
        
        assert handler_called
    
    def test_export_interval(self, exporter):
        """Test export interval enforcement."""
        handler_call_count = 0
        
        def counting_handler(snapshot):
            nonlocal handler_call_count
            handler_call_count += 1
        
        exporter.register_export_handler("counter", counting_handler)
        exporter.export_interval_seconds = 1  # 1 second interval
        
        # First export should work
        exporter.export_to_handlers()
        assert handler_call_count == 1
        
        # Immediate second export should be skipped
        exporter.export_to_handlers()
        assert handler_call_count == 1
        
        # After waiting, should work again
        exporter._last_export = time.time() - 2  # Fake older timestamp
        exporter.export_to_handlers()
        assert handler_call_count == 2


class TestMetricModels:
    """Test metric model classes."""
    
    def test_metric_creation(self):
        """Test creating Metric instance."""
        metric = Metric(
            name="test_metric",
            type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="Test metric",
            labels={"component": "test"},
            value=42.0
        )
        
        assert metric.name == "test_metric"
        assert metric.type == MetricType.COUNTER
        assert metric.unit == MetricUnit.COUNT
        assert metric.value == 42.0
        assert metric.labels == {"component": "test"}
        assert metric.timestamp is not None
    
    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics instance."""
        metrics = PerformanceMetrics(
            execution_count=100,
            success_rate=0.95,
            average_duration_ms=150.5,
            error_rate=0.05
        )
        
        assert metrics.execution_count == 100
        assert metrics.success_rate == 0.95
        assert metrics.average_duration_ms == 150.5
        assert metrics.error_rate == 0.05
    
    def test_system_metrics_creation(self):
        """Test creating SystemMetrics instance."""
        metrics = SystemMetrics(
            cpu_usage_percent=25.5,
            memory_usage_mb=100.0,
            memory_usage_percent=15.2,
            health_score=0.85
        )
        
        assert metrics.cpu_usage_percent == 25.5
        assert metrics.memory_usage_mb == 100.0
        assert metrics.memory_usage_percent == 15.2
        assert metrics.health_score == 0.85
    
    def test_metrics_snapshot_creation(self):
        """Test creating MetricsSnapshot instance."""
        snapshot = MetricsSnapshot()
        
        assert snapshot.timestamp is not None
        assert isinstance(snapshot.system_metrics, SystemMetrics)
        assert isinstance(snapshot.performance_metrics, PerformanceMetrics)
        assert isinstance(snapshot.agent_metrics, dict)
        assert isinstance(snapshot.custom_metrics, dict)