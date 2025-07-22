"""
Tests for health monitoring system.
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.dataqa.orchestration.monitoring.health import (
    HealthChecker,
    HealthEndpoint,
    HealthCheck,
    HealthCheckResult,
    ComponentStatus,
    SystemStatus,
    HealthStatus,
    ComponentType
)


class TestHealthChecker:
    """Test health checking functionality."""
    
    @pytest.fixture
    def health_checker(self):
        """Create a health checker for testing."""
        return HealthChecker()
    
    def test_register_health_check(self, health_checker):
        """Test registering health checks."""
        check = HealthCheck(
            name="test_check",
            component_type=ComponentType.DATABASE,
            description="Test health check",
            timeout_seconds=5.0,
            interval_seconds=30.0
        )
        
        async def check_function():
            return HealthCheckResult(
                check_name="test_check",
                status=HealthStatus.HEALTHY,
                message="All good",
                duration_ms=10.0
            )
        
        health_checker.register_health_check(check, check_function)
        
        assert "test_check" in health_checker._health_checks
        assert "test_check" in health_checker._check_functions
        assert health_checker._health_checks["test_check"] == check
    
    def test_register_component(self, health_checker):
        """Test registering system components."""
        health_checker.register_component(
            "database",
            ComponentType.DATABASE,
            metadata={"host": "localhost", "port": 5432}
        )
        
        assert "database" in health_checker._component_status
        component = health_checker._component_status["database"]
        assert component.component_name == "database"
        assert component.component_type == ComponentType.DATABASE
        assert component.status == HealthStatus.HEALTHY
        assert component.metadata["host"] == "localhost"
    
    @pytest.mark.asyncio
    async def test_run_health_check_success(self, health_checker):
        """Test running successful health check."""
        async def successful_check():
            return HealthCheckResult(
                check_name="success_check",
                status=HealthStatus.HEALTHY,
                message="Check passed",
                duration_ms=15.0
            )
        
        check = HealthCheck(
            name="success_check",
            component_type=ComponentType.SYSTEM,
            description="Successful check"
        )
        
        health_checker.register_health_check(check, successful_check)
        
        result = await health_checker.run_health_check("success_check")
        
        assert result.check_name == "success_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Check passed"
        assert result.duration_ms > 0
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_run_health_check_failure(self, health_checker):
        """Test running failed health check."""
        async def failing_check():
            raise Exception("Check failed")
        
        check = HealthCheck(
            name="fail_check",
            component_type=ComponentType.SYSTEM,
            description="Failing check"
        )
        
        health_checker.register_health_check(check, failing_check)
        
        result = await health_checker.run_health_check("fail_check")
        
        assert result.check_name == "fail_check"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Check failed" in result.message
        assert result.error == "Check failed"
    
    @pytest.mark.asyncio
    async def test_run_health_check_timeout(self, health_checker):
        """Test health check timeout."""
        async def slow_check():
            await asyncio.sleep(1.0)  # Longer than timeout
            return HealthCheckResult(
                check_name="slow_check",
                status=HealthStatus.HEALTHY,
                message="Should timeout",
                duration_ms=1000.0
            )
        
        check = HealthCheck(
            name="slow_check",
            component_type=ComponentType.SYSTEM,
            description="Slow check",
            timeout_seconds=0.1  # Very short timeout
        )
        
        health_checker.register_health_check(check, slow_check)
        
        result = await health_checker.run_health_check("slow_check")
        
        assert result.check_name == "slow_check"
        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message
        assert result.error == "Timeout"
    
    @pytest.mark.asyncio
    async def test_run_health_check_not_found(self, health_checker):
        """Test running non-existent health check."""
        result = await health_checker.run_health_check("non_existent")
        
        assert result.check_name == "non_existent"
        assert result.status == HealthStatus.UNHEALTHY
        assert "not found" in result.message
        assert result.error == "Check not registered"
    
    @pytest.mark.asyncio
    async def test_run_all_health_checks(self, health_checker):
        """Test running all health checks."""
        # Register multiple checks
        async def check1():
            return HealthCheckResult(
                check_name="check1",
                status=HealthStatus.HEALTHY,
                message="OK",
                duration_ms=10.0
            )
        
        async def check2():
            return HealthCheckResult(
                check_name="check2",
                status=HealthStatus.DEGRADED,
                message="Slow",
                duration_ms=50.0
            )
        
        health_checker.register_health_check(
            HealthCheck(name="check1", component_type=ComponentType.SYSTEM, description="Check 1"),
            check1
        )
        health_checker.register_health_check(
            HealthCheck(name="check2", component_type=ComponentType.SYSTEM, description="Check 2"),
            check2
        )
        
        results = await health_checker.run_all_health_checks()
        
        assert len(results) == 2
        assert any(r.check_name == "check1" and r.status == HealthStatus.HEALTHY for r in results)
        assert any(r.check_name == "check2" and r.status == HealthStatus.DEGRADED for r in results)
    
    @pytest.mark.asyncio
    async def test_run_all_health_checks_disabled(self, health_checker):
        """Test that disabled checks are skipped."""
        async def enabled_check():
            return HealthCheckResult(
                check_name="enabled",
                status=HealthStatus.HEALTHY,
                message="OK",
                duration_ms=10.0
            )
        
        async def disabled_check():
            return HealthCheckResult(
                check_name="disabled",
                status=HealthStatus.HEALTHY,
                message="Should not run",
                duration_ms=10.0
            )
        
        health_checker.register_health_check(
            HealthCheck(name="enabled", component_type=ComponentType.SYSTEM, description="Enabled", enabled=True),
            enabled_check
        )
        health_checker.register_health_check(
            HealthCheck(name="disabled", component_type=ComponentType.SYSTEM, description="Disabled", enabled=False),
            disabled_check
        )
        
        results = await health_checker.run_all_health_checks()
        
        assert len(results) == 1
        assert results[0].check_name == "enabled"
    
    @patch('src.dataqa.orchestration.monitoring.health.metrics_collector')
    def test_get_system_status(self, mock_metrics, health_checker):
        """Test getting system status."""
        # Mock metrics
        mock_metrics.get_metric_value.side_effect = lambda name: {
            "concurrent_executions": 5,
            "total_executions": 100,
            "error_rate": 0.02
        }.get(name, 0)
        
        # Register a component
        health_checker.register_component("test_component", ComponentType.SYSTEM)
        
        status = health_checker.get_system_status()
        
        assert isinstance(status, SystemStatus)
        assert status.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
        assert status.uptime_seconds > 0
        assert status.active_executions == 5
        assert status.total_executions == 100
        assert status.error_rate == 0.02
        assert len(status.components) == 1
    
    def test_get_component_status(self, health_checker):
        """Test getting component status."""
        health_checker.register_component("test_component", ComponentType.DATABASE)
        
        status = health_checker.get_component_status("test_component")
        assert status is not None
        assert status.component_name == "test_component"
        assert status.component_type == ComponentType.DATABASE
        
        # Non-existent component
        assert health_checker.get_component_status("non_existent") is None
    
    def test_calculate_overall_status(self, health_checker):
        """Test overall status calculation."""
        # Register components with different statuses
        health_checker.register_component("healthy", ComponentType.SYSTEM)
        health_checker.register_component("degraded", ComponentType.CACHE)
        health_checker.register_component("unhealthy", ComponentType.DATABASE)
        
        # Set component statuses
        health_checker._component_status["healthy"].status = HealthStatus.HEALTHY
        health_checker._component_status["degraded"].status = HealthStatus.DEGRADED
        health_checker._component_status["unhealthy"].status = HealthStatus.UNHEALTHY
        
        overall_status = health_checker._calculate_overall_status()
        
        # Should be unhealthy due to unhealthy component
        assert overall_status == HealthStatus.UNHEALTHY
    
    def test_calculate_overall_status_critical(self, health_checker):
        """Test overall status with critical check failure."""
        # Register critical health check
        async def critical_check():
            return HealthCheckResult(
                check_name="critical",
                status=HealthStatus.UNHEALTHY,
                message="Critical failure",
                duration_ms=10.0
            )
        
        check = HealthCheck(
            name="critical",
            component_type=ComponentType.SYSTEM,
            description="Critical check",
            critical=True
        )
        
        health_checker.register_health_check(check, critical_check)
        
        # Simulate check result
        health_checker._last_results["critical"] = HealthCheckResult(
            check_name="critical",
            status=HealthStatus.UNHEALTHY,
            message="Critical failure",
            duration_ms=10.0
        )
        
        overall_status = health_checker._calculate_overall_status()
        assert overall_status == HealthStatus.CRITICAL
    
    @patch('src.dataqa.orchestration.monitoring.health.metrics_collector')
    def test_calculate_performance_score(self, mock_metrics, health_checker):
        """Test performance score calculation."""
        # Mock metrics for good performance
        mock_metrics.get_metric_value.side_effect = lambda name: {
            "error_rate": 0.01,
            "success_rate": 0.99,
            "cpu_usage_percent": 20.0,
            "memory_usage_percent": 30.0
        }.get(name, 0)
        
        score = health_checker._calculate_performance_score()
        
        assert 0 <= score <= 1
        assert score > 0.8  # Should be high for good metrics
    
    @patch('src.dataqa.orchestration.monitoring.health.metrics_collector')
    def test_get_active_alerts(self, mock_metrics, health_checker):
        """Test getting active alerts."""
        # Mock high error rate
        mock_metrics.get_metric_value.side_effect = lambda name: {
            "error_rate": 0.15,  # 15% error rate
            "cpu_usage_percent": 95.0,  # High CPU
            "memory_usage_percent": 85.0  # High memory
        }.get(name, 0)
        
        # Add critical check failure
        health_checker.register_health_check(
            HealthCheck(name="critical", component_type=ComponentType.SYSTEM, description="Critical", critical=True),
            AsyncMock()
        )
        health_checker._last_results["critical"] = HealthCheckResult(
            check_name="critical",
            status=HealthStatus.UNHEALTHY,
            message="Failed",
            duration_ms=10.0
        )
        
        alerts = health_checker._get_active_alerts()
        
        assert len(alerts) > 0
        assert any("Critical health check failed" in alert for alert in alerts)
        assert any("High error rate" in alert for alert in alerts)
        assert any("High CPU usage" in alert for alert in alerts)
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, health_checker):
        """Test starting and stopping monitoring."""
        assert health_checker._monitoring_task is None
        
        # Start monitoring
        health_checker.start_monitoring()
        assert health_checker._monitoring_task is not None
        assert health_checker._monitoring_enabled is True
        
        # Let it run briefly
        await asyncio.sleep(0.01)
        
        # Stop monitoring
        health_checker.stop_monitoring()
        assert health_checker._monitoring_enabled is False
        
        # Wait for task to complete
        if health_checker._monitoring_task:
            try:
                await asyncio.wait_for(health_checker._monitoring_task, timeout=1.0)
            except asyncio.CancelledError:
                pass


class TestHealthEndpoint:
    """Test health endpoint functionality."""
    
    @pytest.fixture
    def health_endpoint(self):
        """Create a health endpoint for testing."""
        health_checker = HealthChecker()
        return HealthEndpoint(health_checker=health_checker)
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, health_endpoint):
        """Test basic health check endpoint."""
        response = await health_endpoint.health_check_endpoint()
        
        assert "status" in response
        assert "timestamp" in response
        assert "uptime_seconds" in response
        assert response["status"] in ["healthy", "degraded", "unhealthy", "critical"]
    
    @pytest.mark.asyncio
    async def test_detailed_status_endpoint(self, health_endpoint):
        """Test detailed status endpoint."""
        response = await health_endpoint.detailed_status_endpoint()
        
        assert "status" in response
        assert "timestamp" in response
        assert "uptime_seconds" in response
        assert "components" in response
        assert "active_agents" in response
        assert "active_executions" in response
        assert "performance_score" in response
    
    @pytest.mark.asyncio
    async def test_component_status_endpoint(self, health_endpoint):
        """Test component status endpoint."""
        # Register a component
        health_endpoint.health_checker.register_component("test_db", ComponentType.DATABASE)
        
        response = await health_endpoint.component_status_endpoint("test_db")
        
        assert "component_name" in response
        assert "component_type" in response
        assert "status" in response
        assert response["component_name"] == "test_db"
    
    @pytest.mark.asyncio
    async def test_component_status_endpoint_not_found(self, health_endpoint):
        """Test component status endpoint for non-existent component."""
        response = await health_endpoint.component_status_endpoint("non_existent")
        
        assert "error" in response
        assert response["error"] == "Component not found"
        assert response["component_name"] == "non_existent"
    
    @pytest.mark.asyncio
    @patch('src.dataqa.orchestration.monitoring.health.metrics_exporter')
    async def test_metrics_endpoint(self, mock_exporter, health_endpoint):
        """Test metrics endpoint."""
        mock_exporter.export_json_format.return_value = {"test": "data"}
        
        response = await health_endpoint.metrics_endpoint()
        
        assert response == {"test": "data"}
        mock_exporter.export_json_format.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.dataqa.orchestration.monitoring.health.metrics_exporter')
    async def test_prometheus_endpoint(self, mock_exporter, health_endpoint):
        """Test Prometheus endpoint."""
        mock_exporter.export_prometheus_format.return_value = "# Prometheus metrics"
        
        response = await health_endpoint.prometheus_endpoint()
        
        assert response == "# Prometheus metrics"
        mock_exporter.export_prometheus_format.assert_called_once()


class TestHealthModels:
    """Test health monitoring model classes."""
    
    def test_health_check_creation(self):
        """Test creating HealthCheck instance."""
        check = HealthCheck(
            name="test_check",
            component_type=ComponentType.DATABASE,
            description="Test database check",
            timeout_seconds=10.0,
            interval_seconds=60.0,
            critical=True
        )
        
        assert check.name == "test_check"
        assert check.component_type == ComponentType.DATABASE
        assert check.description == "Test database check"
        assert check.timeout_seconds == 10.0
        assert check.interval_seconds == 60.0
        assert check.critical is True
        assert check.enabled is True  # Default
    
    def test_health_check_result_creation(self):
        """Test creating HealthCheckResult instance."""
        result = HealthCheckResult(
            check_name="test_check",
            status=HealthStatus.HEALTHY,
            message="All systems operational",
            duration_ms=25.5
        )
        
        assert result.check_name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All systems operational"
        assert result.duration_ms == 25.5
        assert result.timestamp is not None
        assert result.error is None
    
    def test_component_status_creation(self):
        """Test creating ComponentStatus instance."""
        status = ComponentStatus(
            component_name="database",
            component_type=ComponentType.DATABASE,
            status=HealthStatus.HEALTHY
        )
        
        assert status.component_name == "database"
        assert status.component_type == ComponentType.DATABASE
        assert status.status == HealthStatus.HEALTHY
        assert status.health_checks == []
        assert status.uptime_seconds == 0.0
        assert status.metadata == {}
    
    def test_system_status_creation(self):
        """Test creating SystemStatus instance."""
        status = SystemStatus(
            status=HealthStatus.HEALTHY,
            uptime_seconds=3600.0,
            active_agents=5,
            total_executions=1000,
            performance_score=0.95
        )
        
        assert status.status == HealthStatus.HEALTHY
        assert status.uptime_seconds == 3600.0
        assert status.active_agents == 5
        assert status.total_executions == 1000
        assert status.performance_score == 0.95
        assert status.timestamp is not None
        assert status.components == []
        assert status.alerts == []