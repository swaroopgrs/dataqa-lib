"""
Tests for alert management system.
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.dataqa.orchestration.monitoring.alerts import (
    AlertManager,
    AlertRule,
    AlertCondition,
    Alert,
    NotificationChannel,
    AlertSeverity,
    AlertState,
    email_notification_handler,
    webhook_notification_handler
)


class TestAlertManager:
    """Test alert management functionality."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create an alert manager for testing."""
        return AlertManager()
    
    def test_register_alert_rule(self, alert_manager):
        """Test registering alert rules."""
        condition = AlertCondition(
            metric_name="error_rate",
            operator=">",
            threshold=0.1,
            duration_seconds=60.0
        )
        
        rule = AlertRule(
            name="high_error_rate",
            description="High error rate detected",
            severity=AlertSeverity.ERROR,
            conditions=[condition],
            notification_channels=["email"]
        )
        
        alert_manager.register_alert_rule(rule)
        
        assert "high_error_rate" in alert_manager._alert_rules
        assert alert_manager._alert_rules["high_error_rate"] == rule
    
    def test_register_notification_channel(self, alert_manager):
        """Test registering notification channels."""
        channel = NotificationChannel(
            name="email",
            type="email",
            config={"smtp_server": "localhost", "from": "alerts@test.com"},
            severity_filter=[AlertSeverity.ERROR, AlertSeverity.CRITICAL]
        )
        
        alert_manager.register_notification_channel(channel)
        
        assert "email" in alert_manager._notification_channels
        assert alert_manager._notification_channels["email"] == channel
    
    def test_register_notification_handler(self, alert_manager):
        """Test registering notification handlers."""
        async def custom_handler(alert, channel):
            pass
        
        alert_manager.register_notification_handler("custom", custom_handler)
        
        assert "custom" in alert_manager._notification_handlers
        assert alert_manager._notification_handlers["custom"] == custom_handler
    
    @pytest.mark.asyncio
    @patch('src.dataqa.orchestration.monitoring.alerts.metrics_collector')
    async def test_evaluate_condition_true(self, mock_metrics, alert_manager):
        """Test evaluating condition that is true."""
        # Mock metric value above threshold
        mock_metrics.get_metric_value.return_value = 0.15  # 15% error rate
        
        condition = AlertCondition(
            metric_name="error_rate",
            operator=">",
            threshold=0.1,
            duration_seconds=0.1  # Short duration for testing
        )
        
        # First evaluation - should be false due to duration
        is_met, details = await alert_manager._evaluate_condition(condition, datetime.now(timezone.utc))
        assert details["result"] is True  # Condition is met
        assert details["duration_met"] is False  # But not for required duration
        
        # Wait and evaluate again
        await asyncio.sleep(0.15)
        is_met, details = await alert_manager._evaluate_condition(condition, datetime.now(timezone.utc))
        assert details["result"] is True
        assert details["duration_met"] is True  # Now duration is met
    
    @pytest.mark.asyncio
    @patch('src.dataqa.orchestration.monitoring.alerts.metrics_collector')
    async def test_evaluate_condition_false(self, mock_metrics, alert_manager):
        """Test evaluating condition that is false."""
        # Mock metric value below threshold
        mock_metrics.get_metric_value.return_value = 0.05  # 5% error rate
        
        condition = AlertCondition(
            metric_name="error_rate",
            operator=">",
            threshold=0.1,
            duration_seconds=60.0
        )
        
        is_met, details = await alert_manager._evaluate_condition(condition, datetime.now(timezone.utc))
        
        assert details["result"] is False
        assert details["value"] == 0.05
        assert details["threshold"] == 0.1
        assert details["operator"] == ">"
    
    @pytest.mark.asyncio
    @patch('src.dataqa.orchestration.monitoring.alerts.metrics_collector')
    async def test_evaluate_condition_metric_not_found(self, mock_metrics, alert_manager):
        """Test evaluating condition when metric doesn't exist."""
        mock_metrics.get_metric_value.return_value = None
        
        condition = AlertCondition(
            metric_name="non_existent_metric",
            operator=">",
            threshold=0.1,
            duration_seconds=60.0
        )
        
        is_met, details = await alert_manager._evaluate_condition(condition, datetime.now(timezone.utc))
        
        assert details["result"] is False
        assert details["value"] is None
        assert details["reason"] == "Metric not found"
    
    def test_compare_values(self, alert_manager):
        """Test value comparison operators."""
        assert alert_manager._compare_values(10, ">", 5) is True
        assert alert_manager._compare_values(5, ">", 10) is False
        
        assert alert_manager._compare_values(5, "<", 10) is True
        assert alert_manager._compare_values(10, "<", 5) is False
        
        assert alert_manager._compare_values(10, ">=", 10) is True
        assert alert_manager._compare_values(10, ">=", 5) is True
        assert alert_manager._compare_values(5, ">=", 10) is False
        
        assert alert_manager._compare_values(5, "<=", 10) is True
        assert alert_manager._compare_values(10, "<=", 10) is True
        assert alert_manager._compare_values(10, "<=", 5) is False
        
        assert alert_manager._compare_values(10, "==", 10) is True
        assert alert_manager._compare_values(10, "==", 5) is False
        
        assert alert_manager._compare_values(10, "!=", 5) is True
        assert alert_manager._compare_values(10, "!=", 10) is False
        
        # Invalid operator
        assert alert_manager._compare_values(10, "invalid", 5) is False
    
    @pytest.mark.asyncio
    @patch('src.dataqa.orchestration.monitoring.alerts.metrics_collector')
    async def test_evaluate_rule_new_alert(self, mock_metrics, alert_manager):
        """Test evaluating rule that creates new alert."""
        # Mock metric value that triggers alert
        mock_metrics.get_metric_value.return_value = 0.15
        
        condition = AlertCondition(
            metric_name="error_rate",
            operator=">",
            threshold=0.1,
            duration_seconds=0.01  # Very short for testing
        )
        
        rule = AlertRule(
            name="high_error_rate",
            description="High error rate detected",
            severity=AlertSeverity.ERROR,
            conditions=[condition]
        )
        
        alert_manager.register_alert_rule(rule)
        
        # Wait for duration requirement
        await asyncio.sleep(0.02)
        
        alert = await alert_manager._evaluate_rule(rule, datetime.now(timezone.utc))
        
        assert alert is not None
        assert alert.rule_name == "high_error_rate"
        assert alert.severity == AlertSeverity.ERROR
        assert alert.state == AlertState.FIRING
        assert "error_rate > 0.1" in alert.message
    
    @pytest.mark.asyncio
    @patch('src.dataqa.orchestration.monitoring.alerts.metrics_collector')
    async def test_evaluate_rule_resolve_alert(self, mock_metrics, alert_manager):
        """Test evaluating rule that resolves existing alert."""
        condition = AlertCondition(
            metric_name="error_rate",
            operator=">",
            threshold=0.1,
            duration_seconds=0.01
        )
        
        rule = AlertRule(
            name="high_error_rate",
            description="High error rate detected",
            severity=AlertSeverity.ERROR,
            conditions=[condition]
        )
        
        alert_manager.register_alert_rule(rule)
        
        # First, trigger the alert
        mock_metrics.get_metric_value.return_value = 0.15
        await asyncio.sleep(0.02)
        alert = await alert_manager._evaluate_rule(rule, datetime.now(timezone.utc))
        assert alert is not None
        assert alert.state == AlertState.FIRING
        
        # Now resolve it
        mock_metrics.get_metric_value.return_value = 0.05  # Below threshold
        resolved_alert = await alert_manager._evaluate_rule(rule, datetime.now(timezone.utc))
        
        assert resolved_alert is not None
        assert resolved_alert.state == AlertState.RESOLVED
        assert resolved_alert.ends_at is not None
    
    @pytest.mark.asyncio
    async def test_send_notifications(self, alert_manager):
        """Test sending notifications for alerts."""
        # Register notification channel and handler
        channel = NotificationChannel(
            name="test_email",
            type="email",
            config={"to": "admin@test.com"},
            severity_filter=[AlertSeverity.ERROR]
        )
        alert_manager.register_notification_channel(channel)
        
        notification_sent = False
        
        async def test_handler(alert, channel):
            nonlocal notification_sent
            notification_sent = True
            assert alert.severity == AlertSeverity.ERROR
            assert channel.name == "test_email"
        
        alert_manager.register_notification_handler("email", test_handler)
        
        # Register rule with notification
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            severity=AlertSeverity.ERROR,
            conditions=[],
            notification_channels=["test_email"]
        )
        alert_manager.register_alert_rule(rule)
        
        # Create alert
        alert = Alert(
            id="test_alert",
            rule_name="test_rule",
            severity=AlertSeverity.ERROR,
            state=AlertState.FIRING,
            message="Test alert",
            description="Test description",
            starts_at=datetime.now(timezone.utc)
        )
        
        await alert_manager.send_notifications(alert)
        
        assert notification_sent is True
        assert alert.notification_sent is True
    
    @pytest.mark.asyncio
    async def test_send_notifications_severity_filter(self, alert_manager):
        """Test notification severity filtering."""
        # Register channel that only accepts CRITICAL alerts
        channel = NotificationChannel(
            name="critical_only",
            type="email",
            config={},
            severity_filter=[AlertSeverity.CRITICAL]
        )
        alert_manager.register_notification_channel(channel)
        
        notification_sent = False
        
        async def test_handler(alert, channel):
            nonlocal notification_sent
            notification_sent = True
        
        alert_manager.register_notification_handler("email", test_handler)
        
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            severity=AlertSeverity.ERROR,  # ERROR severity
            conditions=[],
            notification_channels=["critical_only"]
        )
        alert_manager.register_alert_rule(rule)
        
        # Create ERROR alert (should be filtered out)
        alert = Alert(
            id="test_alert",
            rule_name="test_rule",
            severity=AlertSeverity.ERROR,
            state=AlertState.FIRING,
            message="Test alert",
            description="Test description",
            starts_at=datetime.now(timezone.utc)
        )
        
        await alert_manager.send_notifications(alert)
        
        assert notification_sent is False  # Should be filtered out
        assert alert.notification_sent is True  # Still marked as sent
    
    def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts."""
        # Create some alerts
        alert1 = Alert(
            id="alert1",
            rule_name="rule1",
            severity=AlertSeverity.ERROR,
            state=AlertState.FIRING,
            message="Alert 1",
            description="Description 1",
            starts_at=datetime.now(timezone.utc)
        )
        
        alert2 = Alert(
            id="alert2",
            rule_name="rule2",
            severity=AlertSeverity.CRITICAL,
            state=AlertState.FIRING,
            message="Alert 2",
            description="Description 2",
            starts_at=datetime.now(timezone.utc) - timedelta(minutes=5)
        )
        
        alert3 = Alert(
            id="alert3",
            rule_name="rule3",
            severity=AlertSeverity.WARNING,
            state=AlertState.RESOLVED,
            message="Alert 3",
            description="Description 3",
            starts_at=datetime.now(timezone.utc) - timedelta(minutes=10)
        )
        
        alert_manager._active_alerts = {
            "alert1": alert1,
            "alert2": alert2,
            "alert3": alert3
        }
        
        # Get all active alerts (should exclude resolved)
        active = alert_manager.get_active_alerts()
        assert len(active) == 2
        assert alert2 in active  # Should be first (older)
        assert alert1 in active
        
        # Get only CRITICAL alerts
        critical = alert_manager.get_active_alerts(AlertSeverity.CRITICAL)
        assert len(critical) == 1
        assert critical[0] == alert2
    
    def test_get_alert_history(self, alert_manager):
        """Test getting alert history."""
        now = datetime.now(timezone.utc)
        
        # Recent alert
        recent_alert = Alert(
            id="recent",
            rule_name="rule1",
            severity=AlertSeverity.ERROR,
            state=AlertState.RESOLVED,
            message="Recent alert",
            description="Description",
            starts_at=now - timedelta(hours=1)
        )
        
        # Old alert
        old_alert = Alert(
            id="old",
            rule_name="rule2",
            severity=AlertSeverity.ERROR,
            state=AlertState.RESOLVED,
            message="Old alert",
            description="Description",
            starts_at=now - timedelta(hours=25)  # Older than 24 hours
        )
        
        alert_manager._active_alerts = {
            "recent": recent_alert,
            "old": old_alert
        }
        
        # Get 24-hour history
        history = alert_manager.get_alert_history(hours=24)
        assert len(history) == 1
        assert history[0] == recent_alert
        
        # Get 48-hour history
        history = alert_manager.get_alert_history(hours=48)
        assert len(history) == 2
    
    def test_suppress_alert(self, alert_manager):
        """Test suppressing alerts."""
        alert = Alert(
            id="test_alert",
            rule_name="rule1",
            severity=AlertSeverity.ERROR,
            state=AlertState.FIRING,
            message="Test alert",
            description="Description",
            starts_at=datetime.now(timezone.utc)
        )
        
        alert_manager._active_alerts["test_alert"] = alert
        
        # Suppress alert
        result = alert_manager.suppress_alert("test_alert", 30, "Testing suppression")
        
        assert result is True
        assert alert.state == AlertState.SUPPRESSED
        assert alert.suppressed_until is not None
        assert alert.annotations["suppression_reason"] == "Testing suppression"
        
        # Try to suppress non-existent alert
        result = alert_manager.suppress_alert("non_existent", 30)
        assert result is False
    
    def test_resolve_alert(self, alert_manager):
        """Test manually resolving alerts."""
        alert = Alert(
            id="test_alert",
            rule_name="rule1",
            severity=AlertSeverity.ERROR,
            state=AlertState.FIRING,
            message="Test alert",
            description="Description",
            starts_at=datetime.now(timezone.utc)
        )
        
        alert_manager._active_alerts["test_alert"] = alert
        
        # Resolve alert
        result = alert_manager.resolve_alert("test_alert", "Manual resolution")
        
        assert result is True
        assert alert.state == AlertState.RESOLVED
        assert alert.ends_at is not None
        assert alert.annotations["resolution_reason"] == "Manual resolution"
        
        # Try to resolve non-existent alert
        result = alert_manager.resolve_alert("non_existent")
        assert result is False
    
    def test_cleanup_old_alerts(self, alert_manager):
        """Test cleaning up old resolved alerts."""
        now = datetime.now(timezone.utc)
        
        # Recent resolved alert (should be kept)
        recent_alert = Alert(
            id="recent",
            rule_name="rule1",
            severity=AlertSeverity.ERROR,
            state=AlertState.RESOLVED,
            message="Recent alert",
            description="Description",
            starts_at=now - timedelta(hours=1),
            ends_at=now - timedelta(minutes=30)
        )
        
        # Old resolved alert (should be removed)
        old_alert = Alert(
            id="old",
            rule_name="rule2",
            severity=AlertSeverity.ERROR,
            state=AlertState.RESOLVED,
            message="Old alert",
            description="Description",
            starts_at=now - timedelta(days=8),
            ends_at=now - timedelta(days=8)
        )
        
        # Active alert (should be kept)
        active_alert = Alert(
            id="active",
            rule_name="rule3",
            severity=AlertSeverity.ERROR,
            state=AlertState.FIRING,
            message="Active alert",
            description="Description",
            starts_at=now - timedelta(hours=1)
        )
        
        alert_manager._active_alerts = {
            "recent": recent_alert,
            "old": old_alert,
            "active": active_alert
        }
        
        alert_manager._cleanup_old_alerts()
        
        # Should keep recent and active, remove old
        assert "recent" in alert_manager._active_alerts
        assert "active" in alert_manager._active_alerts
        assert "old" not in alert_manager._active_alerts


class TestAlertModels:
    """Test alert model classes."""
    
    def test_alert_condition_creation(self):
        """Test creating AlertCondition instance."""
        condition = AlertCondition(
            metric_name="cpu_usage",
            operator=">",
            threshold=80.0,
            duration_seconds=300.0,
            labels={"instance": "web-1"}
        )
        
        assert condition.metric_name == "cpu_usage"
        assert condition.operator == ">"
        assert condition.threshold == 80.0
        assert condition.duration_seconds == 300.0
        assert condition.labels == {"instance": "web-1"}
    
    def test_alert_rule_creation(self):
        """Test creating AlertRule instance."""
        condition = AlertCondition(
            metric_name="error_rate",
            operator=">",
            threshold=0.1,
            duration_seconds=60.0
        )
        
        rule = AlertRule(
            name="high_error_rate",
            description="High error rate detected",
            severity=AlertSeverity.ERROR,
            conditions=[condition],
            cooldown_seconds=300.0,
            notification_channels=["email", "slack"]
        )
        
        assert rule.name == "high_error_rate"
        assert rule.severity == AlertSeverity.ERROR
        assert len(rule.conditions) == 1
        assert rule.cooldown_seconds == 300.0
        assert rule.notification_channels == ["email", "slack"]
        assert rule.enabled is True  # Default
    
    def test_alert_creation(self):
        """Test creating Alert instance."""
        alert = Alert(
            id="alert_123",
            rule_name="test_rule",
            severity=AlertSeverity.CRITICAL,
            state=AlertState.FIRING,
            message="Critical alert triggered",
            description="System is experiencing issues",
            starts_at=datetime.now(timezone.utc),
            labels={"service": "api"},
            annotations={"runbook": "https://wiki.example.com/runbook"}
        )
        
        assert alert.id == "alert_123"
        assert alert.rule_name == "test_rule"
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.state == AlertState.FIRING
        assert alert.message == "Critical alert triggered"
        assert alert.labels == {"service": "api"}
        assert alert.annotations == {"runbook": "https://wiki.example.com/runbook"}
        assert alert.notification_sent is False  # Default
    
    def test_notification_channel_creation(self):
        """Test creating NotificationChannel instance."""
        channel = NotificationChannel(
            name="production_alerts",
            type="slack",
            config={
                "webhook_url": "https://hooks.slack.com/...",
                "channel": "#alerts"
            },
            severity_filter=[AlertSeverity.ERROR, AlertSeverity.CRITICAL]
        )
        
        assert channel.name == "production_alerts"
        assert channel.type == "slack"
        assert channel.config["channel"] == "#alerts"
        assert AlertSeverity.ERROR in channel.severity_filter
        assert AlertSeverity.CRITICAL in channel.severity_filter
        assert channel.enabled is True  # Default


class TestNotificationHandlers:
    """Test default notification handlers."""
    
    @pytest.mark.asyncio
    async def test_email_notification_handler(self, capsys):
        """Test email notification handler."""
        alert = Alert(
            id="test",
            rule_name="test_rule",
            severity=AlertSeverity.ERROR,
            state=AlertState.FIRING,
            message="Test alert",
            description="Test description",
            starts_at=datetime.now(timezone.utc)
        )
        
        channel = NotificationChannel(
            name="email",
            type="email",
            config={"to": "admin@test.com"}
        )
        
        await email_notification_handler(alert, channel)
        
        # Check that message was printed (since it's a mock implementation)
        captured = capsys.readouterr()
        assert "EMAIL: ERROR - Test alert" in captured.out
    
    @pytest.mark.asyncio
    async def test_webhook_notification_handler(self, capsys):
        """Test webhook notification handler."""
        alert = Alert(
            id="test",
            rule_name="test_rule",
            severity=AlertSeverity.CRITICAL,
            state=AlertState.FIRING,
            message="Critical alert",
            description="Test description",
            starts_at=datetime.now(timezone.utc)
        )
        
        channel = NotificationChannel(
            name="webhook",
            type="webhook",
            config={"url": "https://example.com/webhook"}
        )
        
        await webhook_notification_handler(alert, channel)
        
        # Check that message was printed (since it's a mock implementation)
        captured = capsys.readouterr()
        assert "WEBHOOK: CRITICAL - Critical alert" in captured.out