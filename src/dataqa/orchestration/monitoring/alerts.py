"""
Alert management system for monitoring and observability.

Provides configurable alerting rules, notification channels,
and alert lifecycle management for multi-agent orchestration.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable, Set
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field

from .metrics import metrics_collector
from .health import health_checker, HealthStatus


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(str, Enum):
    """Alert states."""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertCondition(BaseModel):
    """Condition for triggering an alert."""
    
    metric_name: str = Field(description="Metric to monitor")
    operator: str = Field(description="Comparison operator (>, <, >=, <=, ==, !=)")
    threshold: float = Field(description="Threshold value")
    duration_seconds: float = Field(default=60.0, description="Duration condition must be true")
    labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels to match")


class AlertRule(BaseModel):
    """Alert rule definition."""
    
    name: str = Field(description="Alert rule name")
    description: str = Field(description="Alert description")
    severity: AlertSeverity = Field(description="Alert severity")
    conditions: List[AlertCondition] = Field(description="Alert conditions")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    cooldown_seconds: float = Field(default=300.0, description="Cooldown period after firing")
    notification_channels: List[str] = Field(default_factory=list, description="Notification channels")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Alert(BaseModel):
    """Active alert instance."""
    
    id: str = Field(description="Unique alert ID")
    rule_name: str = Field(description="Alert rule name")
    severity: AlertSeverity = Field(description="Alert severity")
    state: AlertState = Field(description="Current alert state")
    message: str = Field(description="Alert message")
    description: str = Field(description="Alert description")
    labels: Dict[str, str] = Field(default_factory=dict, description="Alert labels")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Alert annotations")
    starts_at: datetime = Field(description="When alert started")
    ends_at: Optional[datetime] = Field(None, description="When alert ended")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notification_sent: bool = Field(default=False, description="Whether notification was sent")
    suppressed_until: Optional[datetime] = Field(None, description="Suppression end time")


class NotificationChannel(BaseModel):
    """Notification channel configuration."""
    
    name: str = Field(description="Channel name")
    type: str = Field(description="Channel type (email, slack, webhook, etc.)")
    config: Dict[str, Any] = Field(description="Channel configuration")
    enabled: bool = Field(default=True, description="Whether channel is enabled")
    severity_filter: List[AlertSeverity] = Field(
        default_factory=lambda: list(AlertSeverity), 
        description="Severities to notify for"
    )


@dataclass
class AlertManager:
    """
    Comprehensive alert management system.
    
    Provides rule-based alerting, notification management,
    and alert lifecycle tracking for monitoring systems.
    """
    
    _alert_rules: Dict[str, AlertRule] = field(default_factory=dict)
    _active_alerts: Dict[str, Alert] = field(default_factory=dict)
    _notification_channels: Dict[str, NotificationChannel] = field(default_factory=dict)
    _notification_handlers: Dict[str, Callable[[Alert, NotificationChannel], Awaitable[None]]] = field(default_factory=dict)
    _condition_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    _monitoring_task: Optional[asyncio.Task] = None
    _monitoring_enabled: bool = True
    _evaluation_interval: float = 30.0
    
    def register_alert_rule(self, rule: AlertRule) -> None:
        """Register an alert rule."""
        self._alert_rules[rule.name] = rule
    
    def register_notification_channel(self, channel: NotificationChannel) -> None:
        """Register a notification channel."""
        self._notification_channels[channel.name] = channel
    
    def register_notification_handler(self, channel_type: str, 
                                    handler: Callable[[Alert, NotificationChannel], Awaitable[None]]) -> None:
        """Register a notification handler for a channel type."""
        self._notification_handlers[channel_type] = handler
    
    async def evaluate_rules(self) -> List[Alert]:
        """Evaluate all alert rules and return new/updated alerts."""
        new_alerts = []
        current_time = datetime.now(timezone.utc)
        
        for rule_name, rule in self._alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                alert = await self._evaluate_rule(rule, current_time)
                if alert:
                    new_alerts.append(alert)
            except Exception as e:
                print(f"Error evaluating rule {rule_name}: {e}")
        
        return new_alerts
    
    async def _evaluate_rule(self, rule: AlertRule, current_time: datetime) -> Optional[Alert]:
        """Evaluate a single alert rule."""
        # Check if all conditions are met
        conditions_met = True
        condition_details = []
        
        for condition in rule.conditions:
            is_met, details = await self._evaluate_condition(condition, current_time)
            condition_details.append(details)
            
            if not is_met:
                conditions_met = False
                break
        
        # Generate alert ID
        alert_id = f"{rule.name}_{hash(str(condition_details))}"
        
        # Check if alert already exists
        existing_alert = self._active_alerts.get(alert_id)
        
        if conditions_met:
            if existing_alert:
                # Update existing alert
                if existing_alert.state == AlertState.RESOLVED:
                    # Re-fire resolved alert
                    existing_alert.state = AlertState.FIRING
                    existing_alert.last_updated = current_time
                    existing_alert.ends_at = None
                    existing_alert.notification_sent = False
                    return existing_alert
                else:
                    # Update timestamp
                    existing_alert.last_updated = current_time
                    return None
            else:
                # Create new alert
                alert = Alert(
                    id=alert_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    state=AlertState.FIRING,
                    message=self._generate_alert_message(rule, condition_details),
                    description=rule.description,
                    starts_at=current_time,
                    labels=self._generate_alert_labels(rule, condition_details),
                    annotations=self._generate_alert_annotations(rule, condition_details)
                )
                
                self._active_alerts[alert_id] = alert
                return alert
        else:
            if existing_alert and existing_alert.state == AlertState.FIRING:
                # Resolve alert
                existing_alert.state = AlertState.RESOLVED
                existing_alert.ends_at = current_time
                existing_alert.last_updated = current_time
                return existing_alert
        
        return None
    
    async def _evaluate_condition(self, condition: AlertCondition, 
                                current_time: datetime) -> tuple[bool, Dict[str, Any]]:
        """Evaluate a single alert condition."""
        # Get metric value
        metric_value = metrics_collector.get_metric_value(
            condition.metric_name, 
            condition.labels
        )
        
        if metric_value is None:
            return False, {
                "metric_name": condition.metric_name,
                "value": None,
                "threshold": condition.threshold,
                "operator": condition.operator,
                "result": False,
                "reason": "Metric not found"
            }
        
        # Evaluate condition
        result = self._compare_values(metric_value, condition.operator, condition.threshold)
        
        # Track condition history for duration checking
        condition_key = f"{condition.metric_name}_{condition.operator}_{condition.threshold}"
        
        if condition_key not in self._condition_history:
            self._condition_history[condition_key] = []
        
        self._condition_history[condition_key].append({
            "timestamp": current_time.timestamp(),
            "value": metric_value,
            "result": result
        })
        
        # Keep only recent history
        cutoff_time = current_time.timestamp() - condition.duration_seconds * 2
        self._condition_history[condition_key] = [
            h for h in self._condition_history[condition_key]
            if h["timestamp"] >= cutoff_time
        ]
        
        # Check if condition has been true for required duration
        duration_met = self._check_condition_duration(
            condition_key, condition.duration_seconds, current_time
        )
        
        return result and duration_met, {
            "metric_name": condition.metric_name,
            "value": metric_value,
            "threshold": condition.threshold,
            "operator": condition.operator,
            "result": result,
            "duration_met": duration_met,
            "duration_required": condition.duration_seconds
        }
    
    def _compare_values(self, value: float, operator: str, threshold: float) -> bool:
        """Compare metric value against threshold."""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return abs(value - threshold) < 1e-9
        elif operator == "!=":
            return abs(value - threshold) >= 1e-9
        else:
            return False
    
    def _check_condition_duration(self, condition_key: str, duration_seconds: float,
                                current_time: datetime) -> bool:
        """Check if condition has been true for required duration."""
        history = self._condition_history.get(condition_key, [])
        
        if not history:
            return False
        
        # Check if all recent results within duration window are true
        cutoff_time = current_time.timestamp() - duration_seconds
        recent_results = [
            h["result"] for h in history
            if h["timestamp"] >= cutoff_time
        ]
        
        return len(recent_results) > 0 and all(recent_results)
    
    def _generate_alert_message(self, rule: AlertRule, condition_details: List[Dict[str, Any]]) -> str:
        """Generate alert message from rule and conditions."""
        if len(condition_details) == 1:
            detail = condition_details[0]
            return f"{rule.name}: {detail['metric_name']} {detail['operator']} {detail['threshold']} (current: {detail['value']})"
        else:
            return f"{rule.name}: Multiple conditions triggered"
    
    def _generate_alert_labels(self, rule: AlertRule, condition_details: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate alert labels."""
        labels = {
            "alertname": rule.name,
            "severity": rule.severity.value
        }
        
        # Add metric-specific labels
        for detail in condition_details:
            labels[f"metric_{detail['metric_name']}"] = str(detail['value'])
        
        return labels
    
    def _generate_alert_annotations(self, rule: AlertRule, condition_details: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate alert annotations."""
        annotations = {
            "description": rule.description,
            "summary": self._generate_alert_message(rule, condition_details)
        }
        
        # Add condition details
        for i, detail in enumerate(condition_details):
            annotations[f"condition_{i}"] = f"{detail['metric_name']} {detail['operator']} {detail['threshold']}"
        
        return annotations
    
    async def send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        if alert.notification_sent:
            return
        
        rule = self._alert_rules.get(alert.rule_name)
        if not rule:
            return
        
        # Send to configured channels
        for channel_name in rule.notification_channels:
            channel = self._notification_channels.get(channel_name)
            if not channel or not channel.enabled:
                continue
            
            # Check severity filter
            if alert.severity not in channel.severity_filter:
                continue
            
            # Get handler for channel type
            handler = self._notification_handlers.get(channel.type)
            if not handler:
                continue
            
            try:
                await handler(alert, channel)
            except Exception as e:
                print(f"Error sending notification to {channel_name}: {e}")
        
        alert.notification_sent = True
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = [
            alert for alert in self._active_alerts.values()
            if alert.state in [AlertState.FIRING, AlertState.PENDING]
        ]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda a: a.starts_at, reverse=True)
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [
            alert for alert in self._active_alerts.values()
            if alert.starts_at >= cutoff_time
        ]
    
    def suppress_alert(self, alert_id: str, duration_minutes: int, reason: str = "") -> bool:
        """Suppress an alert for a specified duration."""
        alert = self._active_alerts.get(alert_id)
        if not alert:
            return False
        
        alert.state = AlertState.SUPPRESSED
        alert.suppressed_until = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        alert.annotations["suppression_reason"] = reason
        
        return True
    
    def resolve_alert(self, alert_id: str, reason: str = "") -> bool:
        """Manually resolve an alert."""
        alert = self._active_alerts.get(alert_id)
        if not alert:
            return False
        
        alert.state = AlertState.RESOLVED
        alert.ends_at = datetime.now(timezone.utc)
        alert.annotations["resolution_reason"] = reason
        
        return True
    
    def start_monitoring(self) -> None:
        """Start continuous alert monitoring."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_enabled = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    def stop_monitoring(self) -> None:
        """Stop continuous alert monitoring."""
        self._monitoring_enabled = False
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
    
    async def _monitoring_loop(self) -> None:
        """Continuous alert monitoring loop."""
        while self._monitoring_enabled:
            try:
                # Evaluate rules
                new_alerts = await self.evaluate_rules()
                
                # Send notifications for new/updated alerts
                for alert in new_alerts:
                    if alert.state == AlertState.FIRING:
                        await self.send_notifications(alert)
                
                # Clean up old resolved alerts
                self._cleanup_old_alerts()
                
                # Wait for next evaluation
                await asyncio.sleep(self._evaluation_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(5.0)
    
    def _cleanup_old_alerts(self) -> None:
        """Clean up old resolved alerts."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)
        
        alerts_to_remove = [
            alert_id for alert_id, alert in self._active_alerts.items()
            if alert.state == AlertState.RESOLVED and 
               alert.ends_at and alert.ends_at < cutoff_time
        ]
        
        for alert_id in alerts_to_remove:
            del self._active_alerts[alert_id]


# Default notification handlers
async def email_notification_handler(alert: Alert, channel: NotificationChannel) -> None:
    """Default email notification handler."""
    # This would integrate with an email service
    print(f"EMAIL: {alert.severity.value.upper()} - {alert.message}")


async def webhook_notification_handler(alert: Alert, channel: NotificationChannel) -> None:
    """Default webhook notification handler."""
    # This would send HTTP POST to webhook URL
    print(f"WEBHOOK: {alert.severity.value.upper()} - {alert.message}")


# Global alert manager instance
alert_manager = AlertManager()

# Register default notification handlers
alert_manager.register_notification_handler("email", email_notification_handler)
alert_manager.register_notification_handler("webhook", webhook_notification_handler)