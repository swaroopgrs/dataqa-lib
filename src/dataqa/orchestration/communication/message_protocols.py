"""Message protocols for inter-agent communication."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of messages in the communication system."""
    
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    COORDINATION_REQUEST = "coordination_request"
    APPROVAL_REQUEST = "approval_request"
    ESCALATION = "escalation"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"


class MessagePriority(str, Enum):
    """Message priority levels."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class Message(BaseModel):
    """Base message class for all inter-agent communication."""
    
    message_id: UUID = Field(default_factory=uuid4)
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str] = None  # None for broadcast messages
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[UUID] = None  # For request-response correlation
    reply_to: Optional[str] = None  # For response routing
    ttl_seconds: Optional[int] = None  # Time to live
    headers: Dict[str, str] = Field(default_factory=dict)
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }


class AgentMessage(Message):
    """Message specifically for agent-to-agent communication."""
    
    agent_type: str
    agent_capabilities: Optional[Dict[str, Any]] = None
    execution_context: Optional[Dict[str, Any]] = None


class TaskMessage(Message):
    """Message for task-related communication."""
    
    task_id: UUID
    task_type: str
    task_data: Dict[str, Any] = Field(default_factory=dict)
    requirements: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None


class StatusMessage(Message):
    """Message for status updates and progress reporting."""
    
    status: str
    progress_percentage: Optional[float] = None
    details: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class ErrorMessage(Message):
    """Message for error reporting and handling."""
    
    error_type: str
    error_code: Optional[str] = None
    error_message: str
    stack_trace: Optional[str] = None
    recovery_suggestions: Optional[list[str]] = None


class CoordinationMessage(Message):
    """Message for agent coordination and synchronization."""
    
    coordination_type: str
    participants: list[str]
    coordination_data: Dict[str, Any] = Field(default_factory=dict)


class ApprovalMessage(Message):
    """Message for human-in-the-loop approval workflows."""
    
    approval_type: str
    operation_description: str
    risk_level: str
    approval_data: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: Optional[int] = None


class MessageProtocol:
    """Protocol handler for message serialization and validation."""
    
    @staticmethod
    def serialize(message: Message) -> bytes:
        """Serialize message to bytes for transmission."""
        return message.model_dump_json().encode('utf-8')
    
    @staticmethod
    def deserialize(data: bytes, message_type: type[Message] = Message) -> Message:
        """Deserialize bytes to message object."""
        json_str = data.decode('utf-8')
        return message_type.model_validate_json(json_str)
    
    @staticmethod
    def validate_message(message: Message) -> bool:
        """Validate message structure and content."""
        try:
            # Check required fields
            if not message.sender_id:
                return False
            
            # Check TTL if specified
            if message.ttl_seconds is not None:
                age = (datetime.utcnow() - message.timestamp).total_seconds()
                if age > message.ttl_seconds:
                    return False
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def create_reply(original_message: Message, reply_payload: Dict[str, Any]) -> Message:
        """Create a reply message to an original message."""
        return Message(
            message_type=MessageType.STATUS_UPDATE,
            sender_id="system",  # Will be overridden by actual sender
            recipient_id=original_message.sender_id,
            correlation_id=original_message.message_id,
            payload=reply_payload
        )