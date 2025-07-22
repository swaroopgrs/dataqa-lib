"""Tests for Message data model."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.dataqa.models.message import Message


class TestMessage:
    """Test cases for Message model."""
    
    def test_message_creation_with_required_fields(self):
        """Test creating a message with only required fields."""
        message = Message(role="user", content="Hello, world!")
        
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert isinstance(message.timestamp, datetime)
        assert message.metadata is None
    
    def test_message_creation_with_all_fields(self):
        """Test creating a message with all fields."""
        timestamp = datetime.now()
        metadata = {"source": "test", "priority": "high"}
        
        message = Message(
            role="assistant",
            content="How can I help you?",
            timestamp=timestamp,
            metadata=metadata
        )
        
        assert message.role == "assistant"
        assert message.content == "How can I help you?"
        assert message.timestamp == timestamp
        assert message.metadata == metadata
    
    def test_message_role_validation(self):
        """Test that role field only accepts valid values."""
        # Valid roles should work
        for role in ["user", "assistant", "system"]:
            message = Message(role=role, content="test")
            assert message.role == role
        
        # Invalid role should raise ValidationError
        with pytest.raises(ValidationError):
            Message(role="invalid_role", content="test")
    
    def test_message_content_required(self):
        """Test that content field is required."""
        with pytest.raises(ValidationError):
            Message(role="user")
    
    def test_message_timestamp_auto_generation(self):
        """Test that timestamp is automatically generated if not provided."""
        before = datetime.now()
        message = Message(role="user", content="test")
        after = datetime.now()
        
        assert before <= message.timestamp <= after
    
    def test_message_json_serialization(self):
        """Test that message can be serialized to JSON."""
        message = Message(
            role="user",
            content="test message",
            metadata={"key": "value"}
        )
        
        json_data = message.model_dump()
        
        assert json_data["role"] == "user"
        assert json_data["content"] == "test message"
        assert "timestamp" in json_data
        assert json_data["metadata"] == {"key": "value"}
    
    def test_message_from_dict(self):
        """Test creating message from dictionary."""
        data = {
            "role": "assistant",
            "content": "response",
            "timestamp": "2024-01-01T12:00:00",
            "metadata": {"test": True}
        }
        
        message = Message(**data)
        
        assert message.role == "assistant"
        assert message.content == "response"
        assert message.metadata == {"test": True}