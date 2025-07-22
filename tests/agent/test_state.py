"""Tests for SharedState model."""

import pytest
from datetime import datetime

from src.dataqa.agent.state import SharedState
from src.dataqa.models.document import Document
from src.dataqa.models.execution import ExecutionResult
from src.dataqa.models.message import Message


class TestSharedState:
    """Test cases for SharedState model."""
    
    def test_state_initialization(self):
        """Test SharedState initialization with defaults."""
        state = SharedState()
        
        assert state.conversation_history == []
        assert state.current_query == ""
        assert state.query_analysis is None
        assert state.retrieved_context == []
        assert state.context_summary is None
        assert state.generated_code is None
        assert state.code_type is None
        assert state.code_validation is None
        assert state.execution_results is None
        assert state.pending_approval is None
        assert state.approval_granted is False
        assert state.formatted_response is None
        assert state.current_step == "query_processor"
        assert state.workflow_complete is False
        assert state.error_occurred is False
        assert state.error_message is None
        assert state.iteration_count == 0
        assert state.max_iterations == 10
        assert state.metadata == {}
    
    def test_state_initialization_with_values(self):
        """Test SharedState initialization with custom values."""
        state = SharedState(
            current_query="test query",
            max_iterations=5,
            metadata={"test": "value"}
        )
        
        assert state.current_query == "test query"
        assert state.max_iterations == 5
        assert state.metadata == {"test": "value"}
    
    def test_add_message(self):
        """Test adding messages to conversation history."""
        state = SharedState()
        
        state.add_message("user", "Hello")
        state.add_message("assistant", "Hi there", {"test": "metadata"})
        
        assert len(state.conversation_history) == 2
        
        user_msg = state.conversation_history[0]
        assert user_msg.role == "user"
        assert user_msg.content == "Hello"
        assert user_msg.metadata is None
        assert isinstance(user_msg.timestamp, datetime)
        
        assistant_msg = state.conversation_history[1]
        assert assistant_msg.role == "assistant"
        assert assistant_msg.content == "Hi there"
        assert assistant_msg.metadata == {"test": "metadata"}
    
    def test_get_recent_messages(self):
        """Test getting recent messages from conversation history."""
        state = SharedState()
        
        # Add more messages than the limit
        for i in range(10):
            state.add_message("user", f"Message {i}")
        
        recent = state.get_recent_messages(limit=3)
        assert len(recent) == 3
        assert recent[0].content == "Message 7"
        assert recent[1].content == "Message 8"
        assert recent[2].content == "Message 9"
        
        # Test with empty history
        empty_state = SharedState()
        assert empty_state.get_recent_messages() == []
    
    def test_reset_for_new_query(self):
        """Test resetting state for a new query."""
        state = SharedState()
        
        # Set up some state
        state.add_message("user", "Previous query")
        state.query_analysis = {"intent": "test"}
        state.retrieved_context = [Document(content="test", source="test")]
        state.context_summary = "test summary"
        state.generated_code = "SELECT * FROM test"
        state.code_type = "sql"
        state.code_validation = {"is_valid": True}
        state.execution_results = ExecutionResult(
            success=True,
            execution_time=1.0,
            code_executed="test"
        )
        state.pending_approval = "test code"
        state.approval_granted = True
        state.formatted_response = "test response"
        state.current_step = "complete"
        state.workflow_complete = True
        state.error_occurred = True
        state.error_message = "test error"
        state.iteration_count = 5
        
        # Reset for new query
        state.reset_for_new_query("New query")
        
        # Check that conversation history is preserved
        assert len(state.conversation_history) == 1
        assert state.conversation_history[0].content == "Previous query"
        
        # Check that query-specific state is reset
        assert state.current_query == "New query"
        assert state.query_analysis is None
        assert state.retrieved_context == []
        assert state.context_summary is None
        assert state.generated_code is None
        assert state.code_type is None
        assert state.code_validation is None
        assert state.execution_results is None
        assert state.pending_approval is None
        assert state.approval_granted is False
        assert state.formatted_response is None
        assert state.current_step == "query_processor"
        assert state.workflow_complete is False
        assert state.error_occurred is False
        assert state.error_message is None
        assert state.iteration_count == 0
    
    def test_set_error(self):
        """Test setting error state."""
        state = SharedState()
        
        state.set_error("Test error message")
        
        assert state.error_occurred is True
        assert state.error_message == "Test error message"
        assert state.workflow_complete is True
    
    def test_increment_iteration(self):
        """Test iteration count increment and max check."""
        state = SharedState(max_iterations=3)
        
        # First increment should succeed
        assert state.increment_iteration() is True
        assert state.iteration_count == 1
        assert state.error_occurred is False
        
        # Second increment should succeed
        assert state.increment_iteration() is True
        assert state.iteration_count == 2
        assert state.error_occurred is False
        
        # Third increment should fail (reaches max)
        assert state.increment_iteration() is False
        assert state.iteration_count == 3
        assert state.error_occurred is True
        assert "Maximum iterations" in state.error_message
        assert state.workflow_complete is True
    
    def test_is_ready_for_execution(self):
        """Test checking if state is ready for execution."""
        state = SharedState()
        
        # Initially not ready
        assert state.is_ready_for_execution() is False
        
        # Add generated code but no validation
        state.generated_code = "SELECT * FROM test"
        state.code_type = "sql"
        assert state.is_ready_for_execution() is False
        
        # Add invalid validation
        state.code_validation = {"is_valid": False}
        assert state.is_ready_for_execution() is False
        
        # Add valid validation
        state.code_validation = {"is_valid": True}
        assert state.is_ready_for_execution() is True
    
    def test_requires_approval(self):
        """Test checking if approval is required."""
        state = SharedState()
        
        # Initially no approval required
        assert state.requires_approval() is False
        
        # Set pending approval
        state.pending_approval = "test code"
        assert state.requires_approval() is True
        
        # Grant approval
        state.approval_granted = True
        assert state.requires_approval() is False
        
        # Remove pending approval
        state.pending_approval = None
        state.approval_granted = False
        assert state.requires_approval() is False
    
    def test_state_serialization(self):
        """Test that state can be serialized/deserialized."""
        state = SharedState(
            current_query="test query",
            max_iterations=5
        )
        
        state.add_message("user", "Hello")
        state.query_analysis = {"intent": "test"}
        state.retrieved_context = [Document(content="test", source="test")]
        
        # Test model dump
        data = state.model_dump()
        assert isinstance(data, dict)
        assert data["current_query"] == "test query"
        assert data["max_iterations"] == 5
        assert len(data["conversation_history"]) == 1
        assert len(data["retrieved_context"]) == 1
        
        # Test model reconstruction
        new_state = SharedState.model_validate(data)
        assert new_state.current_query == state.current_query
        assert new_state.max_iterations == state.max_iterations
        assert len(new_state.conversation_history) == 1
        assert len(new_state.retrieved_context) == 1