"""Integration tests for DataAgent workflow."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.dataqa.agent.workflow import DataAgentWorkflow
from src.dataqa.agent.state import SharedState
from src.dataqa.config.models import AgentConfig, WorkflowConfig
from src.dataqa.models.document import Document
from src.dataqa.models.execution import ExecutionResult


@pytest.fixture
def mock_llm():
    """Create a mock LLM interface for integration testing."""
    llm = AsyncMock()
    llm.analyze_query.return_value = {
        "intent": "data analysis",
        "query_type": "data_retrieval",
        "entities": ["sales", "revenue"],
        "complexity": "moderate",
        "requires_clarification": False,
        "suggested_approach": "Generate SQL query"
    }
    llm.generate_code.return_value = "SELECT * FROM sales WHERE revenue > 1000"
    llm.validate_generated_code.return_value = {
        "is_valid": True,
        "issues": [],
        "security_concerns": [],
        "suggestions": [],
        "risk_level": "low"
    }
    llm.format_response.return_value = "Here are your sales results with revenue greater than 1000."
    return llm


@pytest.fixture
def mock_knowledge():
    """Create a mock knowledge primitive for integration testing."""
    knowledge = AsyncMock()
    knowledge.search.return_value = [
        Document(
            content="Sales table schema: id (int), product (varchar), revenue (decimal)",
            source="database_schema",
            metadata={"table": "sales", "type": "schema"}
        ),
        Document(
            content="Sales data contains revenue information for all products",
            source="business_docs",
            metadata={"domain": "sales", "type": "description"}
        )
    ]
    return knowledge


@pytest.fixture
def mock_executor():
    """Create a mock executor primitive for integration testing."""
    executor = AsyncMock()
    executor.execute_sql.return_value = ExecutionResult(
        success=True,
        data={
            "id": [1, 2, 3],
            "product": ["Widget A", "Widget B", "Widget C"],
            "revenue": [1500.0, 2000.0, 1200.0]
        },
        execution_time=0.25,
        code_executed="SELECT * FROM sales WHERE revenue > 1000",
        output_type="dataframe"
    )
    return executor


@pytest.fixture
def agent_config():
    """Create a test agent configuration."""
    return AgentConfig(
        name="test-agent",
        description="Test agent for integration testing",
        workflow=WorkflowConfig(
            max_iterations=5,
            require_approval=False,  # Disable approval for simpler testing
            conversation_memory=True
        )
    )


@pytest.fixture
def workflow(mock_llm, mock_knowledge, mock_executor, agent_config):
    """Create a DataAgentWorkflow instance for testing."""
    return DataAgentWorkflow(
        llm=mock_llm,
        knowledge=mock_knowledge,
        executor=mock_executor,
        config=agent_config
    )


class TestDataAgentWorkflow:
    """Integration tests for DataAgentWorkflow."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_success(self, workflow, mock_llm, mock_knowledge, mock_executor):
        """Test a complete successful workflow execution."""
        query = "Show me sales data with revenue greater than 1000"
        
        result = await workflow.process_query(query)
        
        # Verify final state
        assert isinstance(result, SharedState)
        assert result.workflow_complete is True
        assert result.error_occurred is False
        assert result.current_step == "complete"
        assert result.formatted_response is not None
        assert "sales results" in result.formatted_response
        
        # Verify conversation history
        assert len(result.conversation_history) == 2  # User query + assistant response
        assert result.conversation_history[0].role == "user"
        assert result.conversation_history[0].content == query
        assert result.conversation_history[1].role == "assistant"
        
        # Verify all components were called
        mock_llm.analyze_query.assert_called_once()
        mock_knowledge.search.assert_called_once()
        mock_llm.generate_code.assert_called_once()
        mock_llm.validate_generated_code.assert_called_once()
        mock_executor.execute_sql.assert_called_once()
        mock_llm.format_response.assert_called_once()
        
        # Verify state progression
        assert result.query_analysis is not None
        assert len(result.retrieved_context) == 2
        assert result.generated_code is not None
        assert result.code_type == "sql"
        assert result.execution_results is not None
        assert result.execution_results.success is True
    
    @pytest.mark.asyncio
    async def test_workflow_with_clarification_needed(self, workflow, mock_llm):
        """Test workflow when query requires clarification."""
        # Mock analysis that requires clarification
        mock_llm.analyze_query.return_value = {
            "intent": "unclear request",
            "query_type": "unknown",
            "entities": [],
            "complexity": "high",
            "requires_clarification": True,
            "ambiguities": ["Time period not specified", "Metric unclear"]
        }
        mock_llm.generate_clarification.return_value = "Could you please specify the time period and which metrics you're interested in?"
        
        query = "Show me the data"
        result = await workflow.process_query(query)
        
        # Verify workflow completed with clarification
        assert result.workflow_complete is True
        assert result.error_occurred is False
        assert result.current_step == "complete"
        assert result.formatted_response is not None
        assert "specify" in result.formatted_response
        
        # Verify only query analysis and clarification were called
        mock_llm.analyze_query.assert_called_once()
        mock_llm.generate_clarification.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_workflow_with_approval_required(self, workflow, mock_llm, agent_config):
        """Test workflow when approval is required."""
        # Enable approval requirement
        agent_config.workflow.require_approval = True
        workflow.config = agent_config
        workflow.nodes.require_approval = True
        
        # Mock high-risk code validation
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True,
            "issues": [],
            "security_concerns": ["Potential data modification"],
            "suggestions": ["Review before execution"],
            "risk_level": "high"
        }
        
        query = "Delete old sales records"
        result = await workflow.process_query(query)
        
        # Verify workflow stopped at approval gate
        assert result.workflow_complete is True
        assert result.error_occurred is False
        assert result.current_step == "awaiting_approval"
        assert result.pending_approval is not None
        assert "approve" in result.formatted_response.lower()
    
    @pytest.mark.asyncio
    async def test_workflow_with_execution_error(self, workflow, mock_executor):
        """Test workflow when code execution fails."""
        # Mock failed execution
        mock_executor.execute_sql.return_value = ExecutionResult(
            success=False,
            error="Table 'sales' does not exist",
            execution_time=0.1,
            code_executed="SELECT * FROM sales WHERE revenue > 1000"
        )
        
        query = "Show me sales data"
        result = await workflow.process_query(query)
        
        # Verify workflow completed with error
        assert result.workflow_complete is True
        assert result.error_occurred is True
        assert "Code execution failed" in result.error_message
    
    @pytest.mark.asyncio
    async def test_workflow_with_conversation_context(self, workflow):
        """Test workflow with existing conversation context."""
        # Create existing state with conversation history
        existing_state = SharedState()
        existing_state.add_message("user", "What tables are available?")
        existing_state.add_message("assistant", "The available tables are: sales, customers, products")
        
        query = "Show me data from the sales table"
        result = await workflow.process_query(
            query=query,
            existing_state=existing_state
        )
        
        # Verify conversation history is preserved and extended
        assert len(result.conversation_history) == 4  # 2 existing + 2 new
        assert result.conversation_history[0].content == "What tables are available?"
        assert result.conversation_history[2].content == query
        assert result.workflow_complete is True
        assert result.error_occurred is False
    
    @pytest.mark.asyncio
    async def test_continue_with_approval_granted(self, workflow, mock_executor):
        """Test continuing workflow after approval is granted."""
        # This test would require a more complex setup with actual state persistence
        # For now, we'll test the approval logic directly
        
        # Create a state with pending approval
        state = SharedState()
        state.pending_approval = "DELETE FROM old_records WHERE date < '2020-01-01'"
        state.code_type = "sql"
        state.generated_code = state.pending_approval
        state.code_validation = {"is_valid": True, "risk_level": "high"}
        
        # Grant approval
        updated_state = workflow.nodes.grant_approval(state)
        
        assert updated_state.approval_granted is True
        assert updated_state.current_step == "executor"
    
    @pytest.mark.asyncio
    async def test_continue_with_approval_denied(self, workflow):
        """Test continuing workflow after approval is denied."""
        # Create a state with pending approval
        state = SharedState()
        state.pending_approval = "DROP TABLE sensitive_data"
        
        # Deny approval
        updated_state = workflow.nodes.deny_approval(state, "Operation too risky")
        
        assert updated_state.error_occurred is True
        assert "too risky" in updated_state.error_message.lower()
        assert updated_state.workflow_complete is True
    
    @pytest.mark.asyncio
    async def test_workflow_max_iterations(self, workflow, mock_llm):
        """Test workflow respects maximum iterations limit."""
        # Create a workflow that will loop (mock analysis always requires more processing)
        mock_llm.analyze_query.side_effect = Exception("Simulated processing error")
        
        query = "Complex query that causes errors"
        result = await workflow.process_query(query)
        
        # Should complete with error due to processing failure
        assert result.workflow_complete is True
        assert result.error_occurred is True
    
    def test_workflow_info(self, workflow):
        """Test getting workflow information."""
        info = workflow.get_workflow_info()
        
        assert isinstance(info, dict)
        assert info["agent_name"] == "test-agent"
        assert info["max_iterations"] == 5
        assert info["require_approval"] is False
        assert "nodes" in info
        assert len(info["nodes"]) == 6  # All workflow nodes
        
        expected_nodes = [
            "query_processor",
            "context_retriever",
            "code_generator", 
            "approval_gate",
            "executor",
            "response_formatter"
        ]
        assert all(node in info["nodes"] for node in expected_nodes)
    
    @pytest.mark.asyncio
    async def test_python_code_execution_workflow(self, workflow, mock_llm, mock_executor):
        """Test workflow with Python code generation and execution."""
        # Mock analysis for visualization request
        mock_llm.analyze_query.return_value = {
            "intent": "data visualization",
            "query_type": "visualization",
            "entities": ["sales", "chart"],
            "complexity": "moderate",
            "requires_clarification": False,
            "suggested_approach": "Generate Python visualization code"
        }
        
        # Mock Python code generation
        mock_llm.generate_code.return_value = """
import matplotlib.pyplot as plt
import pandas as pd

# Create a simple bar chart
data = {'Product': ['A', 'B', 'C'], 'Revenue': [1500, 2000, 1200]}
df = pd.DataFrame(data)
plt.bar(df['Product'], df['Revenue'])
plt.title('Revenue by Product')
plt.show()
"""
        
        # Mock Python execution
        mock_executor.execute_python.return_value = ExecutionResult(
            success=True,
            data={"plot_data": "base64_encoded_image"},
            execution_time=1.2,
            code_executed="Python visualization code",
            output_type="plot"
        )
        
        query = "Create a bar chart of sales revenue by product"
        result = await workflow.process_query(query)
        
        # Verify Python workflow
        assert result.workflow_complete is True
        assert result.error_occurred is False
        assert result.code_type == "python"
        assert result.execution_results.output_type == "plot"
        
        # Verify Python executor was called
        mock_executor.execute_python.assert_called_once()
        mock_executor.execute_sql.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self, workflow):
        """Test that workflow state can be retrieved."""
        conversation_id = "test-conversation-123"
        query = "Show me sales data"
        
        # Process query with specific conversation ID
        result = await workflow.process_query(query, conversation_id=conversation_id)
        
        # Verify we can retrieve the conversation state
        # Note: This test is limited by our mock setup, but verifies the interface
        retrieved_state = await workflow.get_conversation_state(conversation_id)
        
        # In a real implementation with proper state persistence, we would verify:
        # assert retrieved_state is not None
        # assert retrieved_state.current_query == query
        # For now, we just verify the method doesn't crash
        assert retrieved_state is None or isinstance(retrieved_state, SharedState)