"""Integration tests for complete agent workflow."""

import pytest
from unittest.mock import AsyncMock

from src.dataqa.agent.workflow import DataAgentWorkflow
from src.dataqa.agent.state import SharedState
from src.dataqa.config.models import AgentConfig, WorkflowConfig
from src.dataqa.models.document import Document
from src.dataqa.models.execution import ExecutionResult


@pytest.fixture
def mock_llm():
    """Create a comprehensive mock LLM interface."""
    llm = AsyncMock()
    
    # Default successful query analysis
    llm.analyze_query.return_value = {
        "intent": "data analysis",
        "query_type": "data_retrieval",
        "entities": ["sales", "revenue"],
        "complexity": "moderate",
        "requires_clarification": False,
        "suggested_approach": "Generate SQL query"
    }
    
    # Default code generation
    llm.generate_code.return_value = "SELECT * FROM sales WHERE revenue > 1000"
    
    # Default code validation
    llm.validate_generated_code.return_value = {
        "is_valid": True,
        "issues": [],
        "security_concerns": [],
        "suggestions": [],
        "risk_level": "low"
    }
    
    # Default response formatting
    llm.format_response.return_value = "Here are your sales results with revenue greater than 1000."
    
    return llm


@pytest.fixture
def mock_knowledge():
    """Create a comprehensive mock knowledge primitive."""
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
    """Create a comprehensive mock executor primitive."""
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
        name="integration-test-agent",
        description="Integration test agent",
        workflow=WorkflowConfig(
            max_iterations=5,
            require_approval=False,
            conversation_memory=True
        )
    )


@pytest.fixture
def workflow(mock_llm, mock_knowledge, mock_executor, agent_config):
    """Create a DataAgentWorkflow instance for integration testing."""
    return DataAgentWorkflow(
        llm=mock_llm,
        knowledge=mock_knowledge,
        executor=mock_executor,
        config=agent_config
    )


class TestAgentIntegration:
    """Integration tests for complete agent workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_sql_workflow(self, workflow, mock_llm, mock_knowledge, mock_executor):
        """Test a complete SQL workflow from query to response."""
        query = "Show me sales data with revenue greater than 1000"
        
        result = await workflow.process_query(query)
        
        # Verify workflow completed successfully
        assert result.workflow_complete is True
        assert result.error_occurred is False
        assert result.current_step == "complete"
        
        # Verify all components were called in correct order
        mock_llm.analyze_query.assert_called_once()
        mock_knowledge.search.assert_called_once()
        mock_llm.generate_code.assert_called_once()
        mock_llm.validate_generated_code.assert_called_once()
        mock_executor.execute_sql.assert_called_once()
        mock_llm.format_response.assert_called_once()
        
        # Verify state progression
        assert result.query_analysis is not None
        assert result.query_analysis["query_type"] == "data_retrieval"
        assert len(result.retrieved_context) == 2
        assert result.generated_code == "SELECT * FROM sales WHERE revenue > 1000"
        assert result.code_type == "sql"
        assert result.execution_results is not None
        assert result.execution_results.success is True
        assert result.formatted_response is not None
        
        # Verify conversation history
        assert len(result.conversation_history) == 2
        assert result.conversation_history[0].role == "user"
        assert result.conversation_history[0].content == query
        assert result.conversation_history[1].role == "assistant"
        assert result.conversation_history[1].content == result.formatted_response
    
    @pytest.mark.asyncio
    async def test_complete_python_workflow(self, workflow, mock_llm, mock_knowledge, mock_executor):
        """Test a complete Python workflow for visualization."""
        # Configure for Python/visualization workflow
        mock_llm.analyze_query.return_value = {
            "intent": "data visualization",
            "query_type": "visualization",
            "entities": ["sales", "chart"],
            "complexity": "moderate",
            "requires_clarification": False,
            "suggested_approach": "Generate Python visualization code"
        }
        
        mock_llm.generate_code.return_value = """
import matplotlib.pyplot as plt
import pandas as pd

data = {'Product': ['A', 'B', 'C'], 'Revenue': [1500, 2000, 1200]}
df = pd.DataFrame(data)
plt.bar(df['Product'], df['Revenue'])
plt.title('Revenue by Product')
plt.show()
"""
        
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
    async def test_multi_turn_conversation(self, workflow):
        """Test multi-turn conversation with context preservation."""
        # First query
        query1 = "What tables are available?"
        result1 = await workflow.process_query(query1, conversation_id="test-conv")
        
        assert result1.workflow_complete is True
        assert len(result1.conversation_history) == 2
        
        # Second query with context
        query2 = "Show me data from the sales table"
        result2 = await workflow.process_query(query2, conversation_id="test-conv", existing_state=result1)
        
        # Verify conversation history is preserved and extended
        assert len(result2.conversation_history) == 4
        assert result2.conversation_history[0].content == query1
        assert result2.conversation_history[2].content == query2
        assert result2.workflow_complete is True
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, workflow, mock_executor):
        """Test error handling and graceful degradation."""
        # Mock execution failure
        mock_executor.execute_sql.return_value = ExecutionResult(
            success=False,
            error="Table 'sales' does not exist",
            execution_time=0.1,
            code_executed="SELECT * FROM sales WHERE revenue > 1000"
        )
        
        query = "Show me sales data"
        result = await workflow.process_query(query)
        
        # Verify error handling
        assert result.workflow_complete is True
        assert result.error_occurred is True
        assert "Code execution failed" in result.error_message
        assert result.execution_results is not None
        assert result.execution_results.success is False
    
    @pytest.mark.asyncio
    async def test_approval_workflow(self, workflow, mock_llm, agent_config):
        """Test approval workflow for high-risk operations."""
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
    async def test_clarification_workflow(self, workflow, mock_llm):
        """Test clarification workflow for ambiguous queries."""
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
        
        # Verify clarification workflow
        assert result.workflow_complete is True
        assert result.error_occurred is False
        assert result.current_step == "complete"
        assert result.formatted_response is not None
        assert "specify" in result.formatted_response
        
        # Verify only query analysis and clarification were called
        mock_llm.analyze_query.assert_called_once()
        mock_llm.generate_clarification.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_workflow_state_transitions(self, workflow):
        """Test that workflow state transitions correctly through all steps."""
        query = "Show me sales data with revenue greater than 1000"
        
        result = await workflow.process_query(query)
        
        # Verify final state
        assert result.current_step == "complete"
        assert result.workflow_complete is True
        
        # Verify state contains all expected data
        assert result.current_query == query
        assert result.query_analysis is not None
        assert result.retrieved_context is not None
        assert result.generated_code is not None
        assert result.code_type is not None
        assert result.code_validation is not None
        assert result.execution_results is not None
        assert result.formatted_response is not None
    
    @pytest.mark.asyncio
    async def test_workflow_performance_metrics(self, workflow):
        """Test that workflow captures performance metrics."""
        query = "Show me sales data with revenue greater than 1000"
        
        result = await workflow.process_query(query)
        
        # Verify execution results contain performance metrics
        assert result.execution_results is not None
        assert result.execution_results.execution_time is not None
        assert result.execution_results.execution_time > 0
        assert result.execution_results.code_executed is not None
    
    def test_workflow_configuration(self, workflow):
        """Test workflow configuration and info retrieval."""
        info = workflow.get_workflow_info()
        
        assert isinstance(info, dict)
        assert info["agent_name"] == "integration-test-agent"
        assert info["max_iterations"] == 5
        assert info["require_approval"] is False
        assert "nodes" in info
        assert len(info["nodes"]) == 6
        
        expected_nodes = [
            "query_processor",
            "context_retriever",
            "code_generator",
            "approval_gate",
            "executor",
            "response_formatter"
        ]
        assert all(node in info["nodes"] for node in expected_nodes)