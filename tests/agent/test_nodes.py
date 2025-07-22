"""Tests for workflow nodes."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.dataqa.agent.nodes import WorkflowNodes
from src.dataqa.agent.state import SharedState
from src.dataqa.exceptions import ExecutionError, KnowledgeError, LLMError
from src.dataqa.models.document import Document
from src.dataqa.models.execution import ExecutionResult
from src.dataqa.models.message import Message


@pytest.fixture
def mock_llm():
    """Create a mock LLM interface."""
    llm = AsyncMock()
    llm.analyze_query.return_value = {
        "intent": "data analysis",
        "query_type": "analysis",
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
    llm.format_response.return_value = "Here are your sales results..."
    llm.generate_clarification.return_value = "Could you please clarify what time period you're interested in?"
    return llm


@pytest.fixture
def mock_knowledge():
    """Create a mock knowledge primitive."""
    knowledge = AsyncMock()
    knowledge.search.return_value = [
        Document(
            content="Sales table contains revenue data",
            source="schema_doc",
            metadata={"table": "sales"}
        )
    ]
    return knowledge


@pytest.fixture
def mock_executor():
    """Create a mock executor primitive."""
    executor = AsyncMock()
    executor.execute_sql.return_value = ExecutionResult(
        success=True,
        data={"revenue": [1000, 2000, 3000]},
        execution_time=0.5,
        code_executed="SELECT * FROM sales WHERE revenue > 1000",
        output_type="dataframe"
    )
    executor.execute_python.return_value = ExecutionResult(
        success=True,
        data={"plot": "base64_image_data"},
        execution_time=1.0,
        code_executed="import matplotlib.pyplot as plt",
        output_type="plot"
    )
    return executor


@pytest.fixture
def workflow_nodes(mock_llm, mock_knowledge, mock_executor):
    """Create WorkflowNodes instance with mocked dependencies."""
    return WorkflowNodes(
        llm=mock_llm,
        knowledge=mock_knowledge,
        executor=mock_executor,
        require_approval=True
    )


@pytest.fixture
def sample_state():
    """Create a sample SharedState for testing."""
    state = SharedState(current_query="Show me sales data with revenue > 1000")
    state.add_message("user", "Show me sales data with revenue > 1000")
    return state


class TestWorkflowNodes:
    """Test cases for WorkflowNodes."""
    
    @pytest.mark.asyncio
    async def test_query_processor_success(self, workflow_nodes, sample_state, mock_llm):
        """Test successful query processing."""
        result = await workflow_nodes.query_processor(sample_state)
        
        # Node returns the state directly
        state = result
        
        assert state.query_analysis is not None
        assert state.query_analysis["query_type"] == "analysis"
        assert state.current_step == "context_retriever"
        assert not state.workflow_complete
        
        mock_llm.analyze_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_processor_requires_clarification(self, workflow_nodes, sample_state, mock_llm):
        """Test query processing when clarification is needed."""
        # Mock analysis that requires clarification
        mock_llm.analyze_query.return_value = {
            "intent": "unclear",
            "query_type": "unknown",
            "entities": [],
            "complexity": "high",
            "requires_clarification": True,
            "ambiguities": ["Time period unclear", "Metric not specified"]
        }
        
        result = await workflow_nodes.query_processor(sample_state)
        state = result
        
        assert state.workflow_complete is True
        assert state.current_step == "complete"
        assert state.formatted_response is not None
        assert "clarify" in state.formatted_response.lower()
        
        mock_llm.generate_clarification.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_processor_llm_error(self, workflow_nodes, sample_state, mock_llm):
        """Test query processing with LLM error."""
        mock_llm.analyze_query.side_effect = LLMError("API error")
        
        result = await workflow_nodes.query_processor(sample_state)
        state = result
        
        assert state.error_occurred is True
        assert "Failed to analyze query" in state.error_message
        assert state.workflow_complete is True
    
    @pytest.mark.asyncio
    async def test_context_retriever_success(self, workflow_nodes, sample_state, mock_knowledge):
        """Test successful context retrieval."""
        result = await workflow_nodes.context_retriever(sample_state)
        state = result
        
        assert len(state.retrieved_context) == 1
        assert state.context_summary is not None
        assert "Sales table contains revenue data" in state.context_summary
        assert state.current_step == "code_generator"
        
        mock_knowledge.search.assert_called_once_with(
            query=sample_state.current_query,
            limit=5
        )
    
    @pytest.mark.asyncio
    async def test_context_retriever_no_results(self, workflow_nodes, sample_state, mock_knowledge):
        """Test context retrieval with no results."""
        mock_knowledge.search.return_value = []
        
        result = await workflow_nodes.context_retriever(sample_state)
        state = result
        
        assert len(state.retrieved_context) == 0
        assert "No relevant context found" in state.context_summary
        assert state.current_step == "code_generator"
    
    @pytest.mark.asyncio
    async def test_context_retriever_knowledge_error(self, workflow_nodes, sample_state, mock_knowledge):
        """Test context retrieval with knowledge error."""
        mock_knowledge.search.side_effect = KnowledgeError("Search failed")
        
        result = await workflow_nodes.context_retriever(sample_state)
        state = result
        
        assert len(state.retrieved_context) == 0
        assert "Context retrieval failed" in state.context_summary
        assert state.current_step == "code_generator"  # Should continue despite error
    
    @pytest.mark.asyncio
    async def test_code_generator_success_low_risk(self, workflow_nodes, sample_state, mock_llm):
        """Test successful code generation with low risk."""
        # Set up state with query analysis
        sample_state.query_analysis = {"query_type": "data_retrieval"}
        sample_state.context_summary = "Sales table schema"
        
        result = await workflow_nodes.code_generator(sample_state)
        state = result
        
        assert state.generated_code is not None
        assert state.code_type == "sql"
        assert state.code_validation is not None
        assert state.code_validation["is_valid"] is True
        assert state.current_step == "executor"  # Low risk, no approval needed
        
        mock_llm.generate_code.assert_called_once()
        mock_llm.validate_generated_code.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_code_generator_success_high_risk(self, workflow_nodes, sample_state, mock_llm):
        """Test successful code generation with high risk requiring approval."""
        # Set up state
        sample_state.query_analysis = {"query_type": "analysis"}
        sample_state.context_summary = "Sales table schema"
        
        # Mock high-risk validation
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True,
            "issues": [],
            "security_concerns": ["Potential data modification"],
            "suggestions": [],
            "risk_level": "high"
        }
        
        result = await workflow_nodes.code_generator(sample_state)
        state = result
        
        assert state.generated_code is not None
        assert state.code_type == "python"
        assert state.pending_approval is not None
        assert state.current_step == "approval_gate"
    
    @pytest.mark.asyncio
    async def test_code_generator_invalid_code(self, workflow_nodes, sample_state, mock_llm):
        """Test code generation with invalid code."""
        # Set up state
        sample_state.query_analysis = {"query_type": "analysis"}
        sample_state.context_summary = "Sales table schema"
        
        # Mock invalid validation
        mock_llm.validate_generated_code.return_value = {
            "is_valid": False,
            "issues": ["Syntax error", "Invalid table name"],
            "security_concerns": [],
            "suggestions": [],
            "risk_level": "low"
        }
        
        result = await workflow_nodes.code_generator(sample_state)
        state = result
        
        assert state.error_occurred is True
        assert "failed validation" in state.error_message
        assert state.workflow_complete is True
    
    @pytest.mark.asyncio
    async def test_approval_gate_pending(self, workflow_nodes, sample_state):
        """Test approval gate with pending approval."""
        sample_state.pending_approval = "SELECT * FROM sensitive_table"
        sample_state.code_type = "sql"
        sample_state.code_validation = {"risk_level": "high", "issues": [], "security_concerns": ["Data access"]}
        
        result = await workflow_nodes.approval_gate(sample_state)
        state = result
        
        assert state.workflow_complete is True
        assert state.current_step == "awaiting_approval"
        assert state.formatted_response is not None
        assert "approve" in state.formatted_response.lower()
    
    @pytest.mark.asyncio
    async def test_approval_gate_granted(self, workflow_nodes, sample_state):
        """Test approval gate with granted approval."""
        sample_state.pending_approval = "SELECT * FROM table"
        sample_state.approval_granted = True
        
        result = await workflow_nodes.approval_gate(sample_state)
        state = result
        
        assert state.current_step == "executor"
        assert not state.workflow_complete
    
    @pytest.mark.asyncio
    async def test_executor_sql_success(self, workflow_nodes, sample_state, mock_executor):
        """Test successful SQL execution."""
        sample_state.generated_code = "SELECT * FROM sales"
        sample_state.code_type = "sql"
        sample_state.code_validation = {"is_valid": True, "risk_level": "low"}
        
        result = await workflow_nodes.execute_code(sample_state)
        state = result
        
        assert state.execution_results is not None
        assert state.execution_results.success is True
        assert state.current_step == "response_formatter"
        
        mock_executor.execute_sql.assert_called_once_with("SELECT * FROM sales")
    
    @pytest.mark.asyncio
    async def test_executor_python_success(self, workflow_nodes, sample_state, mock_executor):
        """Test successful Python execution."""
        sample_state.generated_code = "import pandas as pd"
        sample_state.code_type = "python"
        sample_state.code_validation = {"is_valid": True, "risk_level": "low"}
        
        result = await workflow_nodes.execute_code(sample_state)
        state = result
        
        assert state.execution_results is not None
        assert state.execution_results.success is True
        assert state.current_step == "response_formatter"
        
        mock_executor.execute_python.assert_called_once_with("import pandas as pd")
    
    @pytest.mark.asyncio
    async def test_executor_execution_failure(self, workflow_nodes, sample_state, mock_executor):
        """Test executor with execution failure."""
        sample_state.generated_code = "SELECT * FROM nonexistent_table"
        sample_state.code_type = "sql"
        sample_state.code_validation = {"is_valid": True, "risk_level": "low"}
        
        # Mock failed execution
        mock_executor.execute_sql.return_value = ExecutionResult(
            success=False,
            error="Table 'nonexistent_table' does not exist",
            execution_time=0.1,
            code_executed="SELECT * FROM nonexistent_table"
        )
        
        result = await workflow_nodes.execute_code(sample_state)
        state = result
        
        assert state.error_occurred is True
        assert "Code execution failed" in state.error_message
        assert state.workflow_complete is True
    
    @pytest.mark.asyncio
    async def test_executor_no_code(self, workflow_nodes, sample_state):
        """Test executor with no code to execute."""
        # Don't set generated_code
        
        result = await workflow_nodes.execute_code(sample_state)
        state = result
        
        assert state.error_occurred is True
        assert "No code available" in state.error_message
        assert state.workflow_complete is True
    
    @pytest.mark.asyncio
    async def test_response_formatter_success(self, workflow_nodes, sample_state, mock_llm):
        """Test successful response formatting."""
        sample_state.execution_results = ExecutionResult(
            success=True,
            data={"revenue": [1000, 2000]},
            execution_time=0.5,
            code_executed="SELECT revenue FROM sales",
            output_type="dataframe"
        )
        
        result = await workflow_nodes.response_formatter(sample_state)
        state = result
        
        assert state.formatted_response is not None
        assert state.workflow_complete is True
        assert state.current_step == "complete"
        
        mock_llm.format_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_response_formatter_llm_error(self, workflow_nodes, sample_state, mock_llm):
        """Test response formatting with LLM error (should use fallback)."""
        sample_state.execution_results = ExecutionResult(
            success=True,
            data={"revenue": [1000, 2000]},
            execution_time=0.5,
            code_executed="SELECT revenue FROM sales"
        )
        
        mock_llm.format_response.side_effect = LLMError("Formatting failed")
        
        result = await workflow_nodes.response_formatter(sample_state)
        state = result
        
        assert state.formatted_response is not None
        assert "Query Executed Successfully" in state.formatted_response
        assert state.workflow_complete is True
        assert state.current_step == "complete"
    
    @pytest.mark.asyncio
    async def test_response_formatter_no_results(self, workflow_nodes, sample_state):
        """Test response formatting with no execution results."""
        # Don't set execution_results
        
        result = await workflow_nodes.response_formatter(sample_state)
        state = result
        
        assert state.error_occurred is True
        assert "No execution results available" in state.error_message
        assert state.workflow_complete is True
    
    def test_grant_approval(self, workflow_nodes, sample_state):
        """Test granting approval for pending operations."""
        sample_state.pending_approval = "SELECT * FROM table"
        
        result = workflow_nodes.grant_approval(sample_state)
        state = result
        
        assert state.approval_granted is True
        assert state.current_step == "executor"
    
    def test_grant_approval_no_pending(self, workflow_nodes, sample_state):
        """Test granting approval when no approval is pending."""
        # Don't set pending_approval
        
        result = workflow_nodes.grant_approval(sample_state)
        state = result
        
        # Should not change state significantly
        assert not state.approval_granted
    
    def test_deny_approval(self, workflow_nodes, sample_state):
        """Test denying approval for pending operations."""
        sample_state.pending_approval = "SELECT * FROM table"
        
        result = workflow_nodes.deny_approval(sample_state, "Too risky")
        state = result
        
        assert state.error_occurred is True
        assert "Too risky" in state.error_message
        assert state.workflow_complete is True
    
    def test_deny_approval_no_pending(self, workflow_nodes, sample_state):
        """Test denying approval when no approval is pending."""
        # Don't set pending_approval
        
        result = workflow_nodes.deny_approval(sample_state)
        state = result
        
        # Should not change state significantly
        assert not state.error_occurred