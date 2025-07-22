"""Integration tests for complete DataAgent functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from src.dataqa.agent.agent import DataAgent, create_agent_from_config
from src.dataqa.config.models import (
    AgentConfig, LLMConfig, KnowledgeConfig, ExecutorConfig, WorkflowConfig,
    LLMProvider, KnowledgeProvider, ExecutorProvider
)
from src.dataqa.models.document import Document
from src.dataqa.models.execution import ExecutionResult
from src.dataqa.models.message import Message


@pytest.fixture
def agent_config():
    """Create a comprehensive test agent configuration."""
    return AgentConfig(
        name="test-data-agent",
        description="Test DataAgent for integration testing",
        version="1.0.0",
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            temperature=0.1,
            max_tokens=2000
        ),
        knowledge=KnowledgeConfig(
            provider=KnowledgeProvider.FAISS,
            embedding_model="all-MiniLM-L6-v2",
            top_k=5
        ),
        executor=ExecutorConfig(
            provider=ExecutorProvider.INMEMORY,
            database_type="duckdb",
            max_execution_time=30.0
        ),
        workflow=WorkflowConfig(
            strategy="react",
            max_iterations=10,
            require_approval=False,
            conversation_memory=True
        )
    )


@pytest.fixture
def mock_llm():
    """Create a comprehensive mock LLM interface."""
    llm = AsyncMock()
    
    # Mock query analysis
    llm.analyze_query.return_value = {
        "intent": "data analysis",
        "query_type": "data_retrieval",
        "entities": ["sales", "revenue"],
        "complexity": "moderate",
        "requires_clarification": False,
        "suggested_approach": "Generate SQL query"
    }
    
    # Mock code generation
    llm.generate_code.return_value = "SELECT * FROM sales WHERE revenue > 1000"
    
    # Mock code validation
    llm.validate_generated_code.return_value = {
        "is_valid": True,
        "issues": [],
        "security_concerns": [],
        "suggestions": [],
        "risk_level": "low"
    }
    
    # Mock response formatting
    llm.format_response.return_value = "Here are your sales results with revenue greater than 1000."
    
    # Mock model info
    llm.get_model_info.return_value = {"model": "gpt-4", "status": "available"}
    
    return llm


@pytest.fixture
def mock_knowledge():
    """Create a comprehensive mock knowledge primitive."""
    knowledge = AsyncMock()
    
    # Mock search results
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
    
    # Mock knowledge stats
    knowledge.get_stats.return_value = {"document_count": 100, "index_size": "5MB"}
    
    return knowledge


@pytest.fixture
def mock_executor():
    """Create a comprehensive mock executor primitive."""
    executor = AsyncMock()
    
    # Mock SQL execution
    executor.execute_sql.return_value = ExecutionResult(
        success=True,
        data={
            "columns": ["id", "product", "revenue"],
            "data": [
                [1, "Widget A", 1500.0],
                [2, "Widget B", 2000.0],
                [3, "Widget C", 1200.0]
            ]
        },
        execution_time=0.25,
        code_executed="SELECT * FROM sales WHERE revenue > 1000",
        output_type="dataframe"
    )
    
    # Mock Python execution
    executor.execute_python.return_value = ExecutionResult(
        success=True,
        data={"plot_data": "base64_encoded_image"},
        execution_time=1.2,
        code_executed="Python visualization code",
        output_type="plot"
    )
    
    # Mock schema information
    executor.get_schema.return_value = {
        "tables": ["sales", "products", "customers"],
        "sales": {
            "columns": ["id", "product", "revenue", "date"],
            "types": ["int", "varchar", "decimal", "date"]
        }
    }
    
    # Mock table listing
    executor.list_tables.return_value = ["sales", "products", "customers"]
    
    return executor


@pytest.fixture
def data_agent(agent_config, mock_llm, mock_knowledge, mock_executor):
    """Create a DataAgent instance with mocked components."""
    return DataAgent(
        config=agent_config,
        llm=mock_llm,
        knowledge=mock_knowledge,
        executor=mock_executor
    )


class TestDataAgentIntegration:
    """Integration tests for complete DataAgent functionality."""
    
    def test_agent_initialization(self, data_agent, agent_config):
        """Test DataAgent initialization with configuration."""
        assert data_agent.config == agent_config
        assert data_agent.llm is not None
        assert data_agent.knowledge is not None
        assert data_agent.executor is not None
        assert data_agent.workflow is not None
        assert len(data_agent._conversations) == 0
    
    def test_agent_info(self, data_agent):
        """Test agent information retrieval."""
        info = data_agent.get_agent_info()
        
        assert isinstance(info, dict)
        assert info["name"] == "test-data-agent"
        assert info["description"] == "Test DataAgent for integration testing"
        assert info["version"] == "1.0.0"
        assert info["llm_provider"] == "openai"
        assert info["llm_model"] == "gpt-4"
        assert info["knowledge_provider"] == "faiss"
        assert info["executor_provider"] == "inmemory"
        assert "workflow_info" in info
        assert info["active_conversations"] == 0
    
    @pytest.mark.asyncio
    async def test_basic_query_processing(self, data_agent):
        """Test basic query processing functionality."""
        query = "Show me sales data with revenue greater than 1000"
        
        response = await data_agent.query(query)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "sales results" in response.lower()
        
        # Verify conversation was stored
        assert len(data_agent._conversations) == 1
        assert "default" in data_agent._conversations
        
        # Verify conversation history
        history = await data_agent.get_conversation_history("default")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == query
        assert history[1].role == "assistant"
        assert history[1].content == response
    
    @pytest.mark.asyncio
    async def test_query_with_conversation_id(self, data_agent):
        """Test query processing with specific conversation ID."""
        query = "What tables are available?"
        conversation_id = "test-conversation"
        
        response = await data_agent.query(query, conversation_id=conversation_id)
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Verify conversation was stored with correct ID
        assert conversation_id in data_agent._conversations
        
        # Verify conversation status
        status = await data_agent.get_conversation_status(conversation_id)
        assert status["exists"] is True
        assert status["workflow_complete"] is True
        assert status["error_occurred"] is False
        assert status["message_count"] == 2
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, data_agent):
        """Test multi-turn conversation with context preservation."""
        conversation_id = "multi-turn-test"
        
        # First query
        query1 = "What tables are available?"
        response1 = await data_agent.query(query1, conversation_id=conversation_id)
        assert len(response1) > 0
        
        # Second query in same conversation
        query2 = "Show me data from the sales table"
        response2 = await data_agent.query(query2, conversation_id=conversation_id)
        assert len(response2) > 0
        
        # Verify conversation history contains both exchanges
        history = await data_agent.get_conversation_history(conversation_id)
        assert len(history) == 4
        assert history[0].content == query1
        assert history[1].content == response1
        assert history[2].content == query2
        assert history[3].content == response2
        
        # Verify conversation status
        status = await data_agent.get_conversation_status(conversation_id)
        assert status["message_count"] == 4
    
    @pytest.mark.asyncio
    async def test_approval_workflow(self, data_agent, mock_llm):
        """Test approval workflow for high-risk operations."""
        # Enable approval requirement
        data_agent.config.workflow.require_approval = True
        data_agent.workflow.nodes.require_approval = True
        
        # Mock high-risk code validation
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True,
            "issues": [],
            "security_concerns": ["Potential data modification"],
            "suggestions": ["Review before execution"],
            "risk_level": "high"
        }
        
        conversation_id = "approval-test"
        query = "Delete old sales records"
        
        # Initial query should trigger approval request
        response = await data_agent.query(query, conversation_id=conversation_id)
        
        assert "approval" in response.lower()
        assert "approve" in response.lower()
        
        # Verify conversation status shows pending approval
        status = await data_agent.get_conversation_status(conversation_id)
        assert status["pending_approval"] is True
        
        # Test approval
        approval_response = await data_agent.approve_operation(
            conversation_id=conversation_id,
            approved=True
        )
        
        assert isinstance(approval_response, str)
        assert len(approval_response) > 0
        
        # Test denial
        query2 = "DROP TABLE sales"
        await data_agent.query(query2, conversation_id=conversation_id)
        
        denial_response = await data_agent.approve_operation(
            conversation_id=conversation_id,
            approved=False,
            reason="Too risky"
        )
        
        assert "error" in denial_response.lower() or "denied" in denial_response.lower()
    
    @pytest.mark.asyncio
    async def test_knowledge_operations(self, data_agent, mock_knowledge):
        """Test knowledge base operations."""
        # Test knowledge ingestion
        documents = [
            Document(
                content="Test document content",
                source="test_source",
                metadata={"type": "test"}
            )
        ]
        
        await data_agent.ingest_knowledge(documents)
        mock_knowledge.ingest.assert_called_once_with(documents)
        
        # Test knowledge search
        results = await data_agent.search_knowledge("test query", limit=3)
        
        assert isinstance(results, list)
        assert len(results) == 2  # Based on mock return value
        assert all(isinstance(doc, Document) for doc in results)
        
        mock_knowledge.search.assert_called_with("test query", 3, None)
    
    @pytest.mark.asyncio
    async def test_database_operations(self, data_agent, mock_executor):
        """Test database schema and table operations."""
        # Test schema retrieval
        schema = await data_agent.get_database_schema()
        
        assert isinstance(schema, dict)
        assert "tables" in schema
        assert "sales" in schema["tables"]
        
        mock_executor.get_schema.assert_called_once_with(None)
        
        # Test specific table schema
        table_schema = await data_agent.get_database_schema("sales")
        mock_executor.get_schema.assert_called_with("sales")
        
        # Test table listing
        tables = await data_agent.list_database_tables()
        
        assert isinstance(tables, list)
        assert "sales" in tables
        assert "products" in tables
        assert "customers" in tables
        
        mock_executor.list_tables.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_conversation_management(self, data_agent):
        """Test conversation management operations."""
        conversation_id = "management-test"
        
        # Create a conversation
        await data_agent.query("Test query", conversation_id=conversation_id)
        
        # Verify conversation exists
        status = await data_agent.get_conversation_status(conversation_id)
        assert status["exists"] is True
        
        # Clear conversation
        cleared = await data_agent.clear_conversation(conversation_id)
        assert cleared is True
        
        # Verify conversation is cleared
        status = await data_agent.get_conversation_status(conversation_id)
        assert status["exists"] is False
        
        # Try to clear non-existent conversation
        cleared = await data_agent.clear_conversation("non-existent")
        assert cleared is False
    
    @pytest.mark.asyncio
    async def test_health_check(self, data_agent, mock_llm, mock_knowledge, mock_executor):
        """Test agent health check functionality."""
        health = await data_agent.health_check()
        
        assert isinstance(health, dict)
        assert "agent" in health
        assert "llm" in health
        assert "knowledge" in health
        assert "executor" in health
        assert "timestamp" in health
        
        assert health["agent"] == "healthy"
        assert health["llm"] == "healthy"
        assert health["knowledge"] == "healthy"
        assert health["executor"] == "healthy"
        
        # Verify all components were checked
        mock_llm.get_model_info.assert_called_once()
        mock_knowledge.get_stats.assert_called_once()
        mock_executor.list_tables.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_with_failures(self, data_agent, mock_llm, mock_knowledge, mock_executor):
        """Test health check with component failures."""
        # Mock component failures
        mock_llm.get_model_info.side_effect = Exception("LLM connection failed")
        mock_knowledge.get_stats.side_effect = Exception("Knowledge base unavailable")
        mock_executor.list_tables.side_effect = Exception("Database connection failed")
        
        health = await data_agent.health_check()
        
        assert "unhealthy: LLM connection failed" in health["llm"]
        assert "unhealthy: Knowledge base unavailable" in health["knowledge"]
        assert "unhealthy: Database connection failed" in health["executor"]
        assert health["agent"] == "healthy"  # Agent itself is still healthy
    
    @pytest.mark.asyncio
    async def test_error_handling(self, data_agent, mock_executor):
        """Test error handling in query processing."""
        # Mock execution failure
        mock_executor.execute_sql.return_value = ExecutionResult(
            success=False,
            error="Table 'nonexistent' does not exist",
            execution_time=0.1,
            code_executed="SELECT * FROM nonexistent"
        )
        
        query = "Show me data from nonexistent table"
        response = await data_agent.query(query)
        
        assert isinstance(response, str)
        assert "error" in response.lower()
        
        # Verify conversation status shows error
        status = await data_agent.get_conversation_status("default")
        assert status["error_occurred"] is True
    
    @pytest.mark.asyncio
    async def test_agent_shutdown(self, data_agent):
        """Test agent shutdown and cleanup."""
        # Create some conversations
        await data_agent.query("Test query 1", "conv1")
        await data_agent.query("Test query 2", "conv2")
        
        assert len(data_agent._conversations) == 2
        
        # Shutdown agent
        await data_agent.shutdown()
        
        # Verify conversations are cleared
        assert len(data_agent._conversations) == 0


class TestDataAgentFactory:
    """Test DataAgent factory functions."""
    
    @pytest.mark.asyncio
    async def test_create_agent_from_config(self, agent_config):
        """Test agent creation from configuration."""
        with patch('src.dataqa.agent.agent.DataAgent') as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.health_check.return_value = {
                "agent": "healthy",
                "llm": "healthy",
                "knowledge": "healthy",
                "executor": "healthy"
            }
            mock_agent_class.return_value = mock_agent
            
            agent = await create_agent_from_config(agent_config)
            
            assert agent is not None
            mock_agent_class.assert_called_once_with(agent_config)
            mock_agent.health_check.assert_called_once()


class TestDataAgentComponentIntegration:
    """Test integration between DataAgent and its components."""
    
    @pytest.mark.asyncio
    async def test_llm_integration(self, data_agent, mock_llm):
        """Test LLM integration through complete workflow."""
        query = "Analyze sales trends"
        
        await data_agent.query(query)
        
        # Verify LLM methods were called in correct sequence
        mock_llm.analyze_query.assert_called_once()
        mock_llm.generate_code.assert_called_once()
        mock_llm.validate_generated_code.assert_called_once()
        mock_llm.format_response.assert_called_once()
        
        # Verify query was passed correctly
        analyze_call = mock_llm.analyze_query.call_args
        assert analyze_call[1]["query"] == query
    
    @pytest.mark.asyncio
    async def test_knowledge_integration(self, data_agent, mock_knowledge):
        """Test knowledge base integration through complete workflow."""
        query = "Show me sales data"
        
        await data_agent.query(query)
        
        # Verify knowledge search was called
        mock_knowledge.search.assert_called_once()
        
        # Verify search parameters
        search_call = mock_knowledge.search.call_args
        assert search_call[1]["query"] == query
        assert search_call[1]["limit"] == 5
    
    @pytest.mark.asyncio
    async def test_executor_integration(self, data_agent, mock_executor):
        """Test executor integration through complete workflow."""
        query = "SELECT * FROM sales"
        
        await data_agent.query(query)
        
        # Verify executor was called
        mock_executor.execute_sql.assert_called_once()
        
        # Verify code was passed correctly
        execute_call = mock_executor.execute_sql.call_args
        assert "SELECT * FROM sales" in execute_call[0][0]


class TestDataAgentPerformance:
    """Test DataAgent performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, data_agent):
        """Test handling of concurrent queries."""
        import asyncio
        
        queries = [
            ("Show me sales data", "conv1"),
            ("List all products", "conv2"),
            ("Get customer count", "conv3")
        ]
        
        # Execute queries concurrently
        tasks = [
            data_agent.query(query, conv_id)
            for query, conv_id in queries
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # Verify all queries completed successfully
        assert len(responses) == 3
        assert all(isinstance(response, str) for response in responses)
        assert all(len(response) > 0 for response in responses)
        
        # Verify separate conversations were maintained
        assert len(data_agent._conversations) == 3
    
    @pytest.mark.asyncio
    async def test_memory_management(self, data_agent):
        """Test memory management with multiple conversations."""
        # Create many conversations
        for i in range(10):
            await data_agent.query(f"Query {i}", f"conv_{i}")
        
        assert len(data_agent._conversations) == 10
        
        # Clear some conversations
        for i in range(5):
            await data_agent.clear_conversation(f"conv_{i}")
        
        assert len(data_agent._conversations) == 5
        
        # Verify remaining conversations are intact
        for i in range(5, 10):
            status = await data_agent.get_conversation_status(f"conv_{i}")
            assert status["exists"] is True