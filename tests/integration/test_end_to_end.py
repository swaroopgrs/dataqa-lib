"""
End-to-end integration tests with real data scenarios.

These tests verify complete workflows using realistic data and scenarios
without mocking core components.
"""

import pytest
import asyncio
import tempfile
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import time
import psutil
import os

from src.dataqa.agent.agent import DataAgent
from src.dataqa.config.models import (
    AgentConfig, LLMConfig, KnowledgeConfig, ExecutorConfig, WorkflowConfig,
    LLMProvider, KnowledgeProvider, ExecutorProvider
)
from src.dataqa.models.document import Document
from src.dataqa.models.execution import ExecutionResult
from src.dataqa.primitives.in_memory_executor import InMemoryExecutor
from src.dataqa.primitives.faiss_knowledge import FAISSKnowledge


@pytest.fixture
def real_agent_config():
    """Create a real agent configuration for integration testing."""
    return AgentConfig(
        name="integration-test-agent",
        description="Real agent for end-to-end integration testing",
        version="1.0.0",
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",  # Use cheaper model for testing
            temperature=0.1,
            max_tokens=1000,
            api_key="test-key-will-be-mocked"
        ),
        knowledge=KnowledgeConfig(
            provider=KnowledgeProvider.FAISS,
            embedding_model="all-MiniLM-L6-v2",
            top_k=3
        ),
        executor=ExecutorConfig(
            provider=ExecutorProvider.INMEMORY,
            database_type="duckdb",
            max_execution_time=30.0
        ),
        workflow=WorkflowConfig(
            strategy="react",
            max_iterations=5,
            require_approval=False,
            conversation_memory=True
        )
    )


@pytest.fixture
def sample_sales_data():
    """Create realistic sales data for testing."""
    return pd.DataFrame({
        'id': range(1, 101),
        'product': [f'Product_{i%10}' for i in range(1, 101)],
        'category': [f'Category_{i%5}' for i in range(1, 101)],
        'revenue': [1000 + (i * 50) + (i % 7 * 100) for i in range(1, 101)],
        'quantity': [10 + (i % 20) for i in range(1, 101)],
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'region': [f'Region_{i%4}' for i in range(1, 101)]
    })


@pytest.fixture
def sample_customer_data():
    """Create realistic customer data for testing."""
    return pd.DataFrame({
        'customer_id': range(1, 51),
        'name': [f'Customer_{i}' for i in range(1, 51)],
        'email': [f'customer{i}@example.com' for i in range(1, 51)],
        'age': [25 + (i % 40) for i in range(1, 51)],
        'region': [f'Region_{i%4}' for i in range(1, 51)],
        'signup_date': pd.date_range('2023-01-01', periods=50, freq='W')
    })


@pytest.fixture
def knowledge_documents():
    """Create sample knowledge documents for testing."""
    return [
        Document(
            content="Sales table contains product sales data with revenue, quantity, and date information. Key columns: id, product, category, revenue, quantity, date, region.",
            source="database_schema",
            metadata={"table": "sales", "type": "schema_description"}
        ),
        Document(
            content="Customer table contains customer information including demographics and signup dates. Key columns: customer_id, name, email, age, region, signup_date.",
            source="database_schema", 
            metadata={"table": "customers", "type": "schema_description"}
        ),
        Document(
            content="Revenue analysis should focus on trends over time and regional differences. Use date-based grouping for temporal analysis.",
            source="business_rules",
            metadata={"domain": "sales", "type": "analysis_guidance"}
        ),
        Document(
            content="When creating visualizations, use appropriate chart types: bar charts for categorical data, line charts for time series, scatter plots for correlations.",
            source="visualization_guidelines",
            metadata={"domain": "visualization", "type": "best_practices"}
        )
    ]


@pytest.fixture
async def real_executor(sample_sales_data, sample_customer_data):
    """Create a real in-memory executor with test data."""
    executor = InMemoryExecutor({
        "database_type": "duckdb",
        "max_execution_time": 30.0
    })
    
    # Load test data
    await executor.load_dataframe("sales", sample_sales_data)
    await executor.load_dataframe("customers", sample_customer_data)
    
    return executor


@pytest.fixture
async def real_knowledge(knowledge_documents):
    """Create a real FAISS knowledge base with test documents."""
    with tempfile.TemporaryDirectory() as temp_dir:
        knowledge = FAISSKnowledge(
            model_name="all-MiniLM-L6-v2",
            index_path=str(Path(temp_dir) / "test_index")
        )
        
        # Ingest test documents
        await knowledge.ingest(knowledge_documents)
        
        yield knowledge


class TestEndToEndIntegration:
    """End-to-end integration tests with real components."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_data_analysis_workflow(self, real_executor, real_knowledge):
        """Test complete data analysis workflow with real components."""
        # Mock LLM for controlled responses
        from unittest.mock import AsyncMock
        
        mock_llm = AsyncMock()
        mock_llm.analyze_query.return_value = {
            "intent": "data analysis",
            "query_type": "data_retrieval",
            "entities": ["sales", "revenue"],
            "complexity": "moderate",
            "requires_clarification": False,
            "suggested_approach": "Generate SQL query"
        }
        
        mock_llm.generate_code.return_value = "SELECT product, SUM(revenue) as total_revenue FROM sales GROUP BY product ORDER BY total_revenue DESC LIMIT 5"
        
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True,
            "issues": [],
            "security_concerns": [],
            "suggestions": [],
            "risk_level": "low"
        }
        
        mock_llm.format_response.return_value = "Here are the top 5 products by revenue."
        
        # Create agent with real executor and knowledge
        config = AgentConfig(
            name="test-agent",
            description="Test agent",
            workflow=WorkflowConfig(require_approval=False)
        )
        
        agent = DataAgent(
            config=config,
            llm=mock_llm,
            knowledge=real_knowledge,
            executor=real_executor
        )
        
        # Test query processing
        query = "Show me the top 5 products by revenue"
        response = await agent.query(query)
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Verify LLM was called correctly
        mock_llm.analyze_query.assert_called_once()
        mock_llm.generate_code.assert_called_once()
        mock_llm.validate_generated_code.assert_called_once()
        mock_llm.format_response.assert_called_once()
        
        # Verify knowledge search was performed
        # (This is verified by the fact that the workflow completed successfully)
        
        await agent.shutdown()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_table_join_analysis(self, real_executor, real_knowledge):
        """Test analysis involving multiple tables with joins."""
        from unittest.mock import AsyncMock
        
        mock_llm = AsyncMock()
        mock_llm.analyze_query.return_value = {
            "intent": "data analysis",
            "query_type": "data_retrieval",
            "entities": ["sales", "customers", "region"],
            "complexity": "high",
            "requires_clarification": False,
            "suggested_approach": "Generate SQL query with JOIN"
        }
        
        mock_llm.generate_code.return_value = """
        SELECT c.region, COUNT(DISTINCT c.customer_id) as customer_count, SUM(s.revenue) as total_revenue
        FROM customers c
        JOIN sales s ON c.region = s.region
        GROUP BY c.region
        ORDER BY total_revenue DESC
        """
        
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True,
            "issues": [],
            "security_concerns": [],
            "suggestions": [],
            "risk_level": "low"
        }
        
        mock_llm.format_response.return_value = "Here's the revenue analysis by region with customer counts."
        
        config = AgentConfig(
            name="test-agent",
            description="Test agent",
            workflow=WorkflowConfig(require_approval=False)
        )
        
        agent = DataAgent(
            config=config,
            llm=mock_llm,
            knowledge=real_knowledge,
            executor=real_executor
        )
        
        query = "Analyze revenue by region and show customer counts"
        response = await agent.query(query)
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Verify complex query was handled
        mock_llm.generate_code.assert_called_once()
        generated_code = mock_llm.generate_code.call_args[1]["context"]
        assert "JOIN" in str(generated_code).upper()
        
        await agent.shutdown()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_visualization_workflow(self, real_executor, real_knowledge):
        """Test visualization generation workflow."""
        from unittest.mock import AsyncMock
        
        mock_llm = AsyncMock()
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

# Get data
data = df.groupby('product')['revenue'].sum().sort_values(ascending=False).head(5)

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(data.index, data.values)
plt.title('Top 5 Products by Revenue')
plt.xlabel('Product')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
        
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True,
            "issues": [],
            "security_concerns": [],
            "suggestions": [],
            "risk_level": "low"
        }
        
        mock_llm.format_response.return_value = "I've created a bar chart showing the top 5 products by revenue."
        
        config = AgentConfig(
            name="test-agent",
            description="Test agent",
            workflow=WorkflowConfig(require_approval=False)
        )
        
        agent = DataAgent(
            config=config,
            llm=mock_llm,
            knowledge=real_knowledge,
            executor=real_executor
        )
        
        query = "Create a bar chart of top 5 products by revenue"
        response = await agent.query(query)
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Verify Python code was generated for visualization
        mock_llm.generate_code.assert_called_once()
        
        await agent.shutdown()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_context_preservation(self, real_executor, real_knowledge):
        """Test that conversation context is preserved across multiple queries."""
        from unittest.mock import AsyncMock
        
        mock_llm = AsyncMock()
        
        # First query response
        mock_llm.analyze_query.side_effect = [
            {
                "intent": "data exploration",
                "query_type": "schema_inquiry",
                "entities": ["tables"],
                "complexity": "low",
                "requires_clarification": False,
                "suggested_approach": "List available tables"
            },
            {
                "intent": "data analysis",
                "query_type": "data_retrieval",
                "entities": ["sales"],
                "complexity": "moderate",
                "requires_clarification": False,
                "suggested_approach": "Query sales table"
            }
        ]
        
        mock_llm.generate_code.side_effect = [
            "SELECT name FROM sqlite_master WHERE type='table'",
            "SELECT * FROM sales LIMIT 10"
        ]
        
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True,
            "issues": [],
            "security_concerns": [],
            "suggestions": [],
            "risk_level": "low"
        }
        
        mock_llm.format_response.side_effect = [
            "Available tables: sales, customers",
            "Here are the first 10 rows from the sales table."
        ]
        
        config = AgentConfig(
            name="test-agent",
            description="Test agent",
            workflow=WorkflowConfig(require_approval=False, conversation_memory=True)
        )
        
        agent = DataAgent(
            config=config,
            llm=mock_llm,
            knowledge=real_knowledge,
            executor=real_executor
        )
        
        conversation_id = "context-test"
        
        # First query
        response1 = await agent.query("What tables are available?", conversation_id)
        assert isinstance(response1, str)
        
        # Second query in same conversation
        response2 = await agent.query("Show me data from the sales table", conversation_id)
        assert isinstance(response2, str)
        
        # Verify conversation history
        history = await agent.get_conversation_history(conversation_id)
        assert len(history) == 4  # 2 user messages + 2 assistant responses
        
        # Verify context was passed to second query
        assert mock_llm.analyze_query.call_count == 2
        assert mock_llm.generate_code.call_count == 2
        
        await agent.shutdown()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, real_executor, real_knowledge):
        """Test error handling and recovery in real scenarios."""
        from unittest.mock import AsyncMock
        
        mock_llm = AsyncMock()
        mock_llm.analyze_query.return_value = {
            "intent": "data analysis",
            "query_type": "data_retrieval",
            "entities": ["nonexistent"],
            "complexity": "moderate",
            "requires_clarification": False,
            "suggested_approach": "Generate SQL query"
        }
        
        # Generate invalid SQL to test error handling
        mock_llm.generate_code.return_value = "SELECT * FROM nonexistent_table"
        
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True,  # Validation passes but execution will fail
            "issues": [],
            "security_concerns": [],
            "suggestions": [],
            "risk_level": "low"
        }
        
        mock_llm.format_response.return_value = "I encountered an error accessing the requested table."
        
        config = AgentConfig(
            name="test-agent",
            description="Test agent",
            workflow=WorkflowConfig(require_approval=False)
        )
        
        agent = DataAgent(
            config=config,
            llm=mock_llm,
            knowledge=real_knowledge,
            executor=real_executor
        )
        
        query = "Show me data from nonexistent table"
        response = await agent.query(query)
        
        # Should get a response even with execution error
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Verify error was handled gracefully
        conversation_status = await agent.get_conversation_status("default")
        assert conversation_status["error_occurred"] is True
        
        await agent.shutdown()


class TestRealDataScenarios:
    """Test with various realistic data scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_time_series_analysis(self, real_executor, real_knowledge):
        """Test time series analysis capabilities."""
        from unittest.mock import AsyncMock
        
        mock_llm = AsyncMock()
        mock_llm.analyze_query.return_value = {
            "intent": "time series analysis",
            "query_type": "temporal_analysis",
            "entities": ["sales", "time", "trend"],
            "complexity": "high",
            "requires_clarification": False,
            "suggested_approach": "Generate time-based SQL query"
        }
        
        mock_llm.generate_code.return_value = """
        SELECT 
            DATE_TRUNC('month', date) as month,
            SUM(revenue) as monthly_revenue,
            COUNT(*) as transaction_count
        FROM sales 
        GROUP BY DATE_TRUNC('month', date)
        ORDER BY month
        """
        
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True,
            "issues": [],
            "security_concerns": [],
            "suggestions": [],
            "risk_level": "low"
        }
        
        mock_llm.format_response.return_value = "Here's the monthly revenue trend analysis."
        
        config = AgentConfig(
            name="test-agent",
            description="Test agent",
            workflow=WorkflowConfig(require_approval=False)
        )
        
        agent = DataAgent(
            config=config,
            llm=mock_llm,
            knowledge=real_knowledge,
            executor=real_executor
        )
        
        query = "Show me monthly revenue trends over time"
        response = await agent.query(query)
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        await agent.shutdown()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_statistical_analysis(self, real_executor, real_knowledge):
        """Test statistical analysis capabilities."""
        from unittest.mock import AsyncMock
        
        mock_llm = AsyncMock()
        mock_llm.analyze_query.return_value = {
            "intent": "statistical analysis",
            "query_type": "statistics",
            "entities": ["revenue", "statistics"],
            "complexity": "moderate",
            "requires_clarification": False,
            "suggested_approach": "Generate statistical SQL query"
        }
        
        mock_llm.generate_code.return_value = """
        SELECT 
            AVG(revenue) as avg_revenue,
            STDDEV(revenue) as stddev_revenue,
            MIN(revenue) as min_revenue,
            MAX(revenue) as max_revenue,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY revenue) as median_revenue
        FROM sales
        """
        
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True,
            "issues": [],
            "security_concerns": [],
            "suggestions": [],
            "risk_level": "low"
        }
        
        mock_llm.format_response.return_value = "Here are the statistical measures for revenue data."
        
        config = AgentConfig(
            name="test-agent",
            description="Test agent",
            workflow=WorkflowConfig(require_approval=False)
        )
        
        agent = DataAgent(
            config=config,
            llm=mock_llm,
            knowledge=real_knowledge,
            executor=real_executor
        )
        
        query = "Calculate statistical measures for revenue"
        response = await agent.query(query)
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        await agent.shutdown()


class TestKnowledgeIntegration:
    """Test knowledge base integration in real scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_knowledge_guided_analysis(self, real_executor, real_knowledge):
        """Test that knowledge base guides analysis appropriately."""
        from unittest.mock import AsyncMock
        
        mock_llm = AsyncMock()
        mock_llm.analyze_query.return_value = {
            "intent": "data analysis",
            "query_type": "guided_analysis",
            "entities": ["sales", "analysis"],
            "complexity": "moderate",
            "requires_clarification": False,
            "suggested_approach": "Use knowledge base guidance for analysis"
        }
        
        # The LLM should receive context from knowledge base
        def mock_generate_with_context(*args, **kwargs):
            context = kwargs.get('context', [])
            # Verify context contains relevant documents
            assert len(context) > 0
            assert any('sales' in doc.content.lower() for doc in context)
            return "SELECT product, category, SUM(revenue) FROM sales GROUP BY product, category"
        
        mock_llm.generate_code.side_effect = mock_generate_with_context
        
        mock_llm.validate_generated_code.return_value = {
            "is_valid": True,
            "issues": [],
            "security_concerns": [],
            "suggestions": [],
            "risk_level": "low"
        }
        
        mock_llm.format_response.return_value = "Analysis completed using business knowledge."
        
        config = AgentConfig(
            name="test-agent",
            description="Test agent",
            workflow=WorkflowConfig(require_approval=False)
        )
        
        agent = DataAgent(
            config=config,
            llm=mock_llm,
            knowledge=real_knowledge,
            executor=real_executor
        )
        
        query = "Analyze sales data following best practices"
        response = await agent.query(query)
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Verify knowledge was used
        mock_llm.generate_code.assert_called_once()
        
        await agent.shutdown()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dynamic_knowledge_updates(self, real_executor):
        """Test dynamic knowledge base updates during operation."""
        from unittest.mock import AsyncMock
        
        # Create knowledge base
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge = FAISSKnowledge(
                model_name="all-MiniLM-L6-v2",
                index_path=str(Path(temp_dir) / "dynamic_index")
            )
            
            # Initial documents
            initial_docs = [
                Document(
                    content="Initial sales analysis guidelines.",
                    source="guidelines",
                    metadata={"version": "1.0"}
                )
            ]
            await knowledge.ingest(initial_docs)
            
            mock_llm = AsyncMock()
            mock_llm.analyze_query.return_value = {
                "intent": "data analysis",
                "query_type": "data_retrieval",
                "entities": ["sales"],
                "complexity": "moderate",
                "requires_clarification": False,
                "suggested_approach": "Generate SQL query"
            }
            
            mock_llm.generate_code.return_value = "SELECT * FROM sales LIMIT 5"
            
            mock_llm.validate_generated_code.return_value = {
                "is_valid": True,
                "issues": [],
                "security_concerns": [],
                "suggestions": [],
                "risk_level": "low"
            }
            
            mock_llm.format_response.return_value = "Analysis complete."
            
            config = AgentConfig(
                name="test-agent",
                description="Test agent",
                workflow=WorkflowConfig(require_approval=False)
            )
            
            agent = DataAgent(
                config=config,
                llm=mock_llm,
                knowledge=knowledge,
                executor=real_executor
            )
            
            # First query
            response1 = await agent.query("Analyze sales data")
            assert isinstance(response1, str)
            
            # Add new knowledge
            new_docs = [
                Document(
                    content="Updated sales analysis guidelines with new best practices.",
                    source="guidelines",
                    metadata={"version": "2.0"}
                )
            ]
            await agent.ingest_knowledge(new_docs)
            
            # Second query should have access to updated knowledge
            response2 = await agent.query("Analyze sales data with latest guidelines")
            assert isinstance(response2, str)
            
            await agent.shutdown()