"""
Integration tests for the Python API interface.

These tests verify that the API works end-to-end with real components
(though still using mocks for external services like LLMs).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import dataqa
from dataqa import (
    DataQAClient,
    create_agent,
    create_agent_async,
    agent_session,
    quick_query,
    quick_query_async,
)


class TestAPIIntegration:
    """Integration tests for the Python API."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "name": "integration-test-agent",
            "description": "Agent for integration testing",
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key",
                "temperature": 0.1
            },
            "knowledge": {
                "provider": "faiss",
                "embedding_model": "all-MiniLM-L6-v2"
            },
            "executor": {
                "provider": "inmemory",
                "database_type": "duckdb"
            },
            "workflow": {
                "require_approval": False,
                "enable_visualization": True
            }
        }
    
    def test_import_all_api_components(self):
        """Test that all API components can be imported."""
        # Test main classes
        assert hasattr(dataqa, 'DataAgent')
        assert hasattr(dataqa, 'DataQAClient')
        assert hasattr(dataqa, 'AgentConfig')
        assert hasattr(dataqa, 'Document')
        assert hasattr(dataqa, 'Message')
        
        # Test factory functions
        assert hasattr(dataqa, 'create_agent')
        assert hasattr(dataqa, 'create_agent_async')
        
        # Test context managers
        assert hasattr(dataqa, 'agent_session')
        
        # Test convenience functions
        assert hasattr(dataqa, 'quick_query')
        assert hasattr(dataqa, 'quick_query_async')
    
    @patch('src.dataqa.primitives.llm.create_llm_interface')
    @patch('src.dataqa.primitives.faiss_knowledge.FAISSKnowledge')
    @patch('src.dataqa.primitives.in_memory_executor.InMemoryExecutor')
    def test_create_agent_factory_function(self, mock_executor, mock_knowledge, mock_llm, sample_config):
        """Test the create_agent factory function."""
        # Mock the components
        mock_llm_instance = MagicMock()
        mock_llm_instance.get_model_info = AsyncMock(return_value={"model": "gpt-4"})
        mock_llm.return_value = mock_llm_instance
        
        mock_knowledge_instance = MagicMock()
        mock_knowledge_instance.get_stats = AsyncMock(return_value={"documents": 0})
        mock_knowledge.return_value = mock_knowledge_instance
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.list_tables = AsyncMock(return_value=[])
        mock_executor.return_value = mock_executor_instance
        
        # Create agent using factory function
        agent = create_agent("test-agent", config=sample_config)
        
        # Verify agent was created
        assert agent is not None
        assert agent.config.name == "integration-test-agent"
        
        # Verify components were initialized
        mock_llm.assert_called_once()
        mock_knowledge.assert_called_once()
        mock_executor.assert_called_once()
    
    @patch('src.dataqa.primitives.llm.create_llm_interface')
    @patch('src.dataqa.primitives.faiss_knowledge.FAISSKnowledge')
    @patch('src.dataqa.primitives.in_memory_executor.InMemoryExecutor')
    @pytest.mark.asyncio
    async def test_create_agent_async_factory_function(self, mock_executor, mock_knowledge, mock_llm, sample_config):
        """Test the create_agent_async factory function."""
        # Mock the components
        mock_llm_instance = MagicMock()
        mock_llm_instance.get_model_info = AsyncMock(return_value={"model": "gpt-4"})
        mock_llm.return_value = mock_llm_instance
        
        mock_knowledge_instance = MagicMock()
        mock_knowledge_instance.get_stats = AsyncMock(return_value={"documents": 0})
        mock_knowledge.return_value = mock_knowledge_instance
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.list_tables = AsyncMock(return_value=[])
        mock_executor.return_value = mock_executor_instance
        
        # Create agent using async factory function
        agent = await create_agent_async("test-agent", config=sample_config)
        
        # Verify agent was created
        assert agent is not None
        assert agent.config.name == "integration-test-agent"
        
        # Verify components were initialized
        mock_llm.assert_called_once()
        mock_knowledge.assert_called_once()
        mock_executor.assert_called_once()
    
    @patch('src.dataqa.primitives.llm.create_llm_interface')
    @patch('src.dataqa.primitives.faiss_knowledge.FAISSKnowledge')
    @patch('src.dataqa.primitives.in_memory_executor.InMemoryExecutor')
    @pytest.mark.asyncio
    async def test_agent_session_context_manager(self, mock_executor, mock_knowledge, mock_llm, sample_config):
        """Test the agent_session context manager."""
        # Mock the components
        mock_llm_instance = MagicMock()
        mock_llm_instance.get_model_info = AsyncMock(return_value={"model": "gpt-4"})
        mock_llm.return_value = mock_llm_instance
        
        mock_knowledge_instance = MagicMock()
        mock_knowledge_instance.get_stats = AsyncMock(return_value={"documents": 0})
        mock_knowledge.return_value = mock_knowledge_instance
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.list_tables = AsyncMock(return_value=[])
        mock_executor.return_value = mock_executor_instance
        
        # Use agent session context manager
        async with agent_session("test-agent", config=sample_config) as agent:
            assert agent is not None
            assert agent.config.name == "integration-test-agent"
            
            # Mock a query to test functionality
            with patch.object(agent, 'query', return_value="Test response") as mock_query:
                response = await agent.query("Test query")
                assert response == "Test response"
                mock_query.assert_called_once_with("Test query")
        
        # Verify components were initialized
        mock_llm.assert_called_once()
        mock_knowledge.assert_called_once()
        mock_executor.assert_called_once()
    
    @patch('src.dataqa.primitives.llm.create_llm_interface')
    @patch('src.dataqa.primitives.faiss_knowledge.FAISSKnowledge')
    @patch('src.dataqa.primitives.in_memory_executor.InMemoryExecutor')
    @pytest.mark.asyncio
    async def test_client_multiple_agents(self, mock_executor, mock_knowledge, mock_llm, sample_config):
        """Test DataQAClient with multiple agents."""
        # Mock the components
        mock_llm_instance = MagicMock()
        mock_llm_instance.get_model_info = AsyncMock(return_value={"model": "gpt-4"})
        mock_llm.return_value = mock_llm_instance
        
        mock_knowledge_instance = MagicMock()
        mock_knowledge_instance.get_stats = AsyncMock(return_value={"documents": 0})
        mock_knowledge.return_value = mock_knowledge_instance
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.list_tables = AsyncMock(return_value=[])
        mock_executor.return_value = mock_executor_instance
        
        async with DataQAClient() as client:
            # Create multiple agents
            agent1 = await client.create_agent_async("agent1", config=sample_config)
            agent2 = await client.create_agent_async("agent2", config=sample_config)
            
            # Verify both agents exist
            assert client.get_agent("agent1") == agent1
            assert client.get_agent("agent2") == agent2
            assert set(client.list_agents()) == {"agent1", "agent2"}
            
            # Mock queries for both agents
            with patch.object(agent1, 'query', return_value="Response 1") as mock_query1:
                with patch.object(agent2, 'query', return_value="Response 2") as mock_query2:
                    response1 = await client.query_async("agent1", "Query 1")
                    response2 = await client.query_async("agent2", "Query 2")
                    
                    assert response1 == "Response 1"
                    assert response2 == "Response 2"
                    
                    mock_query1.assert_called_once_with("Query 1", None)
                    mock_query2.assert_called_once_with("Query 2", None)
    
    @patch('src.dataqa.primitives.llm.create_llm_interface')
    @patch('src.dataqa.primitives.faiss_knowledge.FAISSKnowledge')
    @patch('src.dataqa.primitives.in_memory_executor.InMemoryExecutor')
    def test_quick_query_convenience_function(self, mock_executor, mock_knowledge, mock_llm, sample_config):
        """Test the quick_query convenience function."""
        # Mock the components
        mock_llm_instance = MagicMock()
        mock_llm_instance.get_model_info = AsyncMock(return_value={"model": "gpt-4"})
        mock_llm.return_value = mock_llm_instance
        
        mock_knowledge_instance = MagicMock()
        mock_knowledge_instance.get_stats = AsyncMock(return_value={"documents": 0})
        mock_knowledge.return_value = mock_knowledge_instance
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.list_tables = AsyncMock(return_value=[])
        mock_executor.return_value = mock_executor_instance
        
        # Mock the agent query method
        with patch('src.dataqa.agent.agent.DataAgent.query', return_value="Quick response") as mock_query:
            response = quick_query("Test quick query", **sample_config)
            
            assert response == "Quick response"
            mock_query.assert_called_once()
    
    @patch('src.dataqa.primitives.llm.create_llm_interface')
    @patch('src.dataqa.primitives.faiss_knowledge.FAISSKnowledge')
    @patch('src.dataqa.primitives.in_memory_executor.InMemoryExecutor')
    @pytest.mark.asyncio
    async def test_quick_query_async_convenience_function(self, mock_executor, mock_knowledge, mock_llm, sample_config):
        """Test the quick_query_async convenience function."""
        # Mock the components
        mock_llm_instance = MagicMock()
        mock_llm_instance.get_model_info = AsyncMock(return_value={"model": "gpt-4"})
        mock_llm.return_value = mock_llm_instance
        
        mock_knowledge_instance = MagicMock()
        mock_knowledge_instance.get_stats = AsyncMock(return_value={"documents": 0})
        mock_knowledge.return_value = mock_knowledge_instance
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.list_tables = AsyncMock(return_value=[])
        mock_executor.return_value = mock_executor_instance
        
        # Mock the agent query method
        with patch('src.dataqa.agent.agent.DataAgent.query', return_value="Quick async response") as mock_query:
            response = await quick_query_async("Test quick async query", **sample_config)
            
            assert response == "Quick async response"
            mock_query.assert_called_once()
    
    def test_api_configuration_validation(self):
        """Test that API properly validates configurations."""
        # Test invalid configuration
        invalid_config = {
            "name": "test-agent",
            "llm": {
                "provider": "invalid_provider"  # Invalid provider
            }
        }
        
        with pytest.raises(Exception):  # Should raise validation error
            create_agent("test-agent", config=invalid_config)
    
    @patch('src.dataqa.primitives.llm.create_llm_interface')
    @patch('src.dataqa.primitives.faiss_knowledge.FAISSKnowledge')
    @patch('src.dataqa.primitives.in_memory_executor.InMemoryExecutor')
    def test_api_error_handling(self, mock_executor, mock_knowledge, mock_llm, sample_config):
        """Test API error handling."""
        # Mock component that raises an error
        mock_llm.side_effect = Exception("LLM initialization failed")
        
        with pytest.raises(Exception, match="LLM initialization failed"):
            create_agent("test-agent", config=sample_config)
    
    @patch('src.dataqa.primitives.llm.create_llm_interface')
    @patch('src.dataqa.primitives.faiss_knowledge.FAISSKnowledge')
    @patch('src.dataqa.primitives.in_memory_executor.InMemoryExecutor')
    @pytest.mark.asyncio
    async def test_api_resource_cleanup(self, mock_executor, mock_knowledge, mock_llm, sample_config):
        """Test that API properly cleans up resources."""
        # Mock the components
        mock_llm_instance = MagicMock()
        mock_llm_instance.get_model_info = AsyncMock(return_value={"model": "gpt-4"})
        mock_llm.return_value = mock_llm_instance
        
        mock_knowledge_instance = MagicMock()
        mock_knowledge_instance.get_stats = AsyncMock(return_value={"documents": 0})
        mock_knowledge.return_value = mock_knowledge_instance
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.list_tables = AsyncMock(return_value=[])
        mock_executor.return_value = mock_executor_instance
        
        # Test context manager cleanup
        async with DataQAClient() as client:
            agent = await client.create_agent_async("test-agent", config=sample_config)
            
            # Mock the shutdown method
            with patch.object(agent, 'shutdown') as mock_shutdown:
                pass  # Context manager will call shutdown on exit
        
        # Verify shutdown was called
        mock_shutdown.assert_called_once()


class TestAPIDocumentation:
    """Test that API provides good documentation and introspection."""
    
    def test_api_docstrings(self):
        """Test that main API components have docstrings."""
        assert DataQAClient.__doc__ is not None
        assert create_agent.__doc__ is not None
        assert create_agent_async.__doc__ is not None
        assert agent_session.__doc__ is not None
        assert quick_query.__doc__ is not None
        assert quick_query_async.__doc__ is not None
    
    def test_api_type_hints(self):
        """Test that API functions have proper type hints."""
        import inspect
        
        # Check create_agent signature
        sig = inspect.signature(create_agent)
        assert 'name' in sig.parameters
        assert 'config' in sig.parameters
        assert 'config_path' in sig.parameters
        
        # Check create_agent_async signature
        sig_async = inspect.signature(create_agent_async)
        assert 'name' in sig_async.parameters
        assert 'config' in sig_async.parameters
        assert 'config_path' in sig_async.parameters
    
    def test_api_module_exports(self):
        """Test that API module exports are properly defined."""
        import dataqa.api as api_module
        
        # Check __all__ is defined
        assert hasattr(api_module, '__all__')
        assert isinstance(api_module.__all__, list)
        
        # Check all exported items exist
        for item in api_module.__all__:
            assert hasattr(api_module, item)