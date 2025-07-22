"""
Tests for the high-level Python API interface.

This module tests the DataQAClient, factory functions, context managers,
and convenience functions provided by the API module.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from src.dataqa.api import (
    DataQAClient,
    create_agent,
    create_agent_async,
    agent_session,
    quick_query,
    quick_query_async,
)
from src.dataqa.agent.agent import DataAgent
from src.dataqa.config.models import AgentConfig
from src.dataqa.models.document import Document
from src.dataqa.models.message import Message


class TestDataQAClient:
    """Test cases for DataQAClient class."""
    
    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Sample agent configuration."""
        return {
            "name": "test-agent",
            "description": "Test agent for API testing",
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key"
            },
            "knowledge": {
                "provider": "faiss"
            },
            "executor": {
                "provider": "inmemory"
            }
        }
    
    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Mock DataAgent instance."""
        agent = MagicMock(spec=DataAgent)
        agent.query = AsyncMock(return_value="Test response")
        agent.approve_operation = AsyncMock(return_value="Operation approved")
        agent.ingest_knowledge = AsyncMock()
        agent.get_conversation_history = AsyncMock(return_value=[])
        agent.health_check = AsyncMock(return_value={"status": "healthy"})
        agent.shutdown = AsyncMock()
        return agent
    
    def test_client_init(self):
        """Test client initialization."""
        client = DataQAClient()
        assert client.default_config_dir == Path("config")
        assert client._agents == {}
        assert client._event_loop is None
        
        # Test with custom config dir
        custom_dir = Path("/custom/config")
        client = DataQAClient(default_config_dir=custom_dir)
        assert client.default_config_dir == custom_dir
    
    def test_context_manager_sync(self):
        """Test synchronous context manager."""
        with patch.object(DataQAClient, 'shutdown') as mock_shutdown:
            with DataQAClient() as client:
                assert isinstance(client, DataQAClient)
            mock_shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager_async(self):
        """Test asynchronous context manager."""
        with patch.object(DataQAClient, 'shutdown_async') as mock_shutdown:
            async with DataQAClient() as client:
                assert isinstance(client, DataQAClient)
            mock_shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_agent_async(self, sample_config, mock_agent):
        """Test asynchronous agent creation."""
        client = DataQAClient()
        
        with patch('src.dataqa.api.create_agent_from_config', return_value=mock_agent):
            agent = await client.create_agent_async("test-agent", config=sample_config)
            
            assert agent == mock_agent
            assert "test-agent" in client._agents
            assert client._agents["test-agent"] == mock_agent
    
    def test_create_agent_sync(self, sample_config, mock_agent):
        """Test synchronous agent creation."""
        client = DataQAClient()
        
        with patch('src.dataqa.api.create_agent_from_config', return_value=mock_agent):
            with patch.object(client, 'create_agent_async', return_value=mock_agent) as mock_async:
                agent = client.create_agent("test-agent", config=sample_config)
                
                assert agent == mock_agent
                mock_async.assert_called_once_with("test-agent", sample_config, None)
    
    @pytest.mark.asyncio
    async def test_create_agent_with_config_path(self, mock_agent):
        """Test agent creation with config file path."""
        client = DataQAClient()
        
        mock_config = AgentConfig(name="test-agent")
        
        with patch('src.dataqa.api.load_agent_config', return_value=mock_config):
            with patch('src.dataqa.api.create_agent_from_config', return_value=mock_agent):
                with patch('pathlib.Path.exists', return_value=True):
                    agent = await client.create_agent_async("test-agent", config_path="test.yaml")
                    
                    assert agent == mock_agent
    
    @pytest.mark.asyncio
    async def test_create_agent_invalid_config(self):
        """Test agent creation with invalid configuration."""
        client = DataQAClient()
        
        with pytest.raises(ValueError, match="Either config or config_path must be provided"):
            await client.create_agent_async("test-agent")
    
    def test_get_agent(self, mock_agent):
        """Test getting existing agent."""
        client = DataQAClient()
        client._agents["test-agent"] = mock_agent
        
        assert client.get_agent("test-agent") == mock_agent
        assert client.get_agent("nonexistent") is None
    
    def test_list_agents(self, mock_agent):
        """Test listing agents."""
        client = DataQAClient()
        client._agents["agent1"] = mock_agent
        client._agents["agent2"] = mock_agent
        
        agents = client.list_agents()
        assert set(agents) == {"agent1", "agent2"}
    
    @pytest.mark.asyncio
    async def test_query_async(self, mock_agent):
        """Test asynchronous querying."""
        client = DataQAClient()
        client._agents["test-agent"] = mock_agent
        
        response = await client.query_async("test-agent", "Test query")
        
        assert response == "Test response"
        mock_agent.query.assert_called_once_with("Test query", None)
    
    @pytest.mark.asyncio
    async def test_query_async_with_agent_instance(self, mock_agent):
        """Test async querying with agent instance."""
        client = DataQAClient()
        
        response = await client.query_async(mock_agent, "Test query", "conv-123")
        
        assert response == "Test response"
        mock_agent.query.assert_called_once_with("Test query", "conv-123")
    
    @pytest.mark.asyncio
    async def test_query_async_agent_not_found(self):
        """Test async querying with nonexistent agent."""
        client = DataQAClient()
        
        with pytest.raises(ValueError, match="Agent not found: nonexistent"):
            await client.query_async("nonexistent", "Test query")
    
    def test_query_sync(self, mock_agent):
        """Test synchronous querying."""
        client = DataQAClient()
        
        with patch.object(client, 'query_async', return_value="Test response") as mock_async:
            response = client.query("test-agent", "Test query")
            
            assert response == "Test response"
            mock_async.assert_called_once_with("test-agent", "Test query", None)
    
    @pytest.mark.asyncio
    async def test_approve_operation_async(self, mock_agent):
        """Test asynchronous operation approval."""
        client = DataQAClient()
        client._agents["test-agent"] = mock_agent
        
        response = await client.approve_operation_async("test-agent", "conv-123", True, "Approved")
        
        assert response == "Operation approved"
        mock_agent.approve_operation.assert_called_once_with("conv-123", True, "Approved")
    
    @pytest.mark.asyncio
    async def test_ingest_knowledge_async(self, mock_agent):
        """Test asynchronous knowledge ingestion."""
        client = DataQAClient()
        client._agents["test-agent"] = mock_agent
        
        documents = [Document(content="Test doc", metadata={}, source="test")]
        
        await client.ingest_knowledge_async("test-agent", documents)
        
        mock_agent.ingest_knowledge.assert_called_once_with(documents)
    
    @pytest.mark.asyncio
    async def test_get_conversation_history_async(self, mock_agent):
        """Test getting conversation history."""
        client = DataQAClient()
        client._agents["test-agent"] = mock_agent
        
        history = await client.get_conversation_history_async("test-agent", "conv-123")
        
        assert history == []
        mock_agent.get_conversation_history.assert_called_once_with("conv-123")
    
    @pytest.mark.asyncio
    async def test_health_check_async(self, mock_agent):
        """Test health check."""
        client = DataQAClient()
        client._agents["test-agent"] = mock_agent
        
        health = await client.health_check_async("test-agent")
        
        assert health == {"status": "healthy"}
        mock_agent.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_async(self, mock_agent):
        """Test asynchronous shutdown."""
        client = DataQAClient()
        client._agents["agent1"] = mock_agent
        client._agents["agent2"] = mock_agent
        
        await client.shutdown_async()
        
        assert client._agents == {}
        assert mock_agent.shutdown.call_count == 2
    
    def test_shutdown_sync(self):
        """Test synchronous shutdown."""
        client = DataQAClient()
        
        with patch.object(client, 'shutdown_async') as mock_async:
            client.shutdown()
            mock_async.assert_called_once()


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Sample agent configuration."""
        return {
            "name": "test-agent",
            "llm": {"provider": "openai", "model": "gpt-4"},
            "knowledge": {"provider": "faiss"},
            "executor": {"provider": "inmemory"}
        }
    
    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Mock DataAgent instance."""
        return MagicMock(spec=DataAgent)
    
    def test_create_agent(self, sample_config, mock_agent):
        """Test create_agent factory function."""
        with patch.object(DataQAClient, 'create_agent', return_value=mock_agent) as mock_create:
            agent = create_agent("test-agent", config=sample_config)
            
            assert agent == mock_agent
            mock_create.assert_called_once_with("test-agent", sample_config, None)
    
    @pytest.mark.asyncio
    async def test_create_agent_async(self, sample_config, mock_agent):
        """Test create_agent_async factory function."""
        with patch.object(DataQAClient, 'create_agent_async', return_value=mock_agent) as mock_create:
            agent = await create_agent_async("test-agent", config=sample_config)
            
            assert agent == mock_agent
            mock_create.assert_called_once_with("test-agent", sample_config, None)


class TestContextManagers:
    """Test cases for context managers."""
    
    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Mock DataAgent instance."""
        agent = MagicMock(spec=DataAgent)
        agent.shutdown = AsyncMock()
        return agent
    
    @pytest.mark.asyncio
    async def test_agent_session(self, mock_agent):
        """Test agent_session context manager."""
        with patch.object(DataQAClient, 'create_agent_async', return_value=mock_agent):
            async with agent_session("test-agent", config={"name": "test"}) as agent:
                assert agent == mock_agent
            
            # Verify agent was shut down
            mock_agent.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_agent_session_with_exception(self, mock_agent):
        """Test agent_session context manager with exception."""
        with patch.object(DataQAClient, 'create_agent_async', return_value=mock_agent):
            with pytest.raises(ValueError):
                async with agent_session("test-agent", config={"name": "test"}) as agent:
                    raise ValueError("Test error")
            
            # Verify agent was still shut down
            mock_agent.shutdown.assert_called_once()


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_quick_query(self):
        """Test quick_query convenience function."""
        with patch.object(DataQAClient, 'create_agent') as mock_create:
            with patch.object(DataQAClient, 'query', return_value="Quick response") as mock_query:
                response = quick_query("Test query", config_path="test.yaml")
                
                assert response == "Quick response"
                mock_create.assert_called_once()
                mock_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_quick_query_async(self):
        """Test quick_query_async convenience function."""
        mock_agent = MagicMock(spec=DataAgent)
        mock_agent.query = AsyncMock(return_value="Quick async response")
        mock_agent.shutdown = AsyncMock()
        
        with patch.object(DataQAClient, 'create_agent_async', return_value=mock_agent):
            response = await quick_query_async("Test query", config_path="test.yaml")
            
            assert response == "Quick async response"
            mock_agent.query.assert_called_once_with("Test query")
            mock_agent.shutdown.assert_called_once()


class TestConfigurationHandling:
    """Test cases for configuration handling."""
    
    @pytest.mark.asyncio
    async def test_load_config_from_dict(self):
        """Test loading configuration from dictionary."""
        client = DataQAClient()
        config_dict = {
            "name": "test-agent",
            "llm": {"provider": "openai", "model": "gpt-4"},
            "knowledge": {"provider": "faiss"},
            "executor": {"provider": "inmemory"}
        }
        
        config = await client._load_config(config_dict, None, "test-agent")
        
        assert isinstance(config, AgentConfig)
        assert config.name == "test-agent"
        assert config.llm.provider.value == "openai"
    
    @pytest.mark.asyncio
    async def test_load_config_from_agent_config(self):
        """Test loading configuration from AgentConfig instance."""
        client = DataQAClient()
        original_config = AgentConfig(name="test-agent")
        
        config = await client._load_config(original_config, None, "test-agent")
        
        assert config == original_config
    
    @pytest.mark.asyncio
    async def test_load_config_with_kwargs(self):
        """Test loading configuration with additional kwargs."""
        client = DataQAClient()
        config_dict = {"name": "test-agent"}
        
        config = await client._load_config(
            config_dict, 
            None, 
            "test-agent",
            description="Updated description"
        )
        
        assert config.description == "Updated description"
    
    @pytest.mark.asyncio
    async def test_load_config_from_yaml_string(self):
        """Test loading configuration from YAML string."""
        client = DataQAClient()
        yaml_config = """
        name: test-agent
        description: Test agent from YAML
        llm:
          provider: openai
          model: gpt-4
        """
        
        with patch('yaml.safe_load') as mock_yaml:
            mock_yaml.return_value = {
                "name": "test-agent",
                "description": "Test agent from YAML",
                "llm": {"provider": "openai", "model": "gpt-4"}
            }
            
            config = await client._load_config(yaml_config, None, "test-agent")
            
            assert config.name == "test-agent"
            assert config.description == "Test agent from YAML"
    
    @pytest.mark.asyncio
    async def test_load_config_file_not_found(self):
        """Test loading configuration from nonexistent file."""
        client = DataQAClient()
        
        with pytest.raises(FileNotFoundError):
            await client._load_config(None, "nonexistent.yaml", "test-agent")
    
    @pytest.mark.asyncio
    async def test_load_config_no_config_provided(self):
        """Test error when no configuration is provided."""
        client = DataQAClient()
        
        with pytest.raises(ValueError, match="Either config or config_path must be provided"):
            await client._load_config(None, None, "test-agent")


class TestErrorHandling:
    """Test cases for error handling."""
    
    @pytest.mark.asyncio
    async def test_query_with_agent_error(self):
        """Test querying when agent raises an error."""
        client = DataQAClient()
        mock_agent = MagicMock(spec=DataAgent)
        mock_agent.query = AsyncMock(side_effect=Exception("Agent error"))
        client._agents["test-agent"] = mock_agent
        
        with pytest.raises(Exception, match="Agent error"):
            await client.query_async("test-agent", "Test query")
    
    @pytest.mark.asyncio
    async def test_shutdown_with_agent_error(self):
        """Test shutdown when agent raises an error."""
        client = DataQAClient()
        mock_agent = MagicMock(spec=DataAgent)
        mock_agent.shutdown = AsyncMock(side_effect=Exception("Shutdown error"))
        client._agents["test-agent"] = mock_agent
        
        # Should not raise exception, just log error
        await client.shutdown_async()
        
        # Agents should still be cleared
        assert client._agents == {}


class TestIntegration:
    """Integration test cases."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_async(self):
        """Test complete async workflow."""
        mock_agent = MagicMock(spec=DataAgent)
        mock_agent.query = AsyncMock(return_value="Integration test response")
        mock_agent.shutdown = AsyncMock()
        
        config = {
            "name": "integration-agent",
            "llm": {"provider": "openai", "model": "gpt-4"},
            "knowledge": {"provider": "faiss"},
            "executor": {"provider": "inmemory"}
        }
        
        with patch('src.dataqa.api.create_agent_from_config', return_value=mock_agent):
            async with DataQAClient() as client:
                # Create agent
                agent = await client.create_agent_async("integration-agent", config=config)
                assert agent == mock_agent
                
                # Query agent
                response = await client.query_async("integration-agent", "Test integration query")
                assert response == "Integration test response"
                
                # Check agent is tracked
                assert "integration-agent" in client.list_agents()
        
        # Verify cleanup
        mock_agent.shutdown.assert_called_once()
    
    def test_full_workflow_sync(self):
        """Test complete sync workflow."""
        mock_agent = MagicMock(spec=DataAgent)
        
        config = {
            "name": "sync-agent",
            "llm": {"provider": "openai", "model": "gpt-4"},
            "knowledge": {"provider": "faiss"},
            "executor": {"provider": "inmemory"}
        }
        
        with patch('src.dataqa.api.create_agent_from_config', return_value=mock_agent):
            with patch.object(DataQAClient, 'create_agent_async', return_value=mock_agent):
                with patch.object(DataQAClient, 'query_async', return_value="Sync response"):
                    with DataQAClient() as client:
                        # Create agent
                        agent = client.create_agent("sync-agent", config=config)
                        assert agent == mock_agent
                        
                        # Query agent
                        response = client.query("sync-agent", "Test sync query")
                        assert response == "Sync response"