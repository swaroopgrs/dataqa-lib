"""
Unit tests for configuration models.

Tests Pydantic model validation, environment variable resolution,
and configuration consistency checks.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from dataqa.config.models import (
    AgentConfig,
    ExecutorConfig,
    ExecutorProvider,
    KnowledgeConfig,
    KnowledgeProvider,
    LLMConfig,
    LLMProvider,
    WorkflowConfig,
)


class TestLLMConfig:
    """Test LLM configuration model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4"
        assert config.temperature == 0.1
        assert config.timeout == 30.0
        assert config.max_retries == 3

    def test_valid_configuration(self):
        """Test valid configuration creation."""
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-sonnet",
            api_key="test-key",
            temperature=0.5,
            max_tokens=1000
        )
        assert config.provider == LLMProvider.ANTHROPIC
        assert config.model == "claude-3-sonnet"
        assert config.api_key.get_secret_value() == "test-key"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000

    def test_temperature_validation(self):
        """Test temperature range validation."""
        # Valid temperatures
        LLMConfig(temperature=0.0)
        LLMConfig(temperature=1.0)
        LLMConfig(temperature=2.0)
        
        # Invalid temperatures
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.1)

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid values
        LLMConfig(max_tokens=100)
        LLMConfig(max_tokens=None)
        
        # Invalid values
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=0)
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=-1)

    @patch.dict(os.environ, {"TEST_API_KEY": "secret-key"})
    def test_env_var_resolution(self):
        """Test environment variable resolution for API key."""
        config = LLMConfig(api_key="${TEST_API_KEY}")
        assert config.api_key.get_secret_value() == "secret-key"

    def test_env_var_missing(self):
        """Test handling of missing environment variables."""
        config = LLMConfig(api_key="${MISSING_VAR}")
        assert config.api_key is None


class TestKnowledgeConfig:
    """Test knowledge configuration model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = KnowledgeConfig()
        assert config.provider == KnowledgeProvider.FAISS
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.top_k == 5
        assert config.similarity_threshold == 0.7

    def test_valid_configuration(self):
        """Test valid configuration creation."""
        config = KnowledgeConfig(
            provider=KnowledgeProvider.OPENSEARCH,
            embedding_model="custom-model",
            chunk_size=256,
            opensearch_host="localhost",
            opensearch_port=9200
        )
        assert config.provider == KnowledgeProvider.OPENSEARCH
        assert config.embedding_model == "custom-model"
        assert config.chunk_size == 256
        assert config.opensearch_host == "localhost"
        assert config.opensearch_port == 9200

    def test_chunk_size_validation(self):
        """Test chunk size validation."""
        # Valid values
        KnowledgeConfig(chunk_size=1)
        KnowledgeConfig(chunk_size=1000)
        
        # Invalid values
        with pytest.raises(ValidationError):
            KnowledgeConfig(chunk_size=0)
        with pytest.raises(ValidationError):
            KnowledgeConfig(chunk_size=-1)

    def test_similarity_threshold_validation(self):
        """Test similarity threshold validation."""
        # Valid values
        KnowledgeConfig(similarity_threshold=0.0)
        KnowledgeConfig(similarity_threshold=0.5)
        KnowledgeConfig(similarity_threshold=1.0)
        
        # Invalid values
        with pytest.raises(ValidationError):
            KnowledgeConfig(similarity_threshold=-0.1)
        with pytest.raises(ValidationError):
            KnowledgeConfig(similarity_threshold=1.1)

    @patch.dict(os.environ, {"OPENSEARCH_PASS": "secret-pass"})
    def test_opensearch_password_resolution(self):
        """Test OpenSearch password environment variable resolution."""
        config = KnowledgeConfig(opensearch_password="${OPENSEARCH_PASS}")
        assert config.opensearch_password.get_secret_value() == "secret-pass"


class TestExecutorConfig:
    """Test executor configuration model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ExecutorConfig()
        assert config.provider == ExecutorProvider.INMEMORY
        assert config.database_type == "duckdb"
        assert config.max_execution_time == 30.0
        assert config.max_memory_mb == 512
        assert config.max_rows == 10000
        assert config.allow_file_access is False
        assert "pandas" in config.allowed_imports
        assert "exec" in config.blocked_functions

    def test_valid_configuration(self):
        """Test valid configuration creation."""
        config = ExecutorConfig(
            provider=ExecutorProvider.API,
            database_type="postgresql",
            max_execution_time=60.0,
            api_url="https://api.example.com"
        )
        assert config.provider == ExecutorProvider.API
        assert config.database_type == "postgresql"
        assert config.max_execution_time == 60.0
        assert config.api_url == "https://api.example.com"

    def test_execution_limits_validation(self):
        """Test execution limits validation."""
        # Valid values
        ExecutorConfig(max_execution_time=1.0, max_memory_mb=1, max_rows=1)
        
        # Invalid values
        with pytest.raises(ValidationError):
            ExecutorConfig(max_execution_time=0)
        with pytest.raises(ValidationError):
            ExecutorConfig(max_memory_mb=0)
        with pytest.raises(ValidationError):
            ExecutorConfig(max_rows=0)

    @patch.dict(os.environ, {"DB_URL": "postgresql://user:pass@host/db"})
    def test_database_url_resolution(self):
        """Test database URL environment variable resolution."""
        config = ExecutorConfig(database_url="${DB_URL}")
        assert config.database_url.get_secret_value() == "postgresql://user:pass@host/db"


class TestWorkflowConfig:
    """Test workflow configuration model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WorkflowConfig()
        assert config.strategy == "react"
        assert config.max_iterations == 10
        assert config.require_approval is True
        assert config.auto_approve_safe is False
        assert config.conversation_memory is True
        assert config.max_context_length == 4000
        assert config.enable_visualization is True
        assert config.debug_mode is False

    def test_valid_configuration(self):
        """Test valid configuration creation."""
        config = WorkflowConfig(
            strategy="plan_execute",
            max_iterations=5,
            require_approval=False,
            debug_mode=True
        )
        assert config.strategy == "plan_execute"
        assert config.max_iterations == 5
        assert config.require_approval is False
        assert config.debug_mode is True

    def test_max_iterations_validation(self):
        """Test max iterations validation."""
        # Valid values
        WorkflowConfig(max_iterations=1)
        WorkflowConfig(max_iterations=100)
        
        # Invalid values
        with pytest.raises(ValidationError):
            WorkflowConfig(max_iterations=0)
        with pytest.raises(ValidationError):
            WorkflowConfig(max_iterations=-1)


class TestAgentConfig:
    """Test main agent configuration model."""

    def test_minimal_configuration(self):
        """Test minimal valid configuration."""
        config = AgentConfig(name="test-agent")
        assert config.name == "test-agent"
        assert config.version == "1.0.0"
        assert config.log_level == "INFO"
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.knowledge, KnowledgeConfig)
        assert isinstance(config.executor, ExecutorConfig)
        assert isinstance(config.workflow, WorkflowConfig)

    def test_complete_configuration(self):
        """Test complete configuration with all components."""
        config = AgentConfig(
            name="complete-agent",
            description="A complete test agent",
            version="2.0.0",
            llm={
                "provider": "anthropic",
                "model": "claude-3-sonnet",
                "temperature": 0.5
            },
            knowledge={
                "provider": "opensearch",
                "chunk_size": 256
            },
            executor={
                "provider": "api",
                "max_execution_time": 60.0
            },
            workflow={
                "strategy": "plan_execute",
                "max_iterations": 5
            },
            log_level="DEBUG"
        )
        
        assert config.name == "complete-agent"
        assert config.description == "A complete test agent"
        assert config.version == "2.0.0"
        assert config.log_level == "DEBUG"
        assert config.llm.provider == LLMProvider.ANTHROPIC
        assert config.knowledge.provider == KnowledgeProvider.OPENSEARCH
        assert config.executor.provider == ExecutorProvider.API
        assert config.workflow.strategy == "plan_execute"

    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = AgentConfig(name="test", log_level=level)
            assert config.log_level == level
        
        # Case insensitive
        config = AgentConfig(name="test", log_level="debug")
        assert config.log_level == "DEBUG"
        
        # Invalid log level
        with pytest.raises(ValidationError):
            AgentConfig(name="test", log_level="INVALID")

    def test_directory_creation(self):
        """Test that data and cache directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            cache_dir = Path(temp_dir) / "cache"
            
            config = AgentConfig(
                name="test",
                data_dir=data_dir,
                cache_dir=cache_dir
            )
            
            assert data_dir.exists()
            assert cache_dir.exists()
            assert config.data_dir == data_dir
            assert config.cache_dir == cache_dir

    def test_knowledge_path_auto_configuration(self):
        """Test automatic knowledge base path configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            
            config = AgentConfig(
                name="test",
                data_dir=data_dir,
                knowledge={"provider": "faiss"}
            )
            
            expected_path = data_dir / "knowledge" / "faiss_index"
            assert config.knowledge.index_path == expected_path

    def test_missing_name_validation(self):
        """Test that name is required."""
        with pytest.raises(ValidationError):
            AgentConfig()

    def test_extra_config(self):
        """Test extra configuration handling."""
        config = AgentConfig(
            name="test",
            extra_config={
                "custom_setting": "value",
                "nested": {"key": "value"}
            }
        )
        
        assert config.extra_config["custom_setting"] == "value"
        assert config.extra_config["nested"]["key"] == "value"