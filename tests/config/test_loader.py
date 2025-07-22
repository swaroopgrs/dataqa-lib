"""
Unit tests for configuration loader.

Tests YAML loading, environment variable substitution,
and configuration validation.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from dataqa.config.loader import (
    ConfigurationError,
    create_example_config,
    get_config_from_env,
    load_agent_config,
    load_yaml_config,
    save_agent_config,
    substitute_env_vars,
    validate_environment,
)
from dataqa.config.models import AgentConfig, LLMProvider


class TestSubstituteEnvVars:
    """Test environment variable substitution."""

    @patch.dict(os.environ, {"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"})
    def test_simple_substitution(self):
        """Test simple environment variable substitution."""
        data = {"key": "${TEST_VAR}"}
        result = substitute_env_vars(data)
        assert result["key"] == "test_value"

    @patch.dict(os.environ, {"TEST_VAR": "test_value"})
    def test_default_value_not_used(self):
        """Test that default value is not used when env var exists."""
        data = {"key": "${TEST_VAR:default}"}
        result = substitute_env_vars(data)
        assert result["key"] == "test_value"

    def test_default_value_used(self):
        """Test that default value is used when env var doesn't exist."""
        data = {"key": "${MISSING_VAR:default_value}"}
        result = substitute_env_vars(data)
        assert result["key"] == "default_value"

    def test_missing_var_no_default(self):
        """Test missing variable with no default returns empty string."""
        data = {"key": "${MISSING_VAR}"}
        result = substitute_env_vars(data)
        assert result["key"] == ""

    @patch.dict(os.environ, {"VAR1": "value1", "VAR2": "value2"})
    def test_nested_substitution(self):
        """Test substitution in nested data structures."""
        data = {
            "level1": {
                "level2": ["${VAR1}", "${VAR2}"],
                "key": "${VAR1}_${VAR2}"
            }
        }
        result = substitute_env_vars(data)
        assert result["level1"]["level2"] == ["value1", "value2"]
        assert result["level1"]["key"] == "value1_value2"

    def test_non_string_values_unchanged(self):
        """Test that non-string values are not modified."""
        data = {
            "string": "${TEST_VAR}",
            "number": 42,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3]
        }
        result = substitute_env_vars(data)
        assert result["string"] == ""  # Missing env var
        assert result["number"] == 42
        assert result["boolean"] is True
        assert result["null"] is None
        assert result["list"] == [1, 2, 3]


class TestLoadYamlConfig:
    """Test YAML configuration loading."""

    def test_load_valid_yaml(self):
        """Test loading valid YAML configuration."""
        config_data = {
            "name": "test-agent",
            "llm": {"provider": "openai", "model": "gpt-4"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            result = load_yaml_config(config_path)
            assert result["name"] == "test-agent"
            assert result["llm"]["provider"] == "openai"
        finally:
            os.unlink(config_path)

    @patch.dict(os.environ, {"API_KEY": "secret-key"})
    def test_load_yaml_with_env_vars(self):
        """Test loading YAML with environment variable substitution."""
        config_data = {
            "name": "test-agent",
            "llm": {"api_key": "${API_KEY}"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            result = load_yaml_config(config_path)
            assert result["llm"]["api_key"] == "secret-key"
        finally:
            os.unlink(config_path)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_yaml_config("nonexistent.yaml")

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="Failed to parse YAML"):
                load_yaml_config(config_path)
        finally:
            os.unlink(config_path)

    def test_load_empty_file(self):
        """Test loading empty file raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="Configuration file is empty"):
                load_yaml_config(config_path)
        finally:
            os.unlink(config_path)


class TestLoadAgentConfig:
    """Test agent configuration loading."""

    def test_load_valid_config(self):
        """Test loading valid agent configuration."""
        config_data = {
            "name": "test-agent",
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.5
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_agent_config(config_path)
            assert isinstance(config, AgentConfig)
            assert config.name == "test-agent"
            assert config.llm.provider == LLMProvider.OPENAI
            assert config.llm.model == "gpt-4"
            assert config.llm.temperature == 0.5
        finally:
            os.unlink(config_path)

    def test_load_invalid_config(self):
        """Test loading invalid configuration raises validation error."""
        config_data = {
            "name": "test-agent",
            "llm": {
                "temperature": 5.0  # Invalid temperature > 2.0
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="Configuration validation failed"):
                load_agent_config(config_path)
        finally:
            os.unlink(config_path)

    def test_load_missing_required_field(self):
        """Test loading configuration with missing required field."""
        config_data = {
            # Missing required 'name' field
            "llm": {"provider": "openai"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="Configuration validation failed"):
                load_agent_config(config_path)
        finally:
            os.unlink(config_path)


class TestSaveAgentConfig:
    """Test agent configuration saving."""

    def test_save_and_load_roundtrip(self):
        """Test saving and loading configuration roundtrip."""
        original_config = AgentConfig(
            name="test-agent",
            description="Test agent",
            llm={
                "provider": "anthropic",
                "model": "claude-3-sonnet",
                "temperature": 0.7
            }
        )
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            save_agent_config(original_config, config_path)
            loaded_config = load_agent_config(config_path)
            
            assert loaded_config.name == original_config.name
            assert loaded_config.description == original_config.description
            assert loaded_config.llm.provider == original_config.llm.provider
            assert loaded_config.llm.model == original_config.llm.model
            assert loaded_config.llm.temperature == original_config.llm.temperature
        finally:
            os.unlink(config_path)

    def test_save_creates_directories(self):
        """Test that saving creates parent directories."""
        config = AgentConfig(name="test-agent")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "subdir" / "config.yaml"
            save_agent_config(config, config_path)
            
            assert config_path.exists()
            assert config_path.parent.exists()


class TestCreateExampleConfig:
    """Test example configuration creation."""

    def test_create_example_config(self):
        """Test creating example configuration."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            config = create_example_config(config_path)
            
            assert isinstance(config, AgentConfig)
            assert config.name == "example-agent"
            assert Path(config_path).exists()
            
            # Verify we can load it back
            loaded_config = load_agent_config(config_path)
            assert loaded_config.name == config.name
        finally:
            os.unlink(config_path)


class TestValidateEnvironment:
    """Test environment validation."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_validate_with_openai_key(self):
        """Test validation with OpenAI API key set."""
        result = validate_environment()
        assert result["OPENAI_API_KEY"] is True
        assert result["ANTHROPIC_API_KEY"] is False

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_validate_with_anthropic_key(self):
        """Test validation with Anthropic API key set."""
        result = validate_environment()
        assert result["OPENAI_API_KEY"] is False
        assert result["ANTHROPIC_API_KEY"] is True

    def test_validate_no_keys(self):
        """Test validation with no API keys set."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_environment()
            assert result["OPENAI_API_KEY"] is False
            assert result["ANTHROPIC_API_KEY"] is False


class TestGetConfigFromEnv:
    """Test configuration creation from environment variables."""

    def test_no_agent_name_returns_none(self):
        """Test that missing agent name returns None."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_config_from_env()
            assert config is None

    @patch.dict(os.environ, {
        "DATAQA_AGENT_NAME": "env-agent",
        "DATAQA_AGENT_DESCRIPTION": "Agent from environment",
        "DATAQA_LLM_PROVIDER": "anthropic",
        "DATAQA_LLM_MODEL": "claude-3-sonnet",
        "DATAQA_LOG_LEVEL": "DEBUG"
    })
    def test_create_config_from_env(self):
        """Test creating configuration from environment variables."""
        config = get_config_from_env()
        
        assert config is not None
        assert config.name == "env-agent"
        assert config.description == "Agent from environment"
        assert config.llm.provider == LLMProvider.ANTHROPIC
        assert config.llm.model == "claude-3-sonnet"
        assert config.log_level == "DEBUG"

    @patch.dict(os.environ, {
        "DATAQA_AGENT_NAME": "env-agent",
        "DATAQA_LLM_TEMPERATURE": "invalid"  # Invalid float
    })
    def test_invalid_env_config_returns_none(self):
        """Test that invalid environment configuration returns None."""
        config = get_config_from_env()
        assert config is None