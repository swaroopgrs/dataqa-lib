"""
Configuration loading and management utilities.

This module provides functions to load and validate DataQA configurations
from YAML files with environment variable substitution support.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import ValidationError

from .models import AgentConfig


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


def substitute_env_vars(data: Any) -> Any:
    """
    Recursively substitute environment variables in configuration data.
    
    Supports the format ${VAR_NAME} or ${VAR_NAME:default_value}.
    
    Args:
        data: Configuration data (dict, list, str, or other)
        
    Returns:
        Configuration data with environment variables substituted
    """
    if isinstance(data, dict):
        return {key: substitute_env_vars(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [substitute_env_vars(item) for item in data]
    elif isinstance(data, str):
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default_value)
        
        return re.sub(pattern, replace_var, data)
    else:
        return data


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file with environment variable substitution.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        ConfigurationError: If file cannot be loaded or parsed
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML configuration: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to read configuration file: {e}")
    
    if raw_config is None:
        raise ConfigurationError("Configuration file is empty")
    
    # Substitute environment variables
    config = substitute_env_vars(raw_config)
    
    return config


def load_agent_config(config_path: Union[str, Path]) -> AgentConfig:
    """
    Load and validate agent configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated AgentConfig instance
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        config_data = load_yaml_config(config_path)
        return AgentConfig(**config_data)
    except ValidationError as e:
        error_details = []
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error['loc'])
            error_details.append(f"{loc}: {error['msg']}")
        
        raise ConfigurationError(
            f"Configuration validation failed:\n" + "\n".join(error_details)
        )
    except Exception as e:
        raise ConfigurationError(f"Failed to load agent configuration: {e}")


def save_agent_config(config: AgentConfig, config_path: Union[str, Path]) -> None:
    """
    Save agent configuration to YAML file.
    
    Args:
        config: AgentConfig instance to save
        config_path: Path where to save the configuration
        
    Raises:
        ConfigurationError: If configuration cannot be saved
    """
    config_path = Path(config_path)
    
    try:
        # Create parent directories if they don't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and handle special types
        config_dict = config.model_dump(
            exclude_none=True,
            by_alias=True,
            mode='json'
        )
        
        # Convert Path objects to strings for YAML serialization
        def convert_paths(data):
            if isinstance(data, dict):
                return {key: convert_paths(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [convert_paths(item) for item in data]
            elif hasattr(data, '__fspath__'):  # Path-like object
                return str(data)
            else:
                return data
        
        config_dict = convert_paths(config_dict)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                config_dict,
                f,
                default_flow_style=False,
                sort_keys=False,
                indent=2
            )
    except Exception as e:
        raise ConfigurationError(f"Failed to save configuration: {e}")


def create_example_config(config_path: Union[str, Path]) -> AgentConfig:
    """
    Create an example configuration file with default values.
    
    Args:
        config_path: Path where to save the example configuration
        
    Returns:
        Example AgentConfig instance
    """
    example_config = AgentConfig(
        name="example-agent",
        description="Example DataQA agent configuration",
        llm={
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "${OPENAI_API_KEY}",
            "temperature": 0.1,
            "max_tokens": 2000
        },
        knowledge={
            "provider": "faiss",
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 512,
            "top_k": 5
        },
        executor={
            "provider": "inmemory",
            "database_type": "duckdb",
            "max_execution_time": 30.0,
            "require_approval": True
        },
        workflow={
            "strategy": "react",
            "max_iterations": 10,
            "require_approval": True,
            "conversation_memory": True
        }
    )
    
    save_agent_config(example_config, config_path)
    return example_config


def validate_environment() -> Dict[str, bool]:
    """
    Validate that required environment variables are set.
    
    Returns:
        Dictionary mapping environment variable names to availability status
    """
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY") is not None,
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY") is not None,
    }
    
    optional_vars = {
        "DATAQA_LOG_LEVEL": os.getenv("DATAQA_LOG_LEVEL") is not None,
        "DATAQA_DATA_DIR": os.getenv("DATAQA_DATA_DIR") is not None,
        "DATAQA_CACHE_DIR": os.getenv("DATAQA_CACHE_DIR") is not None,
    }
    
    return {**required_vars, **optional_vars}


def get_config_from_env() -> Optional[AgentConfig]:
    """
    Create agent configuration from environment variables.
    
    This is useful for containerized deployments where configuration
    is provided through environment variables rather than files.
    
    Returns:
        AgentConfig instance if sufficient environment variables are set,
        None otherwise
    """
    # Check if we have minimum required configuration
    if not os.getenv("DATAQA_AGENT_NAME"):
        return None
    
    config_data = {
        "name": os.getenv("DATAQA_AGENT_NAME"),
        "description": os.getenv("DATAQA_AGENT_DESCRIPTION"),
    }
    
    # LLM configuration
    llm_config = {}
    if os.getenv("DATAQA_LLM_PROVIDER"):
        llm_config["provider"] = os.getenv("DATAQA_LLM_PROVIDER")
    if os.getenv("DATAQA_LLM_MODEL"):
        llm_config["model"] = os.getenv("DATAQA_LLM_MODEL")
    if os.getenv("DATAQA_LLM_API_KEY"):
        llm_config["api_key"] = os.getenv("DATAQA_LLM_API_KEY")
    if os.getenv("DATAQA_LLM_TEMPERATURE"):
        try:
            llm_config["temperature"] = float(os.getenv("DATAQA_LLM_TEMPERATURE"))
        except ValueError:
            return None
    
    if llm_config:
        config_data["llm"] = llm_config
    
    # Knowledge configuration
    knowledge_config = {}
    if os.getenv("DATAQA_KNOWLEDGE_PROVIDER"):
        knowledge_config["provider"] = os.getenv("DATAQA_KNOWLEDGE_PROVIDER")
    if os.getenv("DATAQA_KNOWLEDGE_INDEX_PATH"):
        knowledge_config["index_path"] = os.getenv("DATAQA_KNOWLEDGE_INDEX_PATH")
    
    if knowledge_config:
        config_data["knowledge"] = knowledge_config
    
    # Executor configuration
    executor_config = {}
    if os.getenv("DATAQA_EXECUTOR_PROVIDER"):
        executor_config["provider"] = os.getenv("DATAQA_EXECUTOR_PROVIDER")
    if os.getenv("DATAQA_DATABASE_URL"):
        executor_config["database_url"] = os.getenv("DATAQA_DATABASE_URL")
    
    if executor_config:
        config_data["executor"] = executor_config
    
    # Global settings
    if os.getenv("DATAQA_LOG_LEVEL"):
        config_data["log_level"] = os.getenv("DATAQA_LOG_LEVEL")
    if os.getenv("DATAQA_DATA_DIR"):
        config_data["data_dir"] = os.getenv("DATAQA_DATA_DIR")
    if os.getenv("DATAQA_CACHE_DIR"):
        config_data["cache_dir"] = os.getenv("DATAQA_CACHE_DIR")
    
    try:
        return AgentConfig(**config_data)
    except ValidationError:
        return None