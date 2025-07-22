#!/usr/bin/env python3
"""
Demonstration script for DataQA configuration system.

This script shows how to:
1. Load configuration from YAML files
2. Create configurations programmatically
3. Handle environment variable substitution
4. Validate configurations
"""

import os
import tempfile
from pathlib import Path

from dataqa.config import (
    AgentConfig,
    ConfigurationError,
    create_example_config,
    load_agent_config,
    save_agent_config,
    validate_environment,
)


def demo_yaml_loading():
    """Demonstrate loading configuration from YAML."""
    print("=== YAML Configuration Loading ===")
    
    try:
        config = load_agent_config("config/example_agent.yaml")
        print(f"✓ Loaded agent: {config.name}")
        print(f"  Description: {config.description}")
        print(f"  LLM: {config.llm.provider.value} ({config.llm.model})")
        print(f"  Knowledge: {config.knowledge.provider.value}")
        print(f"  Executor: {config.executor.provider.value}")
        print(f"  Log level: {config.log_level}")
    except ConfigurationError as e:
        print(f"✗ Configuration error: {e}")
    
    print()


def demo_programmatic_creation():
    """Demonstrate creating configuration programmatically."""
    print("=== Programmatic Configuration Creation ===")
    
    config = AgentConfig(
        name="demo-agent",
        description="Programmatically created agent",
        llm={
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "temperature": 0.3,
            "max_tokens": 1500
        },
        knowledge={
            "provider": "opensearch",
            "chunk_size": 256,
            "top_k": 3
        },
        executor={
            "provider": "api",
            "max_execution_time": 45.0,
            "require_approval": False
        },
        log_level="DEBUG"
    )
    
    print(f"✓ Created agent: {config.name}")
    print(f"  LLM: {config.llm.provider.value} ({config.llm.model})")
    print(f"  Temperature: {config.llm.temperature}")
    print(f"  Knowledge: {config.knowledge.provider.value}")
    print(f"  Chunk size: {config.knowledge.chunk_size}")
    print(f"  Executor: {config.executor.provider.value}")
    print(f"  Max execution time: {config.executor.max_execution_time}s")
    
    print()


def demo_environment_variables():
    """Demonstrate environment variable handling."""
    print("=== Environment Variable Handling ===")
    
    # Check current environment
    env_status = validate_environment()
    print("Environment variable status:")
    for var, available in env_status.items():
        status = "✓" if available else "✗"
        print(f"  {status} {var}: {'Available' if available else 'Not set'}")
    
    # Demonstrate with temporary environment variables
    print("\nTesting with temporary environment variables...")
    
    # Set temporary environment variables
    os.environ["TEST_API_KEY"] = "demo-key-12345"
    os.environ["TEST_DB_URL"] = "postgresql://user:pass@localhost/testdb"
    
    # Create config with environment variable references
    config_data = {
        "name": "env-test-agent",
        "llm": {
            "provider": "openai",
            "api_key": "${TEST_API_KEY}",
            "model": "gpt-3.5-turbo"
        },
        "executor": {
            "database_url": "${TEST_DB_URL}"
        }
    }
    
    # Save and load to test environment substitution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config_data, f)
        temp_config_path = f.name
    
    try:
        config = load_agent_config(temp_config_path)
        print(f"✓ Environment substitution successful:")
        print(f"  API key: {config.llm.api_key.get_secret_value()}")
        print(f"  Database URL: {config.executor.database_url.get_secret_value()}")
    except Exception as e:
        print(f"✗ Environment substitution failed: {e}")
    finally:
        os.unlink(temp_config_path)
        # Clean up environment
        del os.environ["TEST_API_KEY"]
        del os.environ["TEST_DB_URL"]
    
    print()


def demo_validation():
    """Demonstrate configuration validation."""
    print("=== Configuration Validation ===")
    
    # Test valid configuration
    try:
        valid_config = AgentConfig(
            name="valid-agent",
            llm={"temperature": 0.5},  # Valid temperature
            executor={"max_execution_time": 30.0}  # Valid time
        )
        print("✓ Valid configuration accepted")
    except Exception as e:
        print(f"✗ Unexpected validation error: {e}")
    
    # Test invalid configurations
    invalid_configs = [
        {
            "name": "Invalid temperature",
            "config": lambda: AgentConfig(name="test", llm={"temperature": 3.0}),
            "expected_error": "temperature"
        },
        {
            "name": "Invalid log level", 
            "config": lambda: AgentConfig(name="test", log_level="INVALID"),
            "expected_error": "log level"
        },
        {
            "name": "Missing name",
            "config": lambda: AgentConfig(),
            "expected_error": "name"
        }
    ]
    
    for test_case in invalid_configs:
        try:
            test_case["config"]()
            print(f"✗ {test_case['name']}: Should have failed validation")
        except Exception as e:
            print(f"✓ {test_case['name']}: Correctly rejected ({type(e).__name__})")
    
    print()


def demo_example_creation():
    """Demonstrate creating example configurations."""
    print("=== Example Configuration Creation ===")
    
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        example_path = f.name
    
    try:
        config = create_example_config(example_path)
        print(f"✓ Created example configuration at: {example_path}")
        print(f"  Agent name: {config.name}")
        print(f"  Configuration version: {config.version}")
        
        # Verify we can load it back
        loaded_config = load_agent_config(example_path)
        print(f"✓ Successfully loaded example configuration")
        print(f"  Loaded agent: {loaded_config.name}")
        
    except Exception as e:
        print(f"✗ Failed to create example: {e}")
    finally:
        if Path(example_path).exists():
            os.unlink(example_path)
    
    print()


def main():
    """Run all configuration demonstrations."""
    print("DataQA Configuration System Demonstration")
    print("=" * 50)
    print()
    
    demo_yaml_loading()
    demo_programmatic_creation()
    demo_environment_variables()
    demo_validation()
    demo_example_creation()
    
    print("Configuration system demonstration complete!")


if __name__ == "__main__":
    main()