# DataQA Configuration Guide

This guide covers all configuration options available in DataQA for customizing agent behavior.

## Configuration Overview

DataQA uses YAML configuration files with Pydantic validation to define agent behavior. Configuration files follow a hierarchical structure with the following main sections:

- **Agent Metadata**: Basic information about the agent
- **LLM Configuration**: Language model settings
- **Knowledge Configuration**: Knowledge base and RAG settings
- **Executor Configuration**: Code execution environment settings
- **Workflow Configuration**: Agent orchestration and behavior settings

## Basic Configuration Structure

```yaml
# Agent metadata
name: "agent-name"
description: "Agent description"
version: "1.0.0"

# LLM settings
llm:
  provider: "openai"
  model: "gpt-4"
  api_key: "${OPENAI_API_KEY}"

# Knowledge base settings
knowledge:
  provider: "faiss"
  embedding_model: "all-MiniLM-L6-v2"

# Code execution settings
executor:
  provider: "inmemory"
  database_type: "duckdb"

# Workflow settings
workflow:
  strategy: "react"
  require_approval: true

# Global settings
log_level: "INFO"
```

## Configuration Sections

### Agent Metadata

Basic information about the agent:

```yaml
name: "my-agent"                    # Required: Unique agent identifier
description: "Agent description"    # Optional: Human-readable description
version: "1.0.0"                   # Optional: Agent version
```

### LLM Configuration

Language model provider and settings:

```yaml
llm:
  provider: "openai"                # Required: "openai", "anthropic", "local"
  model: "gpt-4"                    # Required: Model name
  api_key: "${OPENAI_API_KEY}"      # Required: API key (use env vars)
  temperature: 0.1                  # Optional: Response randomness (0.0-2.0)
  max_tokens: 2000                  # Optional: Maximum response length
  timeout: 30.0                     # Optional: Request timeout in seconds
  max_retries: 3                    # Optional: Number of retry attempts
  
  # Optional: Provider-specific parameters
  extra_params:
    frequency_penalty: 0.0          # OpenAI: Reduce repetition
    presence_penalty: 0.0           # OpenAI: Encourage topic diversity
    top_p: 1.0                      # OpenAI: Nucleus sampling
    stop: ["END"]                   # OpenAI: Stop sequences
```

#### Supported LLM Providers

**OpenAI**
```yaml
llm:
  provider: "openai"
  model: "gpt-4"                    # or "gpt-3.5-turbo", "gpt-4-turbo"
  api_key: "${OPENAI_API_KEY}"
  base_url: "https://api.openai.com/v1"  # Optional: Custom endpoint
```

**Anthropic**
```yaml
llm:
  provider: "anthropic"
  model: "claude-3-opus-20240229"   # or "claude-3-sonnet-20240229"
  api_key: "${ANTHROPIC_API_KEY}"
```

**Local Models**
```yaml
llm:
  provider: "local"
  model: "llama2-7b"
  base_url: "http://localhost:8000"  # Local model server
  api_key: "not-required"
```

### Knowledge Configuration

Knowledge base and retrieval settings:

```yaml
knowledge:
  provider: "faiss"                 # Required: "faiss", "opensearch", "memory"
  embedding_model: "all-MiniLM-L6-v2"  # Required: Embedding model name
  chunk_size: 512                   # Optional: Document chunk size
  chunk_overlap: 50                 # Optional: Overlap between chunks
  top_k: 5                          # Optional: Number of results to retrieve
  similarity_threshold: 0.7         # Optional: Minimum similarity score
  
  # FAISS-specific settings (default provider)
  index_type: "flat"                # Optional: "flat", "ivf", "hnsw"
  
  # OpenSearch settings (when provider is opensearch)
  opensearch_host: "${OPENSEARCH_HOST}"
  opensearch_port: 9200
  opensearch_username: "${OPENSEARCH_USERNAME}"
  opensearch_password: "${OPENSEARCH_PASSWORD}"
  opensearch_index: "dataqa-knowledge"
  opensearch_ssl: true
  opensearch_verify_certs: true
```

#### Supported Knowledge Providers

**FAISS (Local Vector Store)**
```yaml
knowledge:
  provider: "faiss"
  embedding_model: "all-MiniLM-L6-v2"  # Fast, lightweight
  # or "all-mpnet-base-v2"             # Higher quality
  # or "text-embedding-ada-002"        # OpenAI embeddings
  index_type: "flat"                    # Simple, exact search
  # or "ivf"                            # Faster for large datasets
  # or "hnsw"                           # Best for very large datasets
```

**OpenSearch (Production Vector Store)**
```yaml
knowledge:
  provider: "opensearch"
  embedding_model: "all-mpnet-base-v2"
  opensearch_host: "search-domain.region.es.amazonaws.com"
  opensearch_port: 443
  opensearch_ssl: true
  opensearch_index: "dataqa-prod-knowledge"
```

**Memory (Simple In-Memory)**
```yaml
knowledge:
  provider: "memory"
  embedding_model: "all-MiniLM-L6-v2"
  # Suitable for small knowledge bases or testing
```

### Executor Configuration

Code execution environment settings:

```yaml
executor:
  provider: "inmemory"              # Required: "inmemory", "api", "docker"
  database_type: "duckdb"           # Optional: "duckdb", "sqlite", "postgresql"
  max_execution_time: 30.0          # Optional: Timeout in seconds
  max_memory_mb: 512                # Optional: Memory limit in MB
  max_rows: 10000                   # Optional: Maximum result rows
  
  # Security settings
  allow_file_access: false          # Optional: Allow file operations
  allowed_imports:                  # Optional: Whitelist of allowed imports
    - "pandas"
    - "numpy"
    - "matplotlib"
    - "seaborn"
  blocked_functions:                # Optional: Blacklist of blocked functions
    - "exec"
    - "eval"
    - "open"
    - "__import__"
  
  # Database connection (optional)
  database_url: "${DATABASE_URL}"   # Connection string
  
  # API executor settings (when provider is api)
  api_url: "${EXECUTOR_API_URL}"
  api_key: "${EXECUTOR_API_KEY}"
  api_timeout: 60.0
```

#### Supported Executor Providers

**In-Memory (Development)**
```yaml
executor:
  provider: "inmemory"
  database_type: "duckdb"           # Fast, in-memory SQL engine
  max_execution_time: 60.0
  max_memory_mb: 1024
  allow_file_access: true           # For development convenience
```

**API (Production)**
```yaml
executor:
  provider: "api"
  api_url: "https://executor.company.com/execute"
  api_key: "${EXECUTOR_API_KEY}"
  max_execution_time: 120.0
  # Remote execution for security and scalability
```

**Docker (Isolated)**
```yaml
executor:
  provider: "docker"
  docker_image: "dataqa/executor:latest"
  docker_timeout: 180.0
  docker_memory_limit: "2g"
  # Containerized execution for maximum isolation
```

### Workflow Configuration

Agent orchestration and behavior settings:

```yaml
workflow:
  strategy: "react"                 # Required: "react", "workflow", "plan_execute"
  max_iterations: 10                # Optional: Maximum workflow iterations
  require_approval: true            # Optional: Require human approval
  auto_approve_safe: false          # Optional: Auto-approve safe operations
  conversation_memory: true         # Optional: Maintain conversation context
  max_context_length: 4000          # Optional: Maximum context tokens
  enable_visualization: true        # Optional: Enable chart generation
  debug_mode: false                 # Optional: Enable debug output
  
  # Advanced workflow settings
  planning_depth: 3                 # Optional: Planning lookahead depth
  error_recovery: true              # Optional: Attempt error recovery
  parallel_execution: false         # Optional: Enable parallel operations
```

#### Workflow Strategies

**ReAct (Reasoning and Acting)**
```yaml
workflow:
  strategy: "react"
  max_iterations: 10
  # Best for: Interactive queries, exploratory analysis
  # Behavior: Iterative reasoning and action cycles
```

**Workflow (Linear Task Execution)**
```yaml
workflow:
  strategy: "workflow"
  max_iterations: 5
  # Best for: Structured, predictable tasks
  # Behavior: Linear sequence of predefined steps
```

**Plan-Execute (Strategic Planning)**
```yaml
workflow:
  strategy: "plan_execute"
  max_iterations: 15
  planning_depth: 5
  # Best for: Complex, multi-step analysis
  # Behavior: Create plan first, then execute systematically
```

### Global Settings

System-wide configuration options:

```yaml
log_level: "INFO"                   # Optional: "DEBUG", "INFO", "WARNING", "ERROR"
data_dir: "./data"                  # Optional: Data storage directory
cache_dir: "./cache"                # Optional: Cache storage directory

# Optional: Custom configuration
extra_config:
  custom_prompts:
    system_prompt: "Custom system prompt"
  features:
    experimental_features: true
    beta_features: false
  monitoring:
    metrics_enabled: true
    tracing_enabled: false
```

## Environment Variables

DataQA supports environment variable substitution in configuration files:

```yaml
# Use ${VAR_NAME} syntax for substitution
llm:
  api_key: "${OPENAI_API_KEY}"
  
# With default values
database_url: "${DATABASE_URL:-sqlite:///default.db}"

# Multiple variables
custom_setting: "${PREFIX}_${ENVIRONMENT}_${SUFFIX}"
```

### Common Environment Variables

```bash
# LLM API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Database Connections
export DATABASE_URL="postgresql://user:pass@host:port/db"

# Knowledge Base
export OPENSEARCH_HOST="search-domain.region.es.amazonaws.com"
export OPENSEARCH_USERNAME="admin"
export OPENSEARCH_PASSWORD="password"

# Executor API
export EXECUTOR_API_URL="https://executor.company.com"
export EXECUTOR_API_KEY="exec-key-..."

# Global Settings
export DATAQA_LOG_LEVEL="DEBUG"
export DATAQA_DATA_DIR="/app/data"
export DATAQA_CACHE_DIR="/app/cache"
```

## Configuration Validation

DataQA validates all configuration using Pydantic models. Common validation errors:

### Missing Required Fields
```yaml
# Error: Missing required field
llm:
  provider: "openai"
  # model: "gpt-4"  # This is required!
```

### Invalid Values
```yaml
# Error: Invalid provider
llm:
  provider: "invalid-provider"  # Must be "openai", "anthropic", or "local"
```

### Type Mismatches
```yaml
# Error: Wrong type
executor:
  max_execution_time: "30"  # Should be number, not string
```

## Configuration Inheritance

You can extend base configurations:

**base_config.yaml**
```yaml
name: "base-agent"
llm:
  provider: "openai"
  temperature: 0.1
executor:
  provider: "inmemory"
  max_execution_time: 30.0
```

**specialized_config.yaml**
```yaml
# Inherit from base and override specific settings
extends: "base_config.yaml"
name: "specialized-agent"
llm:
  model: "gpt-4"  # Override model
executor:
  max_execution_time: 60.0  # Override timeout
```

## Configuration Profiles

Use different configurations for different environments:

**config/development.yaml**
```yaml
name: "dev-agent"
llm:
  model: "gpt-3.5-turbo"  # Cheaper for development
  temperature: 0.2
executor:
  allow_file_access: true
  max_execution_time: 120.0
workflow:
  require_approval: false  # Auto-approve for development
  debug_mode: true
log_level: "DEBUG"
```

**config/production.yaml**
```yaml
name: "prod-agent"
llm:
  model: "gpt-4"
  temperature: 0.05
knowledge:
  provider: "opensearch"
executor:
  provider: "api"
  allow_file_access: false
workflow:
  require_approval: true
  debug_mode: false
log_level: "INFO"
```

## Best Practices

### Security
- Always use environment variables for secrets
- Never commit API keys to version control
- Use restrictive executor settings in production
- Enable approval workflows for sensitive operations

### Performance
- Use appropriate LLM models for your use case
- Optimize knowledge base settings for your data size
- Set reasonable timeouts and resource limits
- Enable caching for repeated operations

### Maintainability
- Use descriptive agent names and descriptions
- Document custom configuration in comments
- Use configuration inheritance to reduce duplication
- Validate configurations in CI/CD pipelines

### Monitoring
- Enable appropriate logging levels
- Use structured logging in production
- Monitor resource usage and performance
- Set up alerts for configuration errors

## Configuration Examples

See the following example configurations:

- [Basic Agent](../config/basic_agent.yaml) - Minimal setup
- [Development Agent](../config/development_agent.yaml) - Development-optimized
- [Enterprise Agent](../config/enterprise_agent.yaml) - Production-ready
- [Analytics Agent](../config/analytics_agent.yaml) - Analytics-focused

## Troubleshooting Configuration

Common configuration issues and solutions:

1. **Environment variables not substituted**: Check variable names and ensure they're exported
2. **Validation errors**: Review required fields and data types
3. **Connection failures**: Verify network settings and credentials
4. **Performance issues**: Adjust timeouts and resource limits
5. **Permission errors**: Check file system permissions for data/cache directories

For more troubleshooting help, see the [Troubleshooting Guide](troubleshooting.md).