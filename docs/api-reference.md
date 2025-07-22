# DataQA API Reference

This document provides comprehensive API reference for the DataQA framework.

## Core Classes

### DataAgent

The main agent class for interacting with data through natural language.

```python
class DataAgent:
    def __init__(self, config: AgentConfig)
    def query(self, query: str, conversation_id: str = "default") -> str
    def ingest_knowledge(self, documents: List[Document]) -> None
    def get_agent_info(self) -> Dict[str, Any]
    def shutdown(self) -> None
```

#### Methods

##### `__init__(config: AgentConfig)`

Initialize a new DataAgent with the provided configuration.

**Parameters:**
- `config` (AgentConfig): Agent configuration object

**Example:**
```python
from dataqa import DataAgent, AgentConfig

config = AgentConfig.from_yaml("agent.yaml")
agent = DataAgent(config)
```

##### `query(query: str, conversation_id: str = "default") -> str`

Process a natural language query and return the response.

**Parameters:**
- `query` (str): Natural language query
- `conversation_id` (str, optional): Conversation identifier for context. Defaults to "default"

**Returns:**
- `str`: Agent response with analysis results

**Raises:**
- `ValueError`: If query is empty or invalid
- `ExecutionError`: If code execution fails
- `LLMError`: If LLM request fails

**Example:**
```python
response = agent.query("What are our top 5 products by revenue?")
print(response)
```

##### `ingest_knowledge(documents: List[Document]) -> None`

Add knowledge documents to the agent's knowledge base.

**Parameters:**
- `documents` (List[Document]): List of documents to ingest

**Example:**
```python
from dataqa import Document

docs = [
    Document(
        content="Sales table has columns: id, date, amount, customer_id",
        metadata={"type": "schema"},
        source="schema.md"
    )
]
agent.ingest_knowledge(docs)
```

##### `get_agent_info() -> Dict[str, Any]`

Get information about the agent configuration and status.

**Returns:**
- `Dict[str, Any]`: Agent information including name, version, and status

##### `shutdown() -> None`

Clean up agent resources and close connections.

### DataQAClient

High-level client for managing multiple agents.

```python
class DataQAClient:
    def __init__(self)
    def create_agent(self, name: str, config: Union[Dict, AgentConfig, str]) -> DataAgent
    def list_agents(self) -> List[str]
    def query(self, agent_name: str, query: str, conversation_id: str = "default") -> str
    def shutdown_agent(self, agent_name: str) -> None
    def shutdown_all(self) -> None
```

#### Methods

##### `create_agent(name: str, config: Union[Dict, AgentConfig, str]) -> DataAgent`

Create a new agent with the specified configuration.

**Parameters:**
- `name` (str): Unique agent name
- `config` (Union[Dict, AgentConfig, str]): Agent configuration

**Returns:**
- `DataAgent`: Created agent instance

**Example:**
```python
from dataqa import DataQAClient

client = DataQAClient()
agent = client.create_agent("my-agent", config_path="config.yaml")
```

##### `query(agent_name: str, query: str, conversation_id: str = "default") -> str`

Query a specific agent by name.

**Parameters:**
- `agent_name` (str): Name of the agent to query
- `query` (str): Natural language query
- `conversation_id` (str, optional): Conversation identifier

**Returns:**
- `str`: Agent response

### AgentConfig

Configuration class for defining agent behavior.

```python
class AgentConfig(BaseModel):
    name: str
    description: Optional[str]
    llm: LLMConfig
    knowledge: KnowledgeConfig
    executor: ExecutorConfig
    workflow: WorkflowConfig
    log_level: str = "INFO"
    data_dir: Optional[Path]
    cache_dir: Optional[Path]
```

#### Class Methods

##### `from_yaml(path: Union[str, Path]) -> AgentConfig`

Load configuration from a YAML file.

**Parameters:**
- `path` (Union[str, Path]): Path to YAML configuration file

**Returns:**
- `AgentConfig`: Loaded configuration

**Example:**
```python
config = AgentConfig.from_yaml("config/my_agent.yaml")
```

##### `from_dict(data: Dict[str, Any]) -> AgentConfig`

Create configuration from a dictionary.

**Parameters:**
- `data` (Dict[str, Any]): Configuration dictionary

**Returns:**
- `AgentConfig`: Created configuration

### Document

Represents a knowledge document for the agent.

```python
class Document(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}
    source: str
    embedding: Optional[List[float]] = None
```

#### Fields

- `content` (str): Document text content
- `metadata` (Dict[str, Any]): Additional metadata
- `source` (str): Source identifier
- `embedding` (Optional[List[float]]): Pre-computed embedding vector

**Example:**
```python
doc = Document(
    content="Customer table contains id, name, email, created_date columns",
    metadata={"type": "schema", "table": "customers"},
    source="database_schema.sql"
)
```

## Configuration Classes

### LLMConfig

Configuration for Language Model settings.

```python
class LLMConfig(BaseModel):
    provider: str  # "openai", "anthropic", "local"
    model: str
    api_key: Optional[str]
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: float = 30.0
    max_retries: int = 3
    extra_params: Dict[str, Any] = {}
```

### KnowledgeConfig

Configuration for knowledge base settings.

```python
class KnowledgeConfig(BaseModel):
    provider: str  # "faiss", "opensearch", "memory"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.7
```

### ExecutorConfig

Configuration for code execution settings.

```python
class ExecutorConfig(BaseModel):
    provider: str  # "inmemory", "api", "docker"
    database_type: str = "duckdb"
    max_execution_time: float = 30.0
    max_memory_mb: int = 512
    max_rows: int = 10000
    allow_file_access: bool = False
    allowed_imports: List[str] = []
    blocked_functions: List[str] = []
```

### WorkflowConfig

Configuration for agent workflow behavior.

```python
class WorkflowConfig(BaseModel):
    strategy: str = "react"  # "react", "workflow", "plan_execute"
    max_iterations: int = 10
    require_approval: bool = True
    auto_approve_safe: bool = False
    conversation_memory: bool = True
    max_context_length: int = 4000
    enable_visualization: bool = True
    debug_mode: bool = False
```

## Factory Functions

### create_agent

Create an agent with simplified configuration.

```python
def create_agent(
    name: str,
    config: Optional[Union[Dict, AgentConfig, str]] = None,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> DataAgent
```

**Parameters:**
- `name` (str): Agent name
- `config` (Optional[Union[Dict, AgentConfig, str]]): Configuration object
- `config_path` (Optional[Union[str, Path]]): Path to configuration file
- `**kwargs`: Additional configuration parameters

**Returns:**
- `DataAgent`: Created agent

**Example:**
```python
# From file
agent = create_agent("my-agent", config_path="config.yaml")

# From dict
agent = create_agent("my-agent", config={
    "llm": {"provider": "openai", "model": "gpt-3.5-turbo"}
})

# With kwargs
agent = create_agent("my-agent", 
                    llm={"provider": "openai", "model": "gpt-4"},
                    workflow={"require_approval": False})
```

### create_agent_async

Async version of create_agent.

```python
async def create_agent_async(
    name: str,
    config: Optional[Union[Dict, AgentConfig, str]] = None,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> DataAgent
```

### quick_query

Execute a single query with minimal setup.

```python
def quick_query(
    query: str,
    config_path: Optional[Union[str, Path]] = None,
    agent_name: str = "quick-agent",
    **config_kwargs
) -> str
```

**Parameters:**
- `query` (str): Natural language query
- `config_path` (Optional[Union[str, Path]]): Configuration file path
- `agent_name` (str): Temporary agent name
- `**config_kwargs`: Configuration parameters

**Returns:**
- `str`: Query response

**Example:**
```python
response = quick_query(
    "What's our total revenue?",
    config_path="config/basic_agent.yaml"
)
```

## Context Managers

### agent_session

Context manager for automatic agent cleanup.

```python
@contextmanager
def agent_session(
    name: str,
    config: Optional[Union[Dict, AgentConfig, str]] = None,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> DataAgent
```

**Example:**
```python
from dataqa import agent_session

with agent_session("temp-agent", config_path="config.yaml") as agent:
    response = agent.query("Analyze sales data")
    print(response)
# Agent automatically cleaned up
```

## Async API

All main functions have async equivalents:

- `create_agent_async()`
- `quick_query_async()`
- `DataAgent.query_async()`
- `DataQAClient.create_agent_async()`
- `DataQAClient.query_async()`

**Example:**
```python
import asyncio
from dataqa import create_agent_async

async def main():
    agent = await create_agent_async("async-agent", config_path="config.yaml")
    try:
        response = await agent.query_async("Show me sales trends")
        print(response)
    finally:
        await agent.shutdown_async()

asyncio.run(main())
```

## Exception Classes

### DataQAError

Base exception class for all DataQA errors.

```python
class DataQAError(Exception):
    pass
```

### ConfigurationError

Raised when configuration is invalid.

```python
class ConfigurationError(DataQAError):
    pass
```

### LLMError

Raised when LLM operations fail.

```python
class LLMError(DataQAError):
    pass
```

### ExecutionError

Raised when code execution fails.

```python
class ExecutionError(DataQAError):
    pass
```

### KnowledgeError

Raised when knowledge operations fail.

```python
class KnowledgeError(DataQAError):
    pass
```

## Environment Variables

DataQA supports the following environment variables:

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `DATAQA_LOG_LEVEL`: Global log level (DEBUG, INFO, WARNING, ERROR)
- `DATAQA_DATA_DIR`: Default data directory
- `DATAQA_CACHE_DIR`: Default cache directory
- `DATAQA_CONFIG_DIR`: Default configuration directory

## Type Hints

DataQA provides comprehensive type hints for better IDE support:

```python
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from dataqa.types import (
    AgentResponse,
    QueryResult,
    KnowledgeDocument,
    ExecutionResult
)
```

## Best Practices

### Configuration Management

```python
# Use environment variables for secrets
config = {
    "llm": {
        "api_key": "${OPENAI_API_KEY}"  # Environment variable substitution
    }
}

# Validate configuration early
try:
    config = AgentConfig.from_dict(config_dict)
except ValidationError as e:
    print(f"Configuration error: {e}")
```

### Resource Management

```python
# Always use context managers or explicit cleanup
with agent_session("my-agent", config_path="config.yaml") as agent:
    response = agent.query("Analyze data")

# Or explicit cleanup
agent = create_agent("my-agent", config_path="config.yaml")
try:
    response = agent.query("Analyze data")
finally:
    agent.shutdown()
```

### Error Handling

```python
from dataqa.exceptions import ExecutionError, LLMError

try:
    response = agent.query("Complex analysis query")
except ExecutionError as e:
    print(f"Execution failed: {e}")
except LLMError as e:
    print(f"LLM request failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```