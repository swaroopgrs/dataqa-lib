# Getting Started with DataQA

Welcome to DataQA! This guide will help you get up and running with the DataQA framework for building intelligent data agents.

## What is DataQA?

DataQA is a composable data agent framework that enables natural language interaction with structured data. It combines:

- **Intelligent Grounding (RAG)**: Context-aware code generation with business rules and schema knowledge
- **Secure Execution**: Sandboxed code execution with pluggable backends  
- **Advanced Agentic Framework**: Composable agent strategies with hierarchical capabilities
- **Interactive & Stateful**: Persistent memory and human-in-the-loop interactions
- **Declarative Configuration**: YAML-driven agent definition

## Installation

### Basic Installation

```bash
pip install dataqa
```

### Development Installation

For development with all optional dependencies:

```bash
pip install dataqa[dev,opensearch]
```

### From Source

```bash
git clone https://github.com/your-org/dataqa.git
cd dataqa
pip install -e .
```

## Quick Start

### 1. Set Up Your Environment

First, set up your environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 2. Create Your First Agent

Create a simple configuration file `my_agent.yaml`:

```yaml
name: "my-first-agent"
description: "My first DataQA agent"

llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.1

knowledge:
  provider: "faiss"
  embedding_model: "all-MiniLM-L6-v2"

executor:
  provider: "inmemory"
  database_type: "duckdb"

workflow:
  require_approval: true
  enable_visualization: true
```

### 3. Use the Python API

```python
from dataqa import create_agent

# Create agent from configuration
agent = create_agent("my-first-agent", config_path="my_agent.yaml")

try:
    # Query your data
    response = agent.query("What are the top 5 customers by revenue?")
    print(response)
finally:
    # Clean up
    agent.shutdown()
```

### 4. Use the CLI

```bash
# Run interactive queries
dataqa run --config my_agent.yaml

# Or run a single query
dataqa query --config my_agent.yaml "Show me sales trends for the last quarter"
```

## Core Concepts

### Agents

Agents are the main interface for interacting with your data. They combine:
- **LLM**: For understanding natural language and generating code
- **Knowledge Base**: For storing business context and schema information
- **Executor**: For safely running generated code
- **Workflow**: For orchestrating the interaction flow

### Configuration

DataQA uses YAML configuration files to define agent behavior:

```yaml
name: "agent-name"
description: "Agent description"

# LLM settings
llm:
  provider: "openai"  # or "anthropic", "local"
  model: "gpt-4"
  api_key: "${API_KEY}"

# Knowledge base settings  
knowledge:
  provider: "faiss"  # or "opensearch", "memory"
  embedding_model: "all-MiniLM-L6-v2"

# Code execution settings
executor:
  provider: "inmemory"  # or "api", "docker"
  database_type: "duckdb"

# Workflow settings
workflow:
  strategy: "react"  # or "workflow", "plan_execute"
  require_approval: true
```

### Knowledge Management

Add business context and schema information to improve agent accuracy:

```python
from dataqa import Document

# Create knowledge documents
documents = [
    Document(
        content="Sales data is in the sales_transactions table with columns: id, date, product_id, customer_id, amount",
        metadata={"type": "schema", "table": "sales_transactions"},
        source="schema.md"
    ),
    Document(
        content="Customer segments: Premium (>$10k), Standard ($1k-$10k), Basic (<$1k)",
        metadata={"type": "business_rule"},
        source="business_rules.md"
    )
]

# Ingest knowledge
agent.ingest_knowledge(documents)
```

## Common Usage Patterns

### 1. Basic Data Querying

```python
from dataqa import create_agent

agent = create_agent("data-agent", config_path="config/basic_agent.yaml")

# Simple queries
response = agent.query("How many customers do we have?")
response = agent.query("What's our average order value?")
response = agent.query("Show me sales by month")
```

### 2. Conversational Analysis

```python
# Start a conversation
agent.query("Load our sales data for analysis")

# Follow-up questions maintain context
agent.query("What are the top products?")
agent.query("Show me trends for those products")
agent.query("Create a chart comparing them")
```

### 3. Approval Workflows

```python
# Query that requires approval
response = agent.query("Update customer segments based on recent purchases")

# Check if approval is needed
if agent.has_pending_approval():
    print("Operation requires approval:")
    print(agent.get_pending_operation())
    
    # Approve or reject
    agent.approve_operation(approved=True)
```

### 4. Async Operations

```python
import asyncio
from dataqa import create_agent_async

async def analyze_data():
    agent = await create_agent_async("async-agent", config_path="config/basic_agent.yaml")
    
    try:
        # Process multiple queries concurrently
        queries = [
            "What are our top products?",
            "Show me customer segments",
            "Analyze sales trends"
        ]
        
        tasks = [agent.query(q) for q in queries]
        responses = await asyncio.gather(*tasks)
        
        for query, response in zip(queries, responses):
            print(f"Q: {query}")
            print(f"A: {response}\n")
    finally:
        await agent.shutdown()

# Run async analysis
asyncio.run(analyze_data())
```

## Configuration Examples

### Basic Configuration

Use `config/basic_agent.yaml` for simple use cases:

```yaml
name: "basic-agent"
llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
knowledge:
  provider: "faiss"
executor:
  provider: "inmemory"
workflow:
  require_approval: true
```

### Development Configuration

Use `config/development_agent.yaml` for development:

```yaml
name: "dev-agent"
llm:
  model: "gpt-3.5-turbo"
  temperature: 0.2
executor:
  allow_file_access: true
workflow:
  require_approval: false
  debug_mode: true
log_level: "DEBUG"
```

### Enterprise Configuration

Use `config/enterprise_agent.yaml` for production:

```yaml
name: "enterprise-agent"
llm:
  model: "gpt-4"
  temperature: 0.05
knowledge:
  provider: "opensearch"
executor:
  provider: "api"
  max_memory_mb: 2048
workflow:
  strategy: "plan_execute"
  require_approval: true
```

## Next Steps

- Read the [API Reference](api-reference.md) for detailed documentation
- Check out [Configuration Guide](configuration.md) for advanced settings
- See [Examples](examples/) for more usage patterns
- Review [Troubleshooting](troubleshooting.md) for common issues

## Need Help?

- Check the [Troubleshooting Guide](troubleshooting.md)
- Review [Frequently Asked Questions](faq.md)
- Open an issue on [GitHub](https://github.com/your-org/dataqa/issues)
- Join our [Discord community](https://discord.gg/dataqa)