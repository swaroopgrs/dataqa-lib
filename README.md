# DataQA

A composable data agent framework for natural language data interaction.

## Overview

DataQA is an enterprise-grade Python framework that enables developers to build intelligent data agents capable of querying databases, performing analysis, and generating visualizations through natural language interfaces.

## Features

- **Intelligent Grounding (RAG)**: Context-aware code generation with business rules and schema knowledge
- **Secure Execution**: Sandboxed code execution with pluggable backends
- **Advanced Agentic Framework**: Composable agent strategies with hierarchical capabilities
- **Interactive & Stateful**: Persistent memory and human-in-the-loop interactions
- **Declarative Configuration**: YAML-driven agent definition

## Installation

```bash
pip install dataqa
```

For development with optional dependencies:

```bash
pip install dataqa[dev,opensearch]
```

## Quick Start

```python
from dataqa import DataAgent, AgentConfig

# Load configuration
config = AgentConfig.from_yaml("agent_config.yaml")

# Create agent
agent = DataAgent(config)

# Query your data
response = agent.query("What are the top 5 customers by revenue?")
print(response)
```

## Documentation

### Quick Links
- [Getting Started Guide](docs/getting-started.md) - Complete setup and usage guide
- [API Reference](docs/api-reference.md) - Comprehensive API documentation
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

### Configuration Examples
- [Basic Agent](config/basic_agent.yaml) - Simple configuration for getting started
- [Development Agent](config/development_agent.yaml) - Development-optimized settings
- [Enterprise Agent](config/enterprise_agent.yaml) - Production-ready configuration
- [Analytics Agent](config/analytics_agent.yaml) - Specialized for data analytics

### Usage Examples
- [API Usage Examples](examples/api_usage.py) - Python API usage patterns
- [End-to-End Examples](examples/end_to_end_examples.py) - Complete workflow demonstrations

## License

MIT License - see LICENSE file for details.