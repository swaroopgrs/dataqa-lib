# DataQA: The Conversational Data Agent

A powerful, config-driven framework for building intelligent agents that answer natural language questions about your data.

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

---

## What is DataQA?

The **DataQA CWD Agent** is an intelligent system designed to understand your data and answer complex questions. It follows a "Plan, Work, Dispatch" loop to break down your query, retrieve data using SQL, perform analysis, and deliver a clear, accurate answer.

Instead of writing complex code, you teach the agent about your data through simple YAML files: `schema.yml`, `rules.yml`, and `examples.yml`.

## Who is this for?

- **Data Analysts & Scientists** who want to enable natural language querying for their datasets
- **Teams** that need a robust, reliable way to build and maintain a conversational data assistant
- **Developers** creating data-centric applications without wanting to manage complex agentic loops

---

## Key Features

- **Conversational CWD Agent:** A powerful plan-and-execute loop for tackling multi-step questions
- **Knowledge-Driven:** Define your data's schema, business rules, and query examples in simple YAML files
- **Config, Not Code:** Most agent behavior is defined in a central `agent.yaml` file
- **Built-in Tools:** Comes with tools for SQL generation, data analysis (Pandas), and plotting out-of-the-box
- **Knowledge Asset Tools:** Use DataScanner and Rule Inference to generate and enhance knowledge assets

---

## Get Started

<div class="grid cards" markdown>

-   **Quickstart**

    ---

    Get your first agent running in minutes. Includes installation and setup.

    [:octicons-arrow-right-24: Quickstart Guide](quickstart.md)

-   **User Guide**

    ---

    Learn how to build agents, create knowledge assets, and evaluate performance.

    [:octicons-arrow-right-24: User Guide](guide/understanding_the_cwd_agent.md)

-   **API Reference**

    ---

    Detailed reference for configuration files and API.

    [:octicons-arrow-right-24: API Reference](reference/index.md)

</div>

---

## Quick Example

```python
from dataqa.integrations.local.client import LocalClient
from dataqa.core.client import CoreRequest

client = LocalClient(config_path="agent.yaml")
request = CoreRequest(
    user_query="What was the total revenue for the North region?",
    conversation_id="demo-1"
)
response = await client.process_query(request)
print(response.text)
```

---

## Next Steps

- **[Quickstart](quickstart.md)**: Get started in minutes
- **[Knowledge Asset Tools](guide/knowledge_asset_tools.md)**: Generate and enhance assets with DataScanner and Rule Inference
- **[Create Knowledge Assets](guide/creating_knowledge_assets.md)**: Learn how to create schema, rules, and examples
- **[User Guide](guide/understanding_the_cwd_agent.md)**: Deep dive into the CWD Agent

