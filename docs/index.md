# DataQA Library

A modular, production-grade Python framework for building, composing, and orchestrating intelligent data agents.

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

---

## What is DataQA?

**DataQA** is a declarative, config-driven library for building intelligent agents that can understand, analyze, and respond to natural language queries and data scientists who want to create conversational analytics, automated data pipelines, or custom data assistants.

## Who is DataQA for?

- **Data scientists** and **ML engineers** building LLM-powered analytics.
- **Developers** creating conversational data tools or chatbots.
- **Teams** needing robust, extensible data agent frameworks for structured data.

---

## Key Features

- **Conversational AI Agent (`CWD`Agent`):** Plan-and-execute loop for complex queries.
- **Modular Pipeline Engine:** Compose custom workflows from reusable components.
- **Comprehensive Tooling:** Built-in SQL, Pandas, and plotting tools.
- **Config-Driven:** Define agent behavior in YAML, not code.
- **Extensible:** Add your own tools, prompts, and components.
- **Integrated Benchmarking:** Evaluate accuracy and performance.

---

## Quickstart

```python
from dataqa.integrations.local.client import LocalClient
from dataqa.core.client import CoreRequest

client = LocalClient(config_path="path/to/your/agent.yaml")
request = CoreRequest(user_query="show me sales by region", conversation_id="demo-1")
response = await client.process_query(request)
print(response.text)
```

See the [Quickstart Guide](quickstart.md) for a step-by-step walkthrough.

---

# Architecture Overview

![DataQA Architecture](resources/dataqa_overview.png "DataQA Architecture")

---

# Get Started

<div class="grid cards" markdown>

-   **Quickstart**

    ---

    The fastest way to get a DataQA agent running.

    [:octicons-arrow-right-24: Quickstart Guide](quickstart.md)

-   **User Guide**

    ---

    Learn how to install the library, run the included examples, and understand the core concepts.

    [:octicons-arrow-right-24: User Guide](guide/introduction.md)

-   **API Reference**

    ---

    Explore the detailed API documentation for all classes, methods, and components.

    [:octicons-arrow-right-24: API Reference](reference/agent.md)

-   **Contributing**

    ---

    Learn how to contribute to the library.

    [:octicons-arrow-right-24: Developer Guide](contributing.md)

</div>

---

## Community & Support

- [FAQ](guide/faq.md)
- [TODO] Add a mailing list
