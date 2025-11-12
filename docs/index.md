# DataQA: The Conversational Data Agent

A powerful, config-driven framework for building intelligent agents that answer natural language questions about your data.

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

---

## What is the DataQA CWD Agent?

The **DataQA CWD Agent** is an intelligent system designed to understand your data and answer complex questions. It follows a "Plan, Work, Dispatch" loop to break down your query, retrieve data using SQL, perform analysis, and deliver a clear, accurate answer.

Instead of writing complex code, you teach the agent about your data through simple YAML files.

## Who is this for?

- **Data Analysts & Scientists** who want to enable natural language querying for their datasets.
- **Teams** that need a robust, reliable way to build and maintain a conversational data assistant.
- **Developers** creating data-centric applications without wanting to manage complex agentic loops.

---

## Key Features

- **Conversational CWD Agent:** A powerful plan-and-execute loop for tackling multi-step questions.
- **Knowledge-Driven:** Define your data's schema, business rules, and query examples in simple YAML files.
- **Config, Not Code:** Most agent behavior is defined in a central `agent.yaml` file.
- **Built-in Tools:** Comes with tools for SQL generation, data analysis (Pandas), and plotting out-of-the-box.
- **Local First:** Designed for easy local setup and testing before deployment.

---

## Get Started

<div class="grid cards" markdown>

-   **Quickstart Guide**

    ---

    Get your first agent running in 5 minutes. This is the best place to start.

    [:octicons-arrow-right-24: Quickstart Guide](quickstart.md)

-   **User Guide**

    ---

    Learn the core concepts and master the art of configuring your agent's knowledge base.

    [:octicons-arrow-right-24: User Guide](guide/introduction.md)

-   **Building Assets**

    ---

    Deep dive into creating the `schema.yml`, `rules.yml`, and `examples.yml` files that power your agent.

    [:octicons-arrow-right-24: Building Assets](guide/building_assets.md)

-   **Configuration Reference**

    ---

    Detailed reference for the `agent.yaml` configuration file.

    [:octicons-arrow-right-24: Configuration Reference](reference/agent_config.md)

</div>

---

## Community & Support

- [Troubleshooting Guide](guide/troubleshooting.md)
- [FAQ](guide/faq.md)
