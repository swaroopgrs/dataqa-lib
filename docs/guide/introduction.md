# Introduction

Welcome to the DataQA User Guide!
This guide introduces the core concepts, architecture, and main workflows of DataQA.

---

## What is DataQA?

**DataQA** is a modular Python framework for building intelligent data agents and pipelines that can understand, analyze, and respond to natural language queries about your data. It is designed for both rapid prototyping and production-grade conversational analytics.

---

## Core Concepts

DataQA is built around two primary architectural patterns:

### 1. Conversational Agent (`CWD`Agent`)

The "CWD`Agent`" is designed for complex, multi-step conversational data analysis.
It follows a **Plan-and-Execute** pattern:
- **Planner:** Receives the user's query and creates a step-by-step plan of tasks.
- **Workers:** Executes each task. The library includes:
  - `RetrievalWorker`: Fetches data by generating and running SQL queries.
  - `AnalyticsWorker`: Performs data analysis using tools like Pandas.
  - `PlotWorker`: Generates charts and graphs.
- **Replanner:** After each task, reviews progress, decides if the goal is met, and updates the plan if necessary.

This architecture allows the agent to handle ambiguity, recover from errors, and tackle problems that require multiple intermediate steps.

**Use this when:**
- You need an autonomous system that can reason about a user's goal and dynamically decide what to do next.
- You need a conversational, autonomous system that can reason about a user's goal and dynamically decide what to do next.

### 2. Pipeline Engine

The pipeline engine is for building structured, directed acyclic graphs (DAGs) of operations.
Each node in the graph is a **Component**.
- **Components:** Reusable, single-purpose building blocks with clearly defined inputs and outputs for interacting with LLMs, executing code, retrieving data, and more.
- **Graph Definition:** You define a pipeline in a YAML file by listing the components and specifying how they connect to each other.
- **Execution:** The pipeline executes in a predictable, repeatable sequence defined by the graph structure.

**Use this when:**
- You have a well-defined workflow that doesn't require conversational reasoning.

---

## Architecture Overview

![DataQA Architectural](../resources/dataqa_overview.png "DataQA Architecture")

### Example: Default Agent Workflow

![DataQA Default Agent](../resources/dataqa_default_agent.png "Default Agent")

---

## Key Features

- **Config-Driven:** Define agent and pipeline behavior in YAML, not code.
- **Extensible:** Add your own tools, prompts, and components.
- **Integrated Benchmarking:** Evaluate accuracy and performance.
- **Comprehensive Tooling:** Built-in SQL, Pandas, and plotting tools.

---

## Next Steps

- [Running the Examples](running_examples.md)
- [Building Your First Agent](building_your_first_agent.md)
- [API Reference](../reference/agent.md)
