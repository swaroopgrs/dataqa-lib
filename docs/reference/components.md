# Components API Reference

Components are the reusable building blocks for constructing DataQA agents and pipelines.
This section documents the core component classes, their configuration, and extension points.

---

## Base Classes

All components inherit from these base classes.

### Component

The abstract base class for all DataQA components.

::: dataqa.core.components.base_component.Component

### ComponentConfig

Base configuration class for components.

::: dataqa.core.components.base_component.ComponentConfig

---

## LLM Components

Components for interacting with Large Language Models.

### BasePromptLLMChain

A component for prompt-based LLM chains.

::: dataqa.core.components.llm_component.base_prompt_llm_chain.BasePromptLLMChain

---

## Code Execution

### InMemoryCodeExecutor

Executes SQL or Python code in-memory using DuckDB or PySpark.

::: dataqa.core.components.code_executor.in_memory_code_executor.InMemoryCodeExecutor

---

## Planning Components

Components related to the plan-and-execute agent loop.

### Planner

Generates a step-by-step plan for answering a user query.

::: dataqa.core.components.plan_execute.planner.Planner

### Replanner

Evaluates progress and updates the plan as needed.

::: dataqa.core.components.plan_execute.replanner.Replanner

---

## Analytics & Plotting Tools

See [Tools API Reference](tools.md) for available analytics and plotting tools.

---

## See Also

- [Agent API Reference](agent.md)
- [Tools API Reference](tools.md)
