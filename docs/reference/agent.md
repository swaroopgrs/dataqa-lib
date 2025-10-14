# Agent API Reference

This section documents the main DataQA agent class (`CWD`Agent`), its state, and its core workers.

---

## CWD`Agent`

The `CWD`Agent` is the core conversational agent in DataQA.
It implements a plan-and-execute loop, using LLMs and modular workers to answer complex data questions.

::: dataqa.core.agent.cwd_agent.CwdAgent

**Example Usage:**
```python
from dataqa.integrations.local.client import LocalClient
from dataqa.core.client import CoreRequest

client = LocalClient(config_path="path/to/agent.yaml")
request = CoreRequest(user_query="Show me sales by region", conversation_id="demo-1")
response = await client.process_query(request)
print(response.text)
```

---

## Agent State

The `CWD`State` object tracks the agent's progress, including the plan, completed steps, and generated data.

::: dataqa.core.agent.cwd_agent.CwdState

---

## Workers

Workers are specialized components responsible for executing tasks in the agent's plan.

### RetrievalWorker

Retrieves data from the database by generating and executing SQL queries.

::: dataqa.core.components.plan_execute.retrieval_worker.RetrievalWorker

### AnalyticsWorker

Performs data analysis on retrieved data (e.g., aggregations, statistics).

::: dataqa.core.components.plan_execute.analytics_worker.AnalyticsWorker

### PlotWorker

Generates plots and visualizations from dataframes.

::: dataqa.core.components.plan_execute.plot_worker.PlotWorker

---

## See Also

- [Building Your First Agent](../guide/building_your_first_agent.md)
- [Customizing Agents](../guide/customizing_agents.md)
- [API Reference: Components](components.md)