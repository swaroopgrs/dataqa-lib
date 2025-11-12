# Agent API Reference

This section documents the main DataQA agent class (CWDAgent), its state, and its core workers.

---

## CWDAgent

The `CWDAgent` is the core conversational agent in DataQA. It implements a plan-and-execute loop, using LLMs and modular workers to answer complex data questions.

### Overview

The CWDAgent follows a **Plan-Work-Dispatch** pattern:

1. **Planner**: Creates a step-by-step plan from the user's query
2. **Workers**: Execute tasks (RetrievalWorker, AnalyticsWorker, PlotWorker)
3. **Replanner**: Evaluates results and updates the plan

### Key Components

The agent has the following components that can be configured with different LLMs:

- `default`: Default LLM for components without specific assignment
- `planner`: Creates the initial plan
- `replanner`: Updates the plan after each task
- `retrieval_worker`: Generates and executes SQL queries
- `analytics_worker`: Performs data analysis using Pandas
- `plot_worker`: Generates visualizations

### Usage

```python
from dataqa.integrations.local.client import LocalClient
from dataqa.core.client import CoreRequest

client = LocalClient(config_path="path/to/agent.yaml")
request = CoreRequest(user_query="Show me sales by region", conversation_id="demo-1")
response = await client.process_query(request)
print(response.text)
```

**Note**: In practice, you typically use `LocalClient` or `DBCClient` rather than instantiating `CWDAgent` directly. The client handles agent creation and configuration.

---

## CWDState

The `CWDState` object tracks the agent's progress through a query execution, including:

- **query**: The user's original question
- **plan**: The current plan with list of tasks
- **worker_response**: Results from executed tasks
- **output_dataframes**: DataFrames generated during execution
- **final_answer**: The agent's final response text

The state is passed between components (planner → worker → replanner) and updated as the agent progresses.

---

## Workers

Workers are specialized components responsible for executing tasks in the agent's plan.

### RetrievalWorker

Retrieves data from the database by generating and executing SQL queries.

**Key Responsibilities:**
- Generates SQL from natural language queries
- Executes SQL against the configured data source
- Returns DataFrames with query results

**Configuration:**
- Configured via `workers.retrieval_worker.sql_execution_config` in `agent.yaml`
- Uses schema, rules, and examples from asset files to generate accurate SQL

### AnalyticsWorker

Performs data analysis on retrieved data using Pandas operations.

**Key Responsibilities:**
- Performs aggregations, statistics, and calculations
- Uses built-in analytics tools (e.g., `summarize_dataframe`, `calculate_correlation`)
- Operates on DataFrames from previous tasks

**Available Tools:**
- Data summarization
- Statistical calculations
- Data transformations
- Filtering and grouping operations

### PlotWorker

Generates plots and visualizations from dataframes.

**Key Responsibilities:**
- Creates charts and graphs from data
- Uses matplotlib and seaborn for visualization
- Returns image files or plot objects

**Available Tools:**
- Line plots
- Bar charts
- Scatter plots
- Heatmaps
- Custom visualizations

---

## See Also

- [Building Assets](../guide/building_assets.md)
- [Agent Configuration](agent_config.md)
