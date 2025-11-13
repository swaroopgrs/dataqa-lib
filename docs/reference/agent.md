# Agent API Reference

Detailed documentation for the CWDAgent class, state management, and worker components.

---

## CWDAgent

The `CWDAgent` is the core conversational agent in DataQA. It implements a plan-and-execute loop, using LLMs and modular workers to answer complex data questions.

**Location:** `dataqa.core.agent.cwd_agent.cwd_agent.CWDAgent`

### Architecture

The CWDAgent follows a **Plan-Work-Dispatch** pattern:

1. **Planner**: Creates a step-by-step plan from the user's query
2. **Workers**: Execute tasks (RetrievalWorker, AnalyticsWorker, PlotWorker)
3. **Replanner**: Evaluates results and updates the plan

### Component LLM Assignment

The agent supports assigning different LLMs to different components:

- `default`: Default LLM for components without specific assignment
- `planner`: Creates the initial plan
- `replanner`: Updates the plan after each task
- `retrieval_worker`: Generates and executes SQL queries
- `analytics_worker`: Performs data analysis using Pandas
- `plot_worker`: Generates visualizations

### Direct Usage

**Note**: In practice, you typically use `LocalClient` or `DBCClient` (LLMSuite Database Connect) rather than instantiating `CWDAgent` directly. The clients handle agent creation, configuration, and memory management.

If you need direct access:

```python
from dataqa.core.agent.cwd_agent.cwd_agent import CWDAgent
from dataqa.core.agent.cwd_agent.config import CwdAgentDefinitionConfig

# Agent is typically created via factory
# See LocalAgentFactory or DBC_CWDAgentFactory
```

---

## CWDState

The `CWDState` object tracks the agent's progress through a query execution.

**Location:** `dataqa.core.agent.cwd_agent.cwd_agent.CWDState`

### Key Fields

- **`query`** (str): The user's original question
- **`plan`** (List[Task]): The current plan with list of tasks
- **`worker_response`** (WorkerResponse): Results from executed tasks
- **`output_dataframes`** (List[str]): Names of DataFrames generated during execution
- **`final_answer`** (str): The agent's final response text
- **`history`** (List[str]): Conversation history

### State Flow

The state flows through the agent workflow:
1. Initialized with user query
2. Updated by Planner with initial plan
3. Updated by Workers with task results
4. Updated by Replanner with plan modifications
5. Finalized with answer text

---

## Workers

Workers are specialized components that execute tasks in the agent's plan.

### RetrievalWorker

Retrieves data from the database by generating and executing SQL queries.

**Location:** `dataqa.core.components.plan_execute.retrieval_worker.RetrievalWorker`

**Key Responsibilities:**
- Generates SQL from natural language queries using schema, rules, and examples
- Executes SQL against the configured data source (DuckDB, PySpark, etc.)
- Returns DataFrames with query results
- Handles SQL errors and retries

**Configuration:**
- Configured via `workers.retrieval_worker.sql_execution_config` in `agent.yaml`
- Uses knowledge assets (schema, rules, examples) for SQL generation

---

### AnalyticsWorker

Performs data analysis on retrieved data using Pandas operations.

**Location:** `dataqa.core.components.plan_execute.analytics_worker.AnalyticsWorker`

**Key Responsibilities:**
- Performs aggregations, statistics, and calculations
- Uses built-in analytics tools (summarize, groupby, calculate correlations, etc.)
- Operates on DataFrames from previous tasks
- Returns analysis results as DataFrames or text

**Available Tools:**
- Data summarization
- Statistical calculations
- Data transformations
- Filtering and grouping operations

---

### PlotWorker

Generates plots and visualizations from dataframes.

**Location:** `dataqa.core.components.plan_execute.plot_worker.PlotWorker`

**Key Responsibilities:**
- Creates charts and graphs from data
- Uses matplotlib and seaborn for visualization
- Returns image files or plot objects
- Supports various chart types (line, bar, scatter, heatmap, etc.)

**Available Tools:**
- Line plots
- Bar charts
- Scatter plots
- Heatmaps
- Custom visualizations

---

## See Also

- [API Reference Overview](index.md): Complete API documentation including clients and data models
- [Agent Configuration](agent_config.md): Configuration reference
- [Create Knowledge Assets Manually](../guide/creating_knowledge_assets.md): Learn about asset files

