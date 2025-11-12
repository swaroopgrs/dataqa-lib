# Reference: `agent.yaml` Configuration

This document provides a detailed reference for all the settings available in the main `agent.yaml` configuration file for the CWD Agent.

---

## Top-Level Fields

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `agent_name` | string | No | An optional name for your agent configuration. Defaults to "CwdAgent". |
| `llm_configs` | object | Yes | A dictionary where you define one or more LLM connections. |
| `llm` | object | Yes | Maps the defined LLMs from `llm_configs` to specific agent components. |
| `resource_manager_config` | object | Yes | Configures where the agent finds its knowledge assets (`schema.yml`, etc.). |
| `retriever_config` | object | Yes | Defines how the agent retrieves and uses the knowledge assets for prompts. |
| `workers` | object | Yes | Configures the execution environments for workers, especially the SQL executor. |
| `use_case_name` | string | Yes | A short name for your use case, used in prompts for context. |
| `use_case_description` | string | Yes | A detailed description of what the agent does, also used in prompts. |
| `dialect` | object | Yes | Specifies the SQL dialect and available functions for SQL generation. |
| `max_tasks` | integer | No | The maximum number of tasks an agent can execute for a single query. Defaults to 10. |
| `max_react_recursion` | integer | No | The maximum number of recursions for react workers. Defaults to 10. |
| `timeout` | integer | No | Timeout in seconds for a single query execution. Defaults to 300. |

---

## `llm_configs`

This section is a dictionary where each key is a friendly name for an LLM configuration.

```yaml
llm_configs:
  # A friendly name you choose, e.g., "gpt4_creative" or "default_llm"
  my_default_llm:
    # The full Python path to the LLM implementation class.
    type: "dataqa.llm.openai.AzureOpenAI"
    # Configuration specific to the LLM class.
    config:
      model: "gpt-4o-2024-08-06"  # Your Azure deployment name
      api_version: "2024-08-01-preview"
      api_type: "azure_ad"
      temperature: 0
```

**Supported LLM Types:**
- `dataqa.llm.openai.AzureOpenAI` - Azure OpenAI service
- `dataqa.llm.gemini.GeminiLLM` - Google Gemini

**Common Config Fields:**
- `model`: The model deployment name (for Azure) or model identifier
- `api_version`: API version string (for Azure)
- `api_type`: Authentication type, typically `"azure_ad"` for Azure
- `temperature`: LLM temperature (0-2), typically 0 for deterministic behavior

---

## `llm`

This section maps the LLMs defined in `llm_configs` to the agent's internal components. This allows you to use different models for different purposes (e.g., a powerful model for planning and a cheaper one for summarization).

```yaml
llm:
  # The LLM to use if a component-specific one is not set. REQUIRED.
  default: my_default_llm
  # Optional overrides for specific components.
  planner: my_powerful_llm
  replanner: my_default_llm
  retrieval_worker: my_default_llm
  analytics_worker: my_default_llm
  plot_worker: my_default_llm
```

**Available Components:**
- `planner`: Creates the initial plan from user queries
- `replanner`: Updates the plan after each task execution
- `retrieval_worker`: Generates and executes SQL queries
- `analytics_worker`: Performs data analysis using Pandas
- `plot_worker`: Generates visualizations

---

## `resource_manager_config`

This tells the agent where to load the asset files from.

```yaml
resource_manager_config:
  type: "dataqa.core.components.resource_manager.resource_manager.ResourceManager"
  config:
    # Path to the directory containing schema.yml, rules.yml, etc.
    # <CONFIG_DIR> is a placeholder for the directory of this agent.yaml file.
    asset_directory: "<CONFIG_DIR>/data/"
```

**Special Placeholder:**
- `<CONFIG_DIR>`: Automatically replaced with the absolute path to the directory containing your `agent.yaml` file. This makes your configuration portable.

---

## `retriever_config`

This configures how the agent should retrieve knowledge to build its prompts. For most users, the `AllRetriever` is recommended.

```yaml
retriever_config:
  # Use the AllRetriever to load all assets into the context.
  type: dataqa.core.components.retriever.base_retriever.AllRetriever
  config:
    name: all_retriever
    retrieval_method: "all"
    # Which asset types to load.
    resource_types: ["rule", "schema", "example"]
    # Which components will receive these assets in their prompts.
    module_names: ["planner", "retrieval_worker", "analytics_worker"]
```

**Resource Types:**
- `"schema"`: Loads `schema.yml` files
- `"rule"`: Loads `rules.yml` files
- `"example"`: Loads `examples.yml` files

**Module Names:**
- `"planner"`: The planning component
- `"replanner"`: The replanning component
- `"retrieval_worker"`: The SQL generation component
- `"analytics_worker"`: The analytics component
- `"plot_worker"`: The plotting component

---

## `workers`

This section configures the execution backend for the workers. The most important part is the `sql_execution_config`.

```yaml
workers:
  retrieval_worker:
    # This configures the in-memory SQL engine (DuckDB).
    sql_execution_config:
      name: "sql_executor"
      # A list of data files to load into the in-memory database.
      data_files:
        - path: "<CONFIG_DIR>/data/my_data.csv"
          # The table name to use in SQL queries. MUST match a table_name in schema.yml.
          table_name: my_first_table
        - path: "<CONFIG_DIR>/data/more_data.csv"
          table_name: my_second_table
  analytics_worker: {}  # Optional, uses defaults
  plot_worker: {}       # Optional, uses defaults
```

**Important Notes:**
- The `table_name` in each `data_files` entry **must** match a `table_name` in your `schema.yml`
- Multiple CSV files can be loaded, and the agent can perform `JOIN`s between them
- The SQL executor uses DuckDB by default, which supports SQLite-compatible syntax

---

## `dialect`

This helps the SQL generator produce correct syntax for your target database.

```yaml
dialect:
  # E.g., "sqlite", "snowflake", "redshift"
  value: "sqlite"
  # Optional: A multi-line string listing custom functions available.
  functions: |
    - name: STRFTIME(format, timestring)
      example: STRFTIME('%Y', date_column) = '2024'
    - name: DATEADD(unit, value, date)
      example: DATEADD('day', -30, CURRENT_DATE)
```

**Common Dialect Values:**
- `"sqlite"`: SQLite syntax (used by DuckDB, the default executor)
- `"snowflake"`: Snowflake SQL syntax
- `"redshift"`: Amazon Redshift syntax
- `"postgresql"`: PostgreSQL syntax

---

## `use_case_name` and `use_case_description`

These provide human-readable context that is injected into the agent's prompts.

```yaml
use_case_name: "Sales Reporting"
use_case_description: |
  An agent that answers questions about sales performance from the sales_report table.
  It can calculate revenue, units sold, and performance metrics by region, product, and time period.
```

**Best Practices:**
- Keep `use_case_name` short and descriptive (1-3 words)
- Make `use_case_description` detailed enough to give the agent context about its purpose
- Mention key tables, metrics, and capabilities

---

## Complete Example

```yaml
agent_name: "SalesAgent"

llm_configs:
  default_llm:
    type: "dataqa.llm.openai.AzureOpenAI"
    config:
      model: "gpt-4o-2024-08-06"
      api_version: "2024-08-01-preview"
      api_type: "azure_ad"
      temperature: 0

llm:
  default: default_llm

resource_manager_config:
  type: "dataqa.core.components.resource_manager.resource_manager.ResourceManager"
  config:
    asset_directory: "<CONFIG_DIR>/data/"

retriever_config:
  type: dataqa.core.components.retriever.base_retriever.AllRetriever
  config:
    name: all_retriever
    retrieval_method: "all"
    resource_types: ["rule", "schema", "example"]
    module_names: ["planner", "retrieval_worker"]

workers:
  retrieval_worker:
    sql_execution_config:
      name: "sql_executor"
      data_files:
        - path: "<CONFIG_DIR>/data/sales_data.csv"
          table_name: sales_report

use_case_name: "Sales Reporting"
use_case_description: |
  An agent that answers questions about sales performance from the sales_report table.

dialect:
  value: "sqlite"

max_tasks: 10
timeout: 300
```

---

## See Also

- [Building Assets](../guide/building_assets.md): Learn how to create the asset files referenced in this configuration.
- [Quickstart](../quickstart.md): See a working example configuration.


