# Agent Configuration Reference

Complete reference for the `agent.yaml` configuration file.

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
| `timeout` | integer | No | Timeout in seconds for a single query execution. Defaults to 300. |

---

## `llm_configs`

Define one or more LLM connections.

```yaml
llm_configs:
  default_llm:
    type: "dataqa.llm.openai.AzureOpenAI"
    config:
      model: "gpt-4o-2024-08-06"
      api_version: "2024-08-01-preview"
      api_type: "azure_ad"
      temperature: 0
```

**Supported LLM Type:**
- `dataqa.llm.openai.AzureOpenAI` - Azure OpenAI service

---

## `llm`

Maps LLMs to agent components.

```yaml
llm:
  default: default_llm
  planner: default_llm
  replanner: default_llm
  retrieval_worker: default_llm
  analytics_worker: default_llm
  plot_worker: default_llm
```

---

## `resource_manager_config`

Tells the agent where to load asset files.

```yaml
resource_manager_config:
  type: "dataqa.core.components.resource_manager.resource_manager.ResourceManager"
  config:
    asset_directory: "<CONFIG_DIR>/data/"
```

**Special Placeholder:**
- `<CONFIG_DIR>`: Automatically replaced with the directory containing your `agent.yaml` file.

---

## `retriever_config`

Configures how the agent retrieves knowledge.

```yaml
retriever_config:
  type: dataqa.core.components.retriever.base_retriever.AllRetriever
  config:
    name: all_retriever
    retrieval_method: "all"
    resource_types: ["rule", "schema", "example"]
    module_names: ["planner", "retrieval_worker"]
```

---

## `workers`

Configures execution backends.

```yaml
workers:
  retrieval_worker:
    sql_execution_config:
      name: "sql_executor"
      data_files:
        - path: "<CONFIG_DIR>/data/my_data.csv"
          table_name: my_table
```

---

## `dialect`

Specifies SQL dialect and functions.

```yaml
dialect:
  value: "sqlite"
  functions: |
    - name: STRFTIME(format, timestring)
      example: STRFTIME('%Y', date_column) = '2024'
```

---

## Complete Example

See [Quickstart](../quickstart.md) for a complete working example.

