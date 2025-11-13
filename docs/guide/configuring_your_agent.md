# Configure Your Agent

This guide walks you through creating and configuring your own DataQA CWD Agent from scratch.

---

## Overview

A DataQA agent is defined by:
1. **Configuration file** (`agent.yaml`) - Defines the agent's behavior and setup
2. **Knowledge assets** - Three YAML files that teach the agent about your data:
   - `schema.yml` - Describes your data structure
   - `rules.yml` - Defines business logic
   - `examples.yml` - Provides query examples

---

## Project Structure

A typical agent project looks like this:

```
my-agent/
├── data/
│   ├── schema.yml      # Your data schema
│   ├── rules.yml       # Business rules (optional)
│   └── examples.yml    # Query examples (optional)
├── agent.yaml          # Agent configuration
└── run_agent.py        # Script to run queries
```

---

## Step 1: Create the Agent Configuration

Create `agent.yaml` in your project root:

```yaml
agent_name: "MyAgent"

# Define LLM connections
llm_configs:
  default_llm:
    type: "dataqa.llm.openai.AzureOpenAI"
    config:
      model: "gpt-4o-2024-08-06"
      api_version: "2024-08-01-preview"
      api_type: "azure_ad"
      temperature: 0

# Assign LLMs to components
llm:
  default: default_llm

# Where to find knowledge assets
resource_manager_config:
  type: "dataqa.core.components.resource_manager.resource_manager.ResourceManager"
  config:
    asset_directory: "<CONFIG_DIR>/data/"

# How to retrieve knowledge for prompts
retriever_config:
  type: dataqa.core.components.retriever.base_retriever.AllRetriever
  config:
    name: all_retriever
    retrieval_method: "all"
    resource_types: ["schema", "rule", "example"]
    module_names: ["planner", "retrieval_worker"]

# Configure SQL execution
workers:
  retrieval_worker:
    sql_execution_config:
      name: "sql_executor"
      data_files:
        - path: "<CONFIG_DIR>/data/my_data.csv"
          table_name: my_table

# Use case context
use_case_name: "My Use Case"
use_case_description: "An agent that answers questions about my data."

# SQL dialect
dialect:
  value: "sqlite"
```

### Key Configuration Sections

- **`llm_configs`**: Define your Azure OpenAI LLM connections
- **`llm`**: Map LLMs to agent components (planner, retrieval_worker, etc.)
- **`resource_manager_config`**: Points to your `data/` directory with asset files
- **`retriever_config`**: Controls which assets are loaded and for which components
- **`workers`**: Configures SQL execution (data files, table names)
- **`use_case_name/description`**: Context for prompts
- **`dialect`**: SQL dialect (sqlite, snowflake, etc.)

**Note:** `<CONFIG_DIR>` is automatically replaced with the directory containing `agent.yaml`.

---

## Step 2: Create Knowledge Assets

Create the `data/` directory and add your knowledge assets. See [Knowledge Asset Tools](knowledge_asset_tools.md) or [Create Knowledge Assets Manually](creating_knowledge_assets.md) for detailed guidance.

**Minimum required:** `schema.yml`

**Recommended:** `schema.yml` + `rules.yml` + `examples.yml`

---

## Step 3: Prepare Your Data

Place your data files (CSV, Parquet, etc.) in the `data/` directory or configure paths in `agent.yaml`.

For each data file, specify:
- **Path**: Location of the file (can use `<CONFIG_DIR>`)
- **Table name**: Must match a `table_name` in your `schema.yml`

---

## Step 4: Run Your Agent

Create a Python script to run queries:

```python
import asyncio
from dataqa.integrations.local.client import LocalClient
from dataqa.core.client import CoreRequest

async def main():
    client = LocalClient(config_path="agent.yaml")
    
    request = CoreRequest(
        user_query="Your question here",
        conversation_id="demo-1"
    )
    
    response_generator = client.process_query(request)
    final_response = None
    async for response in response_generator:
        final_response = response
    
    print(final_response.text)
    for df in final_response.output_dataframes:
        print(df)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Advanced Configuration

### Using Different LLMs for Different Components

```yaml
llm_configs:
  powerful_llm:
    type: "dataqa.llm.openai.AzureOpenAI"
    config:
      model: "gpt-4o-2024-08-06"
      # ...
  efficient_llm:
    type: "dataqa.llm.openai.AzureOpenAI"
    config:
      model: "gpt-4o-mini"
      # ...

llm:
  default: efficient_llm
  planner: powerful_llm  # Use powerful model for planning
  retrieval_worker: efficient_llm
```

### Multiple Data Files

```yaml
workers:
  retrieval_worker:
    sql_execution_config:
      data_files:
        - path: "<CONFIG_DIR>/data/customers.csv"
          table_name: customers
        - path: "<CONFIG_DIR>/data/orders.csv"
          table_name: orders
```

The agent can perform JOINs between these tables if relationships are defined in `schema.yml`.

### Custom SQL Dialect

```yaml
dialect:
  value: "snowflake"
  functions: |
    - name: DATEADD(unit, value, date)
      example: DATEADD('day', -30, CURRENT_DATE)
```

---

## Troubleshooting

**File not found errors:**
- Check that paths use `<CONFIG_DIR>` correctly
- Verify all files exist in the specified locations

**Table not found errors:**
- Ensure `table_name` in `data_files` matches `table_name` in `schema.yml`

**Import errors:**
- Verify environment variables are set
- Check that the LLM configuration is correct

---

## Next Steps

- **[Knowledge Asset Tools](knowledge_asset_tools.md)**: Generate and enhance assets with DataScanner and Rule Inference
- **[Create Knowledge Assets Manually](creating_knowledge_assets.md)**: Learn to create comprehensive schema, rules, and examples
- **[Evaluate Your Agent](evaluating_your_agent.md)**: Test and evaluate your agent
- **[API Reference](../reference/index.md)**: Detailed configuration reference

