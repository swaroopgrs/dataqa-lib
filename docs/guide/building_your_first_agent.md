# Building Your First Agent

This guide walks you through creating and running your own DataQA agent from scratch.

---

## 1. Introduction

A DataQA agent is defined by a YAML configuration file and a set of data assets (schema, rules, examples).
You'll learn how to create these files and run your agent on your own data.

---

## 2. Prepare Your Agent Configuration

Create a file called `agent.yaml` in your project directory:

```yaml
agent_name: "my_first_agent"
llm_configs:
  gpt-4.1:
    type: "dataqa.llm.openai.AzureOpenAI"
    config:
      model: "gpt-4.1-2025-04-14"
      api_version: "2024-08-01-preview"
      api_type: "azure_ad"
      temperature: 0
llm:
  default: gpt-4.1
resource_manager_config:
  type: dataqa.components.resource_manager.resource_manager.ResourceManager
  config:
    asset_directory: "data/"
retriever_config:
  type: dataqa.components.retriever.base_retriever.AllRetriever
  config:
    name: all_retriever
    retrieval_method: all
    resource_types: [rule, example, schema]
    module_names: [planner, retrieval_worker]
workers:
  retrieval_worker:
    sql_execution_config:
      data_files:
        - path: "data/fake_data.csv"
          table_name: MY_TABLE
use_case_name: Demo Use Case
use_case_description: |
  This agent answers questions about MY_TABLE.
dialect:
  value: XXX SQL
  functions: |
    List of SQL functions available in this dialect.

**What does each section mean?**
- `llm_configs`: Define which LLM(s) to use and how to connect.
- `resource_manager_config`: Where to find your schema, rules, and examples.
- `retriever_config`: How to retrieve relevant knowledge for each module.
- `workers`: How to execute SQL and analytics.
- `use_case_name` / `use_case_description`: Human-readable description for prompts.
```

---

## 3. Prepare Data Assets

Create a `data/` directory and add the following files:

**schema file (`schema.yml`):**
```yaml
metadata:
  data_source: snowflake
  my_database_name
  query_language: SQL
  updated_at: 2025/05/09
  version: v1.01
tables:
  - name: MY_TABLE
    description: Example table for demo
    columns:
      - name: id
        description: Unique identifier
        type: integer
      - name: Some
        description: Some value
        type: float
```

**rules file (`rules.yml`):**
```yaml
metadata:
  version: v1.01
  updated_at: 2025/05/09
rules:
  - module_name: planner
    data:
      - name: general_rule_guidelines
        instructions: |
          - Always select the `id` column in queries.
```

**examples file (`examples.yml`):**
```yaml
metadata:
  version: v1.01
  updated_at: 2025/05/09
examples:
  - module_name: retrieval_worker
    data:
      - query: "Show all records"
        example: |
          Q: Show all records
          A:
          <SQL>
          SELECT * FROM MY_TABLE;
          </SQL>
```

> **Tip:** See [Building Your First Agent: Asset Preparation](#) for more details and templates.

---

## 4. Run Your Agent

```python
from dataqa.integrations.local.client import LocalClient
from dataqa.core.client import CoreRequest
import asyncio

async def main():
    client = LocalClient(config_path="agent.yaml")
    request = CoreRequest(
        user_query="Show all records",
        conversation_id="demo-1",
    )
    response = await client.process_query(request)
    print(response.text)

asyncio.run(main())
```

---

## 5. Customizing Your Agent

- **Add more tables or columns** to your schema file.
- **Add more rules** for business logic or analytics.
- **Add more in-context examples** for better LLM performance.
- **Change the LLM backend** or add more workers in your config.

See [Customizing Agents](customizing_agents.md) for advanced options.

---

## 6. Troubleshooting

- **Authentication errors:**
  Check your environment variables for LLM credentials.
- **File not found:**
  Make sure your `data/` directory and asset files exist.
- **Unexpected results:**
  Add more rules or examples to guide the agent.
