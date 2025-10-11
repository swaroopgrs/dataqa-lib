# Quickstart

Get started with DataQA! This guide will walk you through installing the library, configuring your first agent, and running your first query.

---

# 1. Install DataQA

Install the latest version from PyPI:

```bash
pip install aicoelin-dataqa
```

Or, if you use Poetry:

```bash
poetry add aicoelin-dataqa
```

---

# 2. Prepare an Example Agent Configuration

Download or copy a sample agent configuration file (YAML):

```yaml
agent.yaml
----
agent:
  llm_configs:
    gpt-4.1:
      type: dataqa.llm.openai.AzureOpenAI
      config:
        model: "gpt-4.1-2025-04-14"
        api_version: "2024-08-01-preview"
        api_type: "azure_ad"
        temperature: 0
  llm:
    default: gpt-4.1
  resource_manager:
    type: dataqa.components.resource_manager.resource_manager.ResourceManager
    asset_directory: "data/"
  retriever:
    type: dataqa.components.retriever.base_retriever.AllRetriever
    config:
      name: all_retriever
      resource_types: [rule, example, schema]
      module_names: [planner, retrieval_worker]
  workers:
    retrieval_workers:
      - sql_execution_config:
          data_files:
            - path: "data/fake_data.csv"
              table_name: MY_TABLE
  use_case_name: Quickstart Demo
  use_case_description: |
    This agent answers questions about MY_TABLE.
```

> **Tip:** You can find more detailed configuration examples in the [User Guide](/guide/building_your_first_agent.md).

---

# 3. Prepare Data Assets

Place your schema, rules, and example YAML files in the `data/` directory.
See [Sample Asset Files](/guide/building_your_first_agent.md#3-prepare-data-assets) for templates.

---

# 4. Setting Up Environment Variables

Before running your agent, set the required environment variables for LLM access.

```bash
export AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
export OPENAI_API_BASE="https://your-azure-openai-endpoint.net/"
```

You can also place these in a `.env` file in your project directory for convenience.

---

# 5. Run Your First Query

```python
from dataqa.integrations.local.client import LocalClient
from dataqa.core.client import CoreRequest

import asyncio

async def main():
    client = LocalClient(config_path="agent.yaml")
    request = CoreRequest(
        user_query="show me all active customers",
        conversation_id="quickstart-demo"
    )
    response = await client.process_query(request)
    print(response.text)

asyncio.run(main())
```
---

## 5. Next Steps

- [User Guide: Core Concepts](guide/introduction.md)
- [Building Your First Agent](guide/building_your_first_agent.md)
- [API Reference](reference/agent.md)
- [Troubleshooting](guide/troubleshooting.md)

---

## Need Help?

- [FAQ](/guide/faq.md)
