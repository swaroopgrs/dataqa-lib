# Quickstart

Get your first DataQA CWD Agent running in minutes. This guide includes installation, setup, and your first query.

---

## 1. Installation

### Prerequisites

- **Python:** 3.11 or higher
- **Package Manager:** pip or Poetry

### Install DataQA

```bash
# Using pip
pip install aicoelin-dataqa

# Or using Poetry
poetry add aicoelin-dataqa
```

### Set Up Environment Variables

The agent needs LLM credentials (e.g., Azure OpenAI). Set these before running:

**Option 1: Environment Variables**
```bash
export AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
export OPENAI_API_BASE="https://your-azure-openai-endpoint.net/"
```

**Option 2: `.env` File**
Create a `.env` file in your project root:
```
AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
OPENAI_API_BASE="https://your-azure-openai-endpoint.net/"
```

---

## 2. Project Setup

Create a new directory for your project:

```bash
mkdir my-data-agent
cd my-data-agent
```

Your project structure will look like this:
```
my-data-agent/
├── data/
│   ├── sales_data.csv
│   └── schema.yml
├── agent.yaml
└── run_agent.py
```

---

## 3. Create Sample Data

Create `data/sales_data.csv`:
```csv
product_id,region,sales_date,units_sold,revenue
101,North,2024-01-15,50,5000
102,South,2024-01-16,30,4500
101,North,2024-02-10,45,4500
103,West,2024-02-12,70,8400
102,South,2024-03-05,35,5250
```

---

## 4. Create Schema File

Create `data/schema.yml` to describe your data:
```yaml
tables:
  - table_name: sales_report
    description: "Contains daily sales records, including product, region, units sold, and revenue."
    columns:
      - name: product_id
        type: integer
        description: "Unique identifier for the product."
      - name: region
        type: varchar
        description: "The sales region, such as 'North', 'South', or 'West'."
      - name: sales_date
        type: date
        description: "The date the sales were recorded."
      - name: units_sold
        type: integer
        description: "The total number of units sold on that day."
      - name: revenue
        type: integer
        description: "The total revenue generated from the sales, in USD."
```

---

## 5. Configure the Agent

Create `agent.yaml`:
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
    resource_types: ["schema"]
    module_names: ["planner", "retrieval_worker"]

workers:
  retrieval_worker:
    sql_execution_config:
      name: "sql_executor"
      data_files:
        - path: "<CONFIG_DIR>/data/sales_data.csv"
          table_name: sales_report

use_case_name: "Sales Reporting"
use_case_description: "An agent that answers questions about sales performance from the sales_report table."

dialect:
  value: "sqlite"
```

**Note:** `<CONFIG_DIR>` is automatically replaced with the directory containing `agent.yaml`.

---

## 6. Run Your First Query

Create `run_agent.py`:
```python
import asyncio
from dataqa.integrations.local.client import LocalClient
from dataqa.core.client import CoreRequest

async def main():
    client = LocalClient(config_path="agent.yaml")
    
    request = CoreRequest(
        user_query="What was the total revenue for the North region?",
        conversation_id="quickstart-1",
        question_id="q1"
    )
    
    response_generator = client.process_query(request)
    final_response = None
    async for response in response_generator:
        final_response = response
    
    print("--- Final Answer ---")
    print(final_response.text)
    
    print("\n--- Output DataFrames ---")
    for i, df in enumerate(final_response.output_dataframes):
        print(f"DataFrame {i+1}:")
        print(df.to_markdown(index=False))

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python run_agent.py
```

You should see the agent's answer and the resulting DataFrame!

---

## What's Next?

- **[Understanding the CWD Agent](guide/understanding_the_cwd_agent.md)**: Understand how the agent works
- **[Knowledge Asset Tools](guide/knowledge_asset_tools.md)**: Generate and enhance assets with DataScanner and Rule Inference
- **[Create Knowledge Assets Manually](guide/creating_knowledge_assets.md)**: Learn to create comprehensive schema, rules, and examples
- **[Configure Your Agent](guide/configuring_your_agent.md)**: Deep dive into agent configuration

