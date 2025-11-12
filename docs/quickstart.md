# Quickstart

This guide will get you from zero to a fully running DataQA CWD Agent in under 5 minutes. We'll use a sample CSV file and a minimal configuration.

---

## 1. Project Setup

First, create a new directory for your project and navigate into it.

```bash
mkdir my-data-agent
cd my-data-agent
```

Your final project structure will look like this:
```
my-data-agent/
├── data/
│   ├── sales_data.csv
│   └── schema.yml
├── agent.yaml
└── run_agent.py
```

---

## 2. Create the Data and Schema

**A. Create your data file:** `data/sales_data.csv`
```csv
product_id,region,sales_date,units_sold,revenue
101,North,2024-01-15,50,5000
102,South,2024-01-16,30,4500
101,North,2024-02-10,45,4500
103,West,2024-02-12,70,8400
102,South,2024-03-05,35,5250
```

**B. Describe your data in `data/schema.yml`:** This file tells the agent what your data means.
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

## 3. Configure the Agent

Create the main agent configuration file: `agent.yaml`
```yaml
# 1. A name for your agent instance
agent_name: "SalesAgent"

# 2. Define the LLM to use
llm_configs:
  default_llm:
    type: "dataqa.llm.openai.AzureOpenAI"
    config:
      model: "gpt-4o-2024-08-06" # Your Azure deployment name
      api_version: "2024-08-01-preview"
      api_type: "azure_ad"
      temperature: 0

# 3. Assign the LLM to agent components
llm:
  default: default_llm

# 4. Point to your data assets directory
resource_manager_config:
  config:
    asset_directory: "<CONFIG_DIR>/data/" # <CONFIG_DIR> is a placeholder for this file's directory

# 5. Tell the agent how to find and use your assets
retriever_config:
  type: dataqa.core.components.retriever.base_retriever.AllRetriever
  config:
    name: all_retriever
    retrieval_method: "all"
    resource_types: ["schema"]
    module_names: ["planner", "retrieval_worker"]

# 6. Configure the SQL execution environment
workers:
  retrieval_worker:
    sql_execution_config:
      name: "sql_executor"
      data_files:
        - path: "<CONFIG_DIR>/data/sales_data.csv"
          table_name: sales_report # The table name you used in schema.yml

# 7. Provide context for prompts
use_case_name: "Sales Reporting"
use_case_description: "An agent that answers questions about sales performance from the sales_report table."
dialect:
  value: "sqlite" # DuckDB (default) uses sqlite syntax for many functions
```

---

## 4. Set Up Environment Variables

Before running, you need to set your LLM credentials. Create a `.env` file or export them in your shell:

```bash
export AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
export OPENAI_API_BASE="https://your-azure-openai-endpoint.net/"
```

Or create a `.env` file:
```
AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
OPENAI_API_BASE="https://your-azure-openai-endpoint.net/"
```

---

## 5. Write the Python Script

Create your Python script to run the agent: `run_agent.py`
```python
import asyncio
from dataqa.integrations.local.client import LocalClient
from dataqa.core.client import CoreRequest

async def main():
    # Make sure your environment variables are set!
    # export AZURE_OPENAI_API_KEY="..."
    # export OPENAI_API_BASE="..."

    client = LocalClient(config_path="agent.yaml")

    request = CoreRequest(
        user_query="What was the total revenue for the North region?",
        conversation_id="quickstart-1",
        question_id="q1"
    )

    # The client returns a generator. The final item is the CoreResponse.
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

---

## 6. Run Your Agent!

Ensure your LLM environment variables are set, then run the script:

```bash
python run_agent.py
```

You should see the agent's final answer and the resulting DataFrame printed to your console!

---

## Next Steps

Congratulations! You've successfully built and run a DataQA CWD Agent.

- **[User Guide: Introduction](guide/introduction.md)**: Understand how the CWD Agent works.
- **[Building Assets](guide/building_assets.md)**: Learn how to create comprehensive `schema.yml`, `rules.yml`, and `examples.yml` files to make your agent smarter.
- **[Configuration Reference](reference/agent_config.md)**: Explore all configuration options.
