Of course. This is a great direction. Focusing the documentation squarely on the end-user who wants to configure and use the CWD Agent is the most effective approach.

Here is a complete rewrite of your documentation suite with the new focus: agent-centric, user-oriented, and heavily emphasizing asset creation.

***

================================================
FILE: docs/index.md
================================================
# DataQA: The Conversational Data Agent

A powerful, config-driven framework for building intelligent agents that answer natural language questions about your data.

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

---

## What is the DataQA Agent?

The **DataQA CWD Agent** is an intelligent system designed to understand your data and answer complex questions. It follows a "Plan, Work, Dispatch" loop to break down your query, retrieve data using SQL, perform analysis, and deliver a clear, accurate answer.

Instead of writing complex code, you teach the agent about your data through simple YAML files.

## Who is this for?

- **Data Analysts & Scientists** who want to enable natural language querying for their datasets.
- **Teams** that need a robust, reliable way to build and maintain a conversational data assistant.
- **Developers** creating data-centric applications without wanting to manage complex agentic loops.

---

## Key Features

- **Conversational `CWD` Agent:** A powerful plan-and-execute loop for tackling multi-step questions.
- **Knowledge-Driven:** Define your data's schema, business rules, and query examples in simple YAML files.
- **Config, Not Code:** Most agent behavior is defined in a central `agent.yaml` file.
- **Built-in Tools:** Comes with tools for SQL generation, data analysis (Pandas), and plotting out-of-the-box.
- **Local First:** Designed for easy local setup and testing before deployment.

---

## Get Started

<div class="grid cards" markdown>

-   **Quickstart Guide**

    ---

    Get your first agent running in 5 minutes. This is the best place to start.

    [:octicons-arrow-right-24: Quickstart Guide](quickstart.md)

-   **User Guide**

    ---

    Learn the core concepts and master the art of configuring your agent's knowledge base.

    [:octicons-arrow-right-24: User Guide](guide/introduction.md)

-   **Configuration Reference**

    ---

    Detailed reference for the `agent.yaml` configuration file and data asset formats.

    [:octicons-arrow-right-24: Configuration Reference](reference/agent_config.md)

</div>

---

## Community & Support

- [Troubleshooting Guide](guide/troubleshooting.md)
- [FAQ](guide/faq.md)

================================================
FILE: docs/installation.md
================================================
# Installation

This guide will help you install DataQA and set up your environment to run the CWD Agent.

---

## Prerequisites

- **Python:** 3.11 or higher
- **Package Manager:** pip

---

# 1. Install DataQA

Install the latest version of the library from PyPI using pip. It's recommended to do this in a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install the library
pip install aicoelin-dataqa
```

---

# 2. Set Up Environment Variables

The DataQA Agent needs credentials to access a Large Language Model (LLM), such as Azure OpenAI. You must set these variables before running your agent.

### **Method 1: In your shell (Recommended for testing)**

```bash
export AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
export OPENAI_API_BASE="https://your-azure-openai-endpoint.net/"

# Optional: For certificate-based authentication
export CLIENT_ID="your-azure-client-id"
export TENANT_ID="your-azure-tenant-id"
export CERT_PATH="/path/to/your/certificate.pem"
```

### **Method 2: In a `.env` file (Recommended for projects)**

Create a file named `.env` in your project's root directory:

```
AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
OPENAI_API_BASE="https://your-azure-openai-endpoint.net/"

# Optional cert auth variables
CLIENT_ID="your-azure-client-id"
TENANT_ID="your-azure-tenant-id"
CERT_PATH="/path/to/your/certificate.pem"
```
The library will automatically load these variables if `python-dotenv` is installed.

---

# 3. Verify Your Installation

You can quickly check that the library is installed and accessible:

```bash
python -c "import dataqa; print(f'DataQA version: {dataqa.__version__}')"
```

If this command prints the version number without errors, your installation was successful.

---

# Next Steps

You are now ready to build and run your first agent!

- **[Quickstart Guide](quickstart.md)**: Your next stop to get an agent running immediately.

================================================
FILE: docs/quickstart.md
================================================
# Quickstart

This guide will get you from zero to a fully running DataQA Agent in under 5 minutes. We'll use a sample CSV file and a minimal configuration.

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

## 4. Write the Python Script

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

## 5. Run Your Agent!

Ensure your LLM environment variables are set, then run the script:

```bash
python run_agent.py
```

You should see the agent's final answer and the resulting DataFrame printed to your console!

---

## Next Steps

Congratulations! You've successfully built and run a DataQA agent.

- **[User Guide: Building Your First Agent](guide/building_your_first_agent.md)**: Dive deeper into configuring `rules.yml` and `examples.yml` to make your agent even smarter.

================================================
FILE: docs/guide/introduction.md
================================================
# Introduction to the CWD Agent

This guide introduces the core concepts behind the DataQA CWD Agent to help you understand how it works and how to configure it effectively.

---

## What is the CWD Agent?

The CWD (Plan, Worker, Dispatcher) Agent is an autonomous system designed for complex, multi-step data analysis. It mimics how a human analyst approaches a problem: **first you make a plan, then you execute the steps.**

It follows a **Plan-and-Execute** loop:

1.  **Planner:** Receives your question and, based on its knowledge (`schema.yml`, `rules.yml`, `examples.yml`), creates a step-by-step plan.
2.  **Worker:** A specialized tool is chosen to execute the first task in the plan.
    *   `RetrievalWorker`: Fetches data by generating and running SQL queries.
    *   `AnalyticsWorker`: Performs data analysis using tools like Pandas.
    *   `PlotWorker`: Generates charts and graphs.
3.  **Replanner (Dispatcher):** After a task is done, the Replanner looks at the result. It decides if the original question has been answered.
    *   If yes, it formulates the final answer.
    *   If no, it updates the plan and sends the next task to a worker.

This loop allows the agent to handle ambiguity, recover from errors, and tackle problems that require multiple intermediate calculations.

### An Analogy: The Research Assistant

Imagine you ask a research assistant, "What's the performance difference between our top two products last quarter?"

1.  **Plan:** The assistant doesn't just type randomly. They think:
    *   *Task 1:* "I need to identify the top two products by sales volume." (Requires SQL)
    *   *Task 2:* "Then, I need to get the sales data for just those two products for last quarter." (Requires SQL)
    *   *Task 3:* "Finally, I need to calculate the sales difference and summarize it." (Requires Analysis)

2.  **Work:** They execute each task in order.
    *   *Execute Task 1:* They run a SQL query to find the top products.
    *   *Execute Task 2:* They run another SQL query to filter the data.
    *   *Execute Task 3:* They use a spreadsheet or Python to do the final calculation.

3.  **Replen/Dispatch:** After each step, they check their progress. "Okay, I have the top products. Now for the next step." Once all tasks are done, they compile the final answer for you.

The CWD Agent works in exactly the same way, using your YAML files as its domain knowledge.

---

## Why This Matters for Configuration

Understanding this workflow helps you configure the agent effectively:

-   **`schema.yml`** is the map the `RetrievalWorker` uses to find data. If the map is wrong, it gets lost.
-   **`rules.yml`** provides the `RetrievalWorker` with special instructions, like "When calculating revenue, always exclude taxes." It also guides the `Planner` on high-level strategy.
-   **`examples.yml`** gives the `RetrievalWorker` a playbook of perfect queries to learn from, making it much more accurate.

---

## Next Steps

- **[Building Your First Agent](building_your_first_agent.md)**: A deep dive into creating the `schema`, `rules`, and `examples` files that power the agent's brain.

================================================
FILE: docs/guide/building_your_first_agent.md
================================================
# Guide: Configuring Your Agent's Knowledge

This is the most important guide for making your CWD Agent smart and accurate. Your agent's performance depends directly on the quality of the knowledge you provide in three key files: `schema.yml`, `rules.yml`, and `examples.yml`.

---

## Your Project Structure

A typical agent project has the following structure. All asset files live in a dedicated directory (e.g., `data/`) which you specify in your main `agent.yaml`.

```
my-agent/
├── data/
│   ├── schema.yml      # The MAP of your data
│   ├── rules.yml       # The RULEBOOK for business logic
│   └── examples.yml    # The PLAYBOOK of how to act
└── agent.yaml          # The main agent configuration
```

---

## 1. `schema.yml`: The Map of Your Data

**Purpose:** This file describes your database structure. It's how the agent knows what tables and columns exist and what they mean. Without a clear schema, the agent is blind.

#### **Detailed Guide & Best Practices**

1.  **Descriptions are Everything:** The `description` fields for tables and columns are what the LLM reads to understand your data and map user questions to the correct fields.
    *   **Bad Column Description:** `acct_st` -> `"Account Status"`
    *   **Good Column Description:** `acct_st` -> `"The current status of the account, also referred to as 'Account State'. Use this column to filter for active or inactive accounts."`

2.  **Explain the "What" and the "How":**
    *   For a **table description**, explain what a single row represents (e.g., `"Each row represents a single credit card transaction for a customer."`).
    *   For a **column description**, mention common synonyms or business jargon (e.g., `"The `xref_c1` column is the unique customer identifier, often called 'ECID' in reports."`).

3.  **Use `values` for Categorical Columns:** This is extremely powerful for columns with a fixed set of codes or categories. It prevents the agent from guessing values and helps it map user language (e.g., "open accounts") to the correct code (`'A1'`).

    ```yaml
    - name: account_condition_code
      type: varchar
      description: "Code indicating the account's current condition."
      values:
        - value: "A1"
          description: "Open account / Active"
        - value: "A2"
          description: "Paid account / Zero balance"
        - value: "05"
          description: "Account transferred"
    ```

4.  **Define Relationships with `primary_keys` and `foreign_keys`:** Explicitly defining keys helps the agent construct correct `JOIN` statements between tables.

    ```yaml
    tables:
      - table_name: customers
        primary_keys: ["customer_id"]
        # ... columns ...
      - table_name: orders
        primary_keys: ["order_id"]
        foreign_keys:
          - column: customer_id
            reference_table: customers
            reference_column: customer_id
        # ... columns ...
    ```

---

## 2. `rules.yml`: The Rulebook for Business Logic

**Purpose:** This file injects explicit instructions and business logic that cannot be inferred from the schema alone, like complex calculations or company policies.

#### **Detailed Guide & Best Practices**

1.  **Target the Right Component:** Rules are injected into specific agent components. For users, the most important one is `retrieval_worker`, which influences **SQL generation**.

    ```yaml
    rules:
      - module_name: "retrieval_worker"
        rules:
          - rule_name: "delinquency_definition"
            instructions: |
              - A delinquent account is defined as: ac_st not in ('NA','CURRENT') and bal_final > 0.
              - When asked for delinquency rate, calculate it as the sum of delinquent balances divided by the total outstanding balance.
    ```

2.  **Be Specific and Actionable:** Vague rules are ignored.
    *   **Bad Rule:** `"Handle dates correctly."`
    *   **Good Rule:** `"When a user asks for data 'year to date' or 'YTD', filter the date column from January 1st of the current year up to today's date."`

3.  **Define Business KPIs:** The `rules.yml` file is the perfect place to define how Key Performance Indicators (KPIs) are calculated. This ensures consistency and accuracy.
    *   `"Customer Lifetime Value (CLV) is calculated as (average_order_value * purchase_frequency) / churn_rate."`

---

## 3. `examples.yml`: The Playbook of How to Act

**Purpose:** LLMs excel at pattern matching. This file provides high-quality, "few-shot" examples to show the agent: "When you see a question like *this*, produce an output *exactly like this*."

#### **Detailed Guide & Best Practices**

1.  **Focus on SQL Generation:** Provide examples for the `retrieval_worker` to teach it how to write perfect SQL for your use case.

2.  **The `<reasoning>` Block is Your Teaching Moment:** The text inside `<reasoning>` is crucial. It teaches the agent *how to think*. It should be a clear, step-by-step breakdown of how you get from the user's question to the final SQL query.

3.  **The `<code>` Block Must Be Perfect:** The SQL inside `<code>` must be syntactically correct and produce the right answer. The agent will learn to mimic this structure, style, and logic.

4.  **Cover Common and Complex Cases:**
    *   **Ambiguous Terms:** If "active user" is an ambiguous term, provide an example that shows the canonical definition in the reasoning and the correct `WHERE` clause in the code.
    *   **Difficult Joins:** If there's a tricky but common join path across multiple tables, create an example for it.
    *   **Complex Calculations:** Show examples for common but complex calculations like Year-over-Year growth, moving averages, etc.

**Example Structure:**
```yaml
examples:
  - module_name: "retrieval_worker"
    examples:
      - query: "What was our YoY revenue growth for auto loans?"
        example:
          question: "What was our YoY revenue growth for auto loans?"
          reasoning: |
            1. The user wants Year-over-Year (YoY) revenue growth.
            2. This requires comparing revenue from the current period to the same period in the previous year.
            3. I will use the LAG() window function partitioned by month to get the prior year's revenue.
            4. The product filter should be 'Auto Loan'.
            5. The final formula is (current_revenue - prior_year_revenue) / prior_year_revenue.
          code: |
            WITH MonthlyRevenue AS (
              SELECT
                STRFTIME('%Y-%m', sales_date) as sales_month,
                SUM(revenue) as total_revenue
              FROM sales_report
              WHERE product_type = 'Auto Loan'
              GROUP BY 1
            ),
            YoY AS (
              SELECT
                sales_month,
                total_revenue,
                LAG(total_revenue, 12) OVER (ORDER BY sales_month) as prior_year_revenue
              FROM MonthlyRevenue
            )
            SELECT
              sales_month,
              (total_revenue - prior_year_revenue) * 100.0 / prior_year_revenue as yoy_growth_pct
            FROM YoY
            WHERE prior_year_revenue IS NOT NULL;
```

---

## Next Steps

- **[Configuration Reference](reference/agent_config.md)**: See the detailed guide for the main `agent.yaml` file.
- **[Troubleshooting](troubleshooting.md)**: Tips for when your agent isn't behaving as expected.

================================================
FILE: docs/guide/benchmarking.md
================================================
# Evaluating Your Agent

DataQA includes a benchmarking suite to help you test and evaluate your agent's performance as you refine its knowledge base. This is crucial for tracking improvements and catching regressions.

---

## 1. Why Evaluate?

- **Measure Accuracy:** Objectively determine if your agent is answering questions correctly.
- **Track Progress:** See if changes to your `schema.yml`, `rules.yml`, or `examples.yml` are making the agent smarter.
- **Prevent Regressions:** Ensure that fixing one type of query doesn't break another.

---

## 2. Preparing Test Data

The foundation of any evaluation is a good test set. Create a YAML file containing a list of questions you expect your agent to answer correctly.

**Example: `test_questions.yml`**
```yaml
# A friendly name for your test suite
use_case: "Sales Agent Evaluation"

# A list of test items
data:
  - id: "sales_total_revenue"
    question: "What is the total revenue of all sales?"
    # The expected SQL query the agent should generate
    solution:
      - function_arguments:
          sql: "SELECT SUM(revenue) FROM sales_report;"
    # Optional: A ground truth text answer for LLM-based judging
    ground_truth_output: "The total revenue is $27,650."
    active: true

  - id: "sales_by_region"
    question: "Show me the total revenue per region."
    solution:
      - function_arguments:
          sql: "SELECT region, SUM(revenue) FROM sales_report GROUP BY region;"
    active: true
```
*   `id`: A unique identifier for the test case.
*   `question`: The user query to test.
*   `solution`: The ground truth, typically the exact SQL you expect the agent to generate.
*   `ground_truth_output`: An optional final text answer, used for more advanced LLM-based judging.

---

## 3. Running the Benchmark

The benchmarking suite is run from the command line. You point it to your agent's configuration and your test data file.

```bash
# Set your LLM environment variables first!
export AZURE_OPENAI_API_KEY="..."
export OPENAI_API_BASE="..."

# Run the test
python -m dataqa.core.components.knowledge_extraction.rule_inference_batch_test \
    --config /path/to/your/agent.yaml \
    --test-data /path/to/your/test_questions.yml
```

*(Note: The entry point `rule_inference_batch_test.py` is planned to be renamed to a more general test runner in a future release.)*

---

## 4. Interpreting Results

The benchmark script will output:
- **Execution Logs:** To your console, showing each test question and the agent's generated SQL.
- **Result Files:** An Excel (`.xlsx`) and a Pickle (`.pkl`) file in a `temp/` directory containing detailed results for each test case, including:
    - The original question.
    - The expected SQL.
    - The generated SQL.
    - An `llm_label` (Correct/Wrong) if `ground_truth_output` was provided and evaluated.

By comparing the "expected_sql" and "generated_sql" columns in the output Excel file, you can quickly identify where your agent is succeeding or failing.

---

## 5. Iterative Improvement

The typical workflow is:
1.  Run the benchmark and identify a failing test case.
2.  Analyze *why* it failed. Is a column description unclear? Is a business rule missing? Is there a complex query pattern that needs an example?
3.  Update your `schema.yml`, `rules.yml`, or `examples.yml` to fix the issue.
4.  Run the benchmark again to confirm the fix and ensure no other tests have broken.
5.  Repeat.

================================================
FILE: docs/guide/troubleshooting.md
================================================
# Troubleshooting Guide

This guide covers common issues you might encounter while configuring and running your DataQA Agent, and how to resolve them.

---

## 1. Installation & Setup Issues

**Problem: `ModuleNotFoundError: No module named 'dataqa'`**
**Solution:**
- Make sure you have installed the library: `pip install aicoelin-dataqa`
- If you're using a virtual environment, ensure it is activated.

**Problem: `AuthenticationError`, `401 Unauthorized`, or `KeyError` for API keys**
**Solution:** This is almost always an environment variable issue.
- Double-check that `AZURE_OPENAI_API_KEY` and `OPENAI_API_BASE` are correctly set in your shell or `.env` file.
- Verify that the variables are loaded in your Python script by printing them: `import os; print(os.getenv("AZURE_OPENAI_API_KEY"))`.
- Ensure there are no typos in the variable names.

---

## 2. Configuration & YAML Issues

**Problem: `FileNotFoundError` for `schema.yml` or a data file.**
**Solution:** The agent cannot find your asset files.
- Check the `asset_directory` path in your `resource_manager_config`. The special placeholder `<CONFIG_DIR>` refers to the directory where your `agent.yaml` is located.
- Ensure your file paths under `data_files` in the `sql_execution_config` are correct.
- Verify that the files (`schema.yml`, `rules.yml`, `examples.yml`, data CSVs) exist in the specified locations.

**Problem: `YAMLError` or `yaml.scanner.ScannerError`**
**Solution:** You have a syntax error in one of your YAML files.
- **Indentation is critical in YAML.** Use a text editor that shows spaces to find indentation mistakes.
- Check for missing colons (`:`), incorrect list dashes (`-`), or unclosed quotes.
- Use an online YAML validator to help find the error.

---

## 3. Agent Behavior Issues

**Problem: The agent generates incorrect SQL (e.g., uses the wrong table/column, hallucinates a function).**
**Solution:** This is a knowledge gap. The agent doesn't understand your data well enough.
1.  **Start with `schema.yml`:** Are your table and column descriptions crystal clear? Have you added descriptions for categorical `values`? This is the most common fix.
2.  **Add an `example.yml`:** For complex or common query patterns, provide a perfect example with clear `<reasoning>` and `<code>` blocks. This is the most powerful way to guide the agent.
3.  **Add a `rules.yml`:** If it's a hard business rule (e.g., "always exclude test accounts"), add it as an instruction for the `retrieval_worker`.

**Problem: The agent says it cannot answer the question or asks for clarification.**
**Solution:**
- This can be a good thing! It means the agent identified an ambiguity.
- Review your `schema.yml`. Does the user's query contain a term that could map to multiple columns? Clarify the column descriptions.
- For example, if you have `order_date` and `ship_date`, and the user asks "show orders from last week," the agent might be confused. You can add a rule or clarify in the descriptions which date is the default for such queries.

**Problem: The agent's plan seems illogical or inefficient.**
**Solution:** This is a higher-level strategy problem.
- Add a rule for the `planner` module in `rules.yml` to guide its strategy.
- **Example:** `"When a user asks to compare performance, the plan should always include a task to retrieve data for the current period and a separate task to retrieve data for the comparison period."`

---

## 4. Still Stuck?

- Check the [FAQ](faq.md) for more common questions.

================================================
FILE: docs/guide/faq.md
================================================
# Frequently Asked Questions (FAQ)

Find answers to common questions about using and configuring the DataQA CWD Agent.

---

## General

**Q: What is the CWD Agent?**
A: It's an autonomous agent that follows a Plan-Work-Dispatch loop to answer complex, multi-step questions about your data using natural language.

**Q: What is the primary way to control the agent's behavior?**
A: Through three YAML files: `schema.yml` (to describe your data), `rules.yml` (to define business logic), and `examples.yml` (to show correct query patterns).

---

## Installation & Setup

**Q: What Python version do I need?**
A: Python 3.11 or higher.

**Q: How do I provide my LLM credentials?**
A: Set the environment variables `AZURE_OPENAI_API_KEY` and `OPENAI_API_BASE` in your shell or a `.env` file before running your application. See the [Installation Guide](../installation.md) for details.

---

## Configuration

**Q: What does `<CONFIG_DIR>` mean in the YAML file?**
A: It's a special placeholder that gets replaced with the absolute path to the directory containing your `agent.yaml` file. This makes your configuration portable. For example, `path: "<CONFIG_DIR>/data/my_data.csv"` will always resolve correctly.

**Q: My agent isn't using my business logic. Where do I put rules?**
A: Place rules that affect SQL generation under the `retrieval_worker` `module_name` in your `rules.yml` file. This is the most common use case.

**Q: Can I use a different LLM for planning versus SQL generation?**
A: Yes. In your `agent.yaml`, you can define multiple LLMs in `llm_configs` and assign them to specific components (like `planner` or `retrieval_worker`) in the `llm` section.

---

## Agent Behavior

**Q: The agent generated SQL for a column that doesn't exist. Why?**
A: Your `schema.yml` is likely out of date or the column descriptions are misleading. Ensure your schema perfectly reflects your database and that descriptions are unambiguous.

**Q: The agent fails on a query involving a complex calculation. How do I fix it?**
A: This is a perfect use case for `examples.yml`. Create a new example showing a similar question, a detailed `<reasoning>` block explaining the steps, and the perfect `<code>` block with the correct SQL.

**Q: Can the agent work with multiple CSV files or database tables?**
A: Yes. List all your CSV files under `data_files` in the `sql_execution_config` section of your `agent.yaml`. For each file, provide a `table_name` that matches an entry in your `schema.yml`. The agent can then perform `JOIN`s between these tables as if they were in a real database.

---

## Support

**Q: Where can I get more help?**
A:
- Review the [Troubleshooting Guide](troubleshooting.md).
- Re-read the [Guide to Configuring Your Agent's Knowledge](building_your_first_agent.md), as most issues can be solved by improving the asset files.

================================================
FILE: docs/reference/agent_config.md
================================================
# Reference: `agent.yaml` Configuration

This document provides a detailed reference for all the settings available in the main `agent.yaml` configuration file for the CWD Agent.

---

### Top-Level Fields

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `agent_name` | string | No | An optional name for your agent configuration. |
| `llm_configs` | object | Yes | A dictionary where you define one or more LLM connections. |
| `llm` | object | Yes | Maps the defined LLMs from `llm_configs` to specific agent components. |
| `resource_manager_config` | object | Yes | Configures where the agent finds its knowledge assets (`schema.yml`, etc.). |
| `retriever_config` | object | Yes | Defines how the agent retrieves and uses the knowledge assets for prompts. |
| `workers` | object | Yes | Configures the execution environments for workers, especially the SQL executor. |
| `use_case_name` | string | Yes | A short name for your use case, used in prompts for context. |
| `use_case_description`| string | Yes | A detailed description of what the agent does, also used in prompts. |
| `dialect` | object | Yes | Specifies the SQL dialect and available functions for SQL generation. |
| `max_tasks` | integer | No | The maximum number of tasks an agent can execute for a single query. Defaults to 10. |
| `timeout` | integer | No | Timeout in seconds for a single query execution. Defaults to 300. |

---

### `llm_configs`

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

---

### `llm`

This section maps the LLMs defined in `llm_configs` to the agent's internal components. This allows you to use different models for different purposes (e.g., a powerful model for planning and a cheaper one for summarization).

```yaml
llm:
  # The LLM to use if a component-specific one is not set.
  default: my_default_llm
  # Optional overrides for specific components.
  planner: my_powerful_llm # (if you defined another LLM)
  replanner: my_default_llm
  retrieval_worker: my_default_llm
  analytics_worker: my_default_llm
  plot_worker: my_default_llm
```

---

### `resource_manager_config`

This tells the agent where to load the asset files from.

```yaml
resource_manager_config:
  config:
    # Path to the directory containing schema.yml, rules.yml, etc.
    # <CONFIG_DIR> is a placeholder for the directory of this agent.yaml file.
    asset_directory: "<CONFIG_DIR>/data/"
```

---

### `retriever_config`

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

---

### `workers`

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
```

---

### `dialect`

This helps the SQL generator produce correct syntax for your target database.

```yaml
dialect:
  # E.g., "sqlite", "snowflake", "redshift"
  value: "sqlite"
  # Optional: A multi-line string listing custom functions available.
  functions: |
    - name: STRFTIME(format, timestring)
      example: STRFTIME('%Y', date_column) = '2024'
```

================================================
FILE: docs/reference/asset_files.md
================================================
# Reference: Asset File Formats

This document provides a reference for the structure and fields of the three core knowledge asset files: `schema.yml`, `rules.yml`, and `examples.yml`.

---

## `schema.yml`

Describes the structure of your database.

**Root Keys:** `tables`

```yaml
tables:
  - table_name: string                  # REQUIRED. The name of the table as used in SQL queries.
    description: string               # REQUIRED. A detailed, natural language description of what the table contains and what each row represents.
    primary_keys: list[string]        # Optional. A list of column names that form the primary key.
    foreign_keys: list[object]        # Optional. A list of foreign key definitions.
      - column: string                # The name of the column in this table.
        reference_table: string       # The name of the table this column references.
        reference_column: string      # The name of the column in the reference table.
    columns: list[object]             # REQUIRED. A list of column definitions.
      - name: string                  # REQUIRED. The name of the column.
        type: string                  # REQUIRED. The SQL data type (e.g., VARCHAR, INTEGER, DATE).
        description: string           # REQUIRED. A detailed description of the column's meaning and use.
        example_values: list[string]  # Optional. A few example values to provide context.
        values: list[object]          # Optional. For categorical columns, a list of possible values and their meanings.
          - value: any                # The actual value stored in the database.
            description: string       # The human-readable meaning of the value.
```

---

## `rules.yml`

Defines explicit business logic and instructions for the agent.

**Root Keys:** `rules`

```yaml
rules:
  - module_name: string                 # REQUIRED. The agent component to apply the rule to (e.g., "retrieval_worker", "planner").
    rules: list[object]                 # REQUIRED. A list of rule definitions for this module.
      - rule_name: string                 # REQUIRED. A unique, descriptive name for the rule.
        instructions: string            # REQUIRED. The instruction text for the LLM. Can be a multi-line string.
```

---

## `examples.yml`

Provides high-quality, few-shot examples to guide the agent's SQL generation.

**Root Keys:** `examples`

```yaml
examples:
  - module_name: string                 # REQUIRED. The agent component the example is for (usually "retrieval_worker").
    examples: list[object]              # REQUIRED. A list of example definitions.
      - query: string                   # REQUIRED. The user-like query for this example.
        example: object                 # REQUIRED. The content of the example.
          question: string              # The user's question.
          reasoning: string             # A multi-line string explaining the step-by-step logic to get to the SQL.
          code: string                  # A multi-line string containing the perfect, complete SQL code.
```
