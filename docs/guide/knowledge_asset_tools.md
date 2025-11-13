# Knowledge Asset Tools: Generate & Enhance Assets

DataQA provides powerful tools to automatically generate and enhance knowledge assets from your data and test cases. Use these tools to create new assets or enrich existing ones, saving hours of manual work.

---

## Overview

Two powerful tools help you create and enhance knowledge assets:

1. **DataScanner** - Extracts schema from data files and enriches existing `schema.yml` with inferred descriptions
2. **Rule Inference** - Generates business rules from test cases and enhances existing `rules.yml` with new insights

These tools work in two modes:
- **Create Mode**: Generate new assets from scratch
- **Enhance Mode**: Enrich and improve existing assets

---

## DataScanner: Schema Extraction & Enhancement

DataScanner analyzes your data files to extract schema structure and can enrich existing `schema.yml` files with inferred descriptions and metadata.

### What DataScanner Does

**Create Mode** - Generate new schema:
1. **Extract Schema**: Analyzes your data files (CSV, Parquet, etc.) to extract:
   - Table names and structures
   - Column names and data types
   - Sample values and statistics

2. **Infer Metadata** (Optional): Uses an LLM to generate:
   - Descriptive table descriptions
   - Detailed column descriptions
   - Suggested categorical values

**Enhance Mode** - Enrich existing schema:
- Adds missing column descriptions to existing `schema.yml`
- Infers descriptions for columns that lack them
- Suggests categorical values based on data analysis
- Preserves your existing manual descriptions

### Using DataScanner

#### Method 1: Command Line

Create a configuration file `scanner_config.yml`:

```yaml
data_files:
  - path: "data/sales_data.csv"
    table_name: sales_report
  - path: "data/customers.csv"
    table_name: customers
output_path: "data/"
```

Run DataScanner:

```bash
# Set your LLM environment variables
export AZURE_OPENAI_API_KEY="..."
export OPENAI_API_BASE="..."

# Run schema extraction only (no LLM needed)
python -m dataqa.core.components.knowledge_extraction.data_scanner \
    --config scanner_config.yml

# Or use the full pipeline (extract + infer with LLM)
# This will generate schema_inferred.yml with LLM-generated descriptions
```

#### Method 2: Python API

```python
import asyncio
from dataqa.core.components.knowledge_extraction.data_scanner import DataScanner

config = {
    "data_files": [
        {"path": "data/sales_data.csv", "table_name": "sales_report"}
    ],
    "output_path": "data/"
}

scanner = DataScanner(config)

# Step 1: Extract schema structure (no LLM needed)
await scanner.extract_schema()
# Output: data/schema_extracted.yml

# Step 2: Infer metadata using LLM (requires LLM credentials)
await scanner.infer_metadata()
# Output: data/schema_inferred.yml
```

#### Method 3: Streamlit UI

DataQA includes a Streamlit-based UI for DataScanner:

```bash
streamlit run dataqa/ui/app.py
```

Navigate to the "Data Scanner" page in the UI to:
- Upload data files
- Configure extraction settings
- Run extraction and inference
- Preview and download the generated schema

### DataScanner Output

DataScanner generates two files:

1. **`schema_extracted.yml`**: Raw schema with structure only (no descriptions)
2. **`schema_inferred.yml`**: Complete schema with LLM-inferred descriptions

You can use `schema_inferred.yml` as-is or refine it manually based on your domain knowledge.

### Using DataScanner to Enhance Existing Schema

You can also use DataScanner to enrich an existing `schema.yml`:

```python
# Load existing schema
scanner = DataScanner(config)
scanner.database_schema = load_existing_schema("data/schema.yml")

# Infer metadata for columns that lack descriptions
await scanner.infer_metadata()

# Output: Enhanced schema with new descriptions
```

This is particularly useful when:
- You have a basic schema but want to add descriptions
- New columns were added to your data
- You want to improve existing descriptions with LLM inference

### Refining Generated or Enhanced Schema

After DataScanner generates or enhances your schema:

1. **Review descriptions**: LLM-generated descriptions are good starting points but may need domain-specific refinement
2. **Add categorical values**: For columns with fixed codes, add explicit `values` definitions
3. **Define relationships**: Add `primary_keys` and `foreign_keys` for better JOIN support
4. **Enhance with examples**: Add `example_values` for better context

---

## Rule Inference: Rule Generation & Enhancement

Rule Inference helps you discover missing business rules and enhance existing `rules.yml` by comparing what your agent generates versus what's expected.

### What Rule Inference Does

**Create Mode** - Generate new rules:
1. **Compare SQL**: Takes a test case with:
   - User question
   - Generated SQL (what the agent produced)
   - Expected SQL (what should have been produced)

2. **Identify Gaps**: Analyzes the differences to identify:
   - Missing filters (e.g., "exclude test accounts")
   - Incorrect calculations (e.g., "use net_revenue, not gross_revenue")
   - Business logic gaps (e.g., "active accounts must have transactions in last 90 days")

3. **Generate Rules**: Automatically creates rule suggestions for `rules.yml`

**Enhance Mode** - Improve existing rules:
- Analyzes existing `rules.yml` against test failures
- Suggests refinements to existing rules
- Identifies conflicting or redundant rules
- Consolidates similar rules into more comprehensive ones

### Using Rule Inference

#### Method 1: Streamlit UI (Recommended)

The easiest way to use Rule Inference is through the Streamlit UI:

```bash
streamlit run dataqa/ui/app.py
```

Navigate to the "Rule Inference" page:

1. **Load Test Data**: Upload your test questions YAML file
2. **Run Agent**: Generate SQL for each test case
3. **Compare**: Rule Inference compares generated vs expected SQL
4. **Review Suggestions**: Review suggested rules
5. **Consolidate**: Merge similar rules into your `rules.yml`

#### Method 2: Python API

```python
import asyncio
from dataqa.core.components.knowledge_extraction.rule_inference import RuleInference
from dataqa.core.llm.openai import AzureOpenAI

# Initialize LLM
llm = AzureOpenAI(
    model="gpt-4o-2024-08-06",
    api_version="2024-08-01-preview",
    api_type="azure_ad"
)

# Initialize Rule Inference
rule_inference = RuleInference(llm=llm, prompt=prompt)

# Compare SQL and infer rules
result = await rule_inference(
    query="What is total revenue?",
    generated_sql="SELECT SUM(revenue) FROM sales;",
    expected_sql="SELECT SUM(revenue) FROM sales WHERE status = 'COMPLETED';",
    config=config
)

# result contains suggested rules
print(result["rules"])
```

#### Method 3: Batch Processing

For processing multiple test cases at once:

```python
from dataqa.core.components.knowledge_extraction.rule_inference_batch_test import (
    RuleInferenceExperiment
)

experiment = RuleInferenceExperiment(
    config_path="agent.yaml",
    test_data_file="test_questions.yml",
    output_file_path="output/",
    max_iteration=5
)

# Load test cases
experiment.load_test_data()

# Run batch inference
await experiment.update_rules_from_question_batch()

# Consolidate rules
consolidated_rules = await experiment.consolidate_rules()
```

### Rule Inference Workflow

1. **Prepare Test Data**: Create a test questions file with expected SQL
2. **Run Agent**: Generate SQL for each test case
3. **Identify Failures**: Find cases where generated SQL ≠ expected SQL
4. **Run Rule Inference**: Analyze differences and generate rule suggestions
5. **Review & Refine**: Review suggested rules, refine as needed
6. **Add to rules.yml**: Integrate validated rules into your `rules.yml`
7. **Re-test**: Verify that rules fix the issues

### Example: Discovering a Missing Rule

**Test Case:**
- Question: "What is total revenue for active accounts?"
- Generated SQL: `SELECT SUM(revenue) FROM accounts WHERE status = 'ACTIVE';`
- Expected SQL: `SELECT SUM(revenue) FROM accounts WHERE status = 'ACTIVE' AND last_transaction_date >= DATE('now', '-90 days');`

**Rule Inference Suggests:**
```yaml
rules:
  - module_name: "retrieval_worker"
    rules:
      - rule_name: "active_account_definition"
        instructions: |
          - An active account requires both: status = 'ACTIVE' AND last_transaction_date >= DATE('now', '-90 days')
          - When filtering for active accounts, always include both conditions.
```

---

## Best Practices

### Using Knowledge Asset Tools Effectively

1. **Start with DataScanner**: Use it to generate your initial `schema.yml`
2. **Refine Manually**: Review and enhance the generated schema with domain knowledge
3. **Create Test Cases**: Write test questions with expected SQL
4. **Use Rule Inference**: Run Rule Inference on failing test cases
5. **Iterate**: Continuously refine based on evaluation results

### When to Use These Tools

**Use DataScanner when:**
- You have data files but no schema (create mode)
- You have a basic schema but want to add/enhance descriptions (enhance mode)
- You have many tables/columns to document
- New columns were added to your data

**Use Rule Inference when:**
- You have test cases with expected SQL (create mode)
- Your agent generates incorrect SQL
- You want to discover missing business logic
- You want to refine existing rules based on test results (enhance mode)
- You need to consolidate or deduplicate rules

### When to Write Manually

**Write schema manually when:**
- You need precise control over descriptions
- You have complex relationships to document
- You want to include domain-specific terminology from the start
- You prefer full control over the asset structure

**Write rules manually when:**
- Rules are well-documented business requirements
- You need to define KPIs or complex calculations
- Rules are simple and straightforward
- You want to maintain complete control over rule definitions

---

## Integration with Your Workflow

### Typical Workflow

1. **Initial Creation Phase**:
   - Use DataScanner to generate initial `schema.yml` from data files
   - Review and refine the generated schema

2. **Testing Phase**:
   - Create test cases with expected SQL
   - Run your agent and identify failures

3. **Rule Discovery Phase**:
   - Use Rule Inference on failing test cases to generate new rules
   - Review and integrate suggested rules

4. **Enhancement Phase** (Ongoing):
   - Use DataScanner to enrich schema when new columns are added
   - Use Rule Inference to refine existing rules based on new test results
   - Manually add examples for complex patterns
   - Continuously improve based on evaluation

---

## Next Steps

- **[Create Knowledge Assets Manually](creating_knowledge_assets.md)**: Learn to write schema, rules, and examples from scratch or refine tool-generated assets
- **[Evaluate Your Agent](evaluating_your_agent.md)**: Test your agent's performance to identify areas for enhancement
- **[Configure Your Agent](configuring_your_agent.md)**: Set up your agent configuration

