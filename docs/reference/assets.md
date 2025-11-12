# Reference: Asset File Formats

This document provides a reference for the structure and fields of the three core knowledge asset files: `schema.yml`, `rules.yml`, and `examples.yml`.

---

## `schema.yml`

Describes the structure of your database.

**Root Keys:** `tables`

```yaml
tables:
  - table_name: string                  # REQUIRED. The name of the table as used in SQL queries.
    description: string                  # REQUIRED. A detailed, natural language description of what the table contains and what each row represents.
    primary_keys: list[string]          # Optional. A list of column names that form the primary key.
    foreign_keys: list[object]          # Optional. A list of foreign key definitions.
      - column: string                  # The name of the column in this table.
        reference_table: string          # The name of the table this column references.
        reference_column: string         # The name of the column in the reference table.
    columns: list[object]               # REQUIRED. A list of column definitions.
      - name: string                    # REQUIRED. The name of the column.
        type: string                    # REQUIRED. The SQL data type (e.g., VARCHAR, INTEGER, DATE).
        description: string             # REQUIRED. A detailed description of the column's meaning and use.
        example_values: list[string]    # Optional. A few example values to provide context.
        values: list[object]            # Optional. For categorical columns, a list of possible values and their meanings.
          - value: any                  # The actual value stored in the database.
            description: string          # The human-readable meaning of the value.
```

### Example

```yaml
tables:
  - table_name: sales_report
    description: |
      Contains daily sales records. Each row represents a single sale transaction
      on a specific date for a product in a region.
    primary_keys: ["transaction_id"]
    columns:
      - name: transaction_id
        type: integer
        description: "Unique identifier for each transaction."
      - name: product_id
        type: integer
        description: "Unique identifier for the product. Use this to join with product details tables."
      - name: region
        type: varchar
        description: "The sales region where the sale occurred."
        example_values: ["North", "South", "West"]
        values:
          - value: "North"
            description: "Northern sales region"
          - value: "South"
            description: "Southern sales region"
      - name: sales_date
        type: date
        description: "The date the sales transaction was recorded. Format: YYYY-MM-DD."
      - name: revenue
        type: integer
        description: "The total revenue generated from this sale, in USD."
```

---

## `rules.yml`

Defines explicit business logic and instructions for the agent.

**Root Keys:** `rules`

```yaml
rules:
  - module_name: string                 # REQUIRED. The agent component to apply the rule to (e.g., "retrieval_worker", "planner").
    rules: list[object]                 # REQUIRED. A list of rule definitions for this module.
      - rule_name: string               # REQUIRED. A unique, descriptive name for the rule.
        instructions: string            # REQUIRED. The instruction text for the LLM. Can be a multi-line string.
```

### Module Names

- `"planner"`: Rules for the planning component
- `"replanner"`: Rules for the replanning component
- `"retrieval_worker"`: Rules for SQL generation (most common)
- `"analytics_worker"`: Rules for data analysis
- `"plot_worker"`: Rules for visualization generation

### Example

```yaml
rules:
  - module_name: "retrieval_worker"
    rules:
      - rule_name: "active_account_definition"
        instructions: |
          - An active account is defined as: account_status = 'ACTIVE' AND last_transaction_date >= DATE('now', '-90 days')
          - When filtering for active accounts, always use this exact WHERE clause.
          
      - rule_name: "revenue_exclusions"
        instructions: |
          - Always exclude test accounts when calculating revenue: WHERE account_type != 'TEST'
          - Always exclude refunded transactions: AND transaction_status != 'REFUNDED'
          
  - module_name: "planner"
    rules:
      - rule_name: "comparison_strategy"
        instructions: |
          - When a user asks to compare performance, the plan should always include:
            1. A task to retrieve data for the current period
            2. A separate task to retrieve data for the comparison period
            3. A final task to calculate the difference
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
          question: string              # The user's question (typically same as query).
          reasoning: string             # A multi-line string explaining the step-by-step logic to get to the SQL.
          code: string                  # A multi-line string containing the perfect, complete SQL code.
```

### Example

```yaml
examples:
  - module_name: "retrieval_worker"
    examples:
      - query: "Show me total revenue by region for last month"
        example:
          question: "Show me total revenue by region for last month"
          reasoning: |
            1. The user wants revenue aggregated by region.
            2. "Last month" means the previous calendar month from today.
            3. I need to filter sales_date to be within last month's date range.
            4. I'll GROUP BY region and SUM(revenue).
            5. Last month calculation: WHERE sales_date >= DATE('now', 'start of month', '-1 month') AND sales_date < DATE('now', 'start of month')
          code: |
            SELECT
              region,
              SUM(revenue) as total_revenue
            FROM sales_report
            WHERE sales_date >= DATE('now', 'start of month', '-1 month')
              AND sales_date < DATE('now', 'start of month')
            GROUP BY region
            ORDER BY total_revenue DESC;
```

---

## Best Practices

1. **Schema Descriptions:**
   - Be detailed and specific
   - Include synonyms and business jargon
   - Use `values` for categorical columns with codes

2. **Rules:**
   - Be actionable and specific
   - Target the right component (`retrieval_worker` for SQL, `planner` for strategy)
   - Define business KPIs explicitly

3. **Examples:**
   - Focus on common and complex query patterns
   - Make the `reasoning` block clear and step-by-step
   - Ensure the `code` block is syntactically correct

---

## See Also

- [Building Assets](../guide/building_assets.md): A comprehensive guide to creating these files.
- [Quickstart](../quickstart.md): See examples in action.
