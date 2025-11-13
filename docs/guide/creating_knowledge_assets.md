# Creating Knowledge Assets Manually

This guide teaches you how to manually create the three knowledge asset files that power your CWD Agent: `schema.yml`, `rules.yml`, and `examples.yml`. These files teach your agent about your data, business logic, and query patterns.

**Note:** This guide is standalone and can be used whether you're building agents locally or using LLMSuite Database Connect (DBC) integration. The knowledge asset format is the same.

**Tip:** Before writing from scratch, consider using [Knowledge Asset Tools](knowledge_asset_tools.md) to automatically generate or enhance assets from your data.

---

## Overview

Your agent's intelligence comes from three YAML files:

1. **`schema.yml`** - The Map: Describes your database structure
2. **`rules.yml`** - The Rulebook: Defines business logic and instructions
3. **`examples.yml`** - The Playbook: Provides perfect query examples

---

## 1. `schema.yml`: The Map of Your Data

**Purpose:** This file describes your database structure. It's how the agent knows what tables and columns exist and what they mean.

### Basic Structure

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
```

### Best Practices

#### 1. **Descriptions are Everything**

The `description` fields are what the LLM reads to understand your data and map user questions to the correct fields.

- **Bad:** `acct_st` → `"Account Status"`
- **Good:** `acct_st` → `"The current status of the account, also referred to as 'Account State'. Use this column to filter for active or inactive accounts. Common values: 'ACTIVE', 'INACTIVE', 'CLOSED'."`

#### 2. **Explain the "What" and the "How"**

- **Table description:** Explain what a single row represents
  - Example: `"Each row represents a single credit card transaction for a customer."`
- **Column description:** Mention synonyms, business jargon, and usage
  - Example: `"The xref_c1 column is the unique customer identifier, often called 'ECID' in reports."`

#### 3. **Use `values` for Categorical Columns**

For columns with fixed codes or categories, define the values explicitly:

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

This prevents the agent from guessing values and helps it map user language (e.g., "open accounts") to the correct code.

#### 4. **Define Relationships**

Explicitly define primary and foreign keys to help the agent construct correct JOINs:

```yaml
tables:
  - table_name: customers
    primary_keys: ["customer_id"]
    columns:
      - name: customer_id
        type: integer
        description: "Unique customer identifier"
        
  - table_name: orders
    primary_keys: ["order_id"]
    foreign_keys:
      - column: customer_id
        reference_table: customers
        reference_column: customer_id
    columns:
      - name: order_id
        type: integer
        description: "Unique order identifier"
      - name: customer_id
        type: integer
        description: "References customers.customer_id"
```

### Complete Example

```yaml
tables:
  - table_name: sales_report
    description: |
      Contains daily sales records. Each row represents a single sale transaction
      on a specific date for a product in a region. Use this table to answer
      questions about sales performance, revenue, and product distribution.
    columns:
      - name: product_id
        type: integer
        description: "Unique identifier for the product. Use this to join with product details tables."
      - name: region
        type: varchar
        description: "The sales region where the sale occurred. Common values: 'North', 'South', 'East', 'West'."
        example_values: ["North", "South", "West"]
      - name: sales_date
        type: date
        description: "The date the sales transaction was recorded. Format: YYYY-MM-DD. Use this for date-based filtering and grouping."
      - name: units_sold
        type: integer
        description: "The total number of product units sold in this transaction. Always a positive integer."
      - name: revenue
        type: integer
        description: "The total revenue generated from this sale, in USD. Calculated as units_sold * unit_price."
```

---

## 2. `rules.yml`: The Rulebook for Business Logic

**Purpose:** This file injects explicit instructions and business logic that cannot be inferred from the schema alone.

### Basic Structure

```yaml
rules:
  - module_name: "retrieval_worker"
    rules:
      - rule_name: "delinquency_definition"
        instructions: |
          - A delinquent account is defined as: ac_st not in ('NA','CURRENT') and bal_final > 0.
          - When asked for delinquency rate, calculate it as the sum of delinquent balances divided by the total outstanding balance.
```

### Best Practices

#### 1. **Target the Right Component**

Rules are injected into specific agent components:

- **`retrieval_worker`**: Most common - influences SQL generation
- **`planner`**: Guides high-level planning strategy

```yaml
rules:
  - module_name: "retrieval_worker"
    rules:
      - rule_name: "revenue_calculation"
        instructions: |
          - When calculating revenue, always exclude test accounts (account_type = 'TEST').
          - Revenue should be calculated as SUM(amount) WHERE status = 'COMPLETED'.
          
  - module_name: "planner"
    rules:
      - rule_name: "comparison_strategy"
        instructions: |
          - When a user asks to compare performance, the plan should always include:
            1. A task to retrieve data for the current period
            2. A separate task to retrieve data for the comparison period
            3. A final task to calculate the difference
```

#### 2. **Be Specific and Actionable**

- **Bad:** `"Handle dates correctly."`
- **Good:** `"When a user asks for data 'year to date' or 'YTD', filter the date column from January 1st of the current year up to today's date using: WHERE sales_date >= DATE('now', 'start of year') AND sales_date <= DATE('now')."`

#### 3. **Define Business KPIs**

```yaml
rules:
  - module_name: "retrieval_worker"
    rules:
      - rule_name: "kpi_definitions"
        instructions: |
          - Customer Lifetime Value (CLV) is calculated as: (average_order_value * purchase_frequency) / churn_rate
          - Churn rate is: COUNT(DISTINCT churned_customers) / COUNT(DISTINCT all_customers) WHERE churn_date IS NOT NULL
          - Monthly Recurring Revenue (MRR) is: SUM(monthly_subscription_fee) WHERE subscription_status = 'ACTIVE'
```

### Complete Example

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
          - Revenue calculations should use the 'net_revenue' column, not 'gross_revenue'
          
      - rule_name: "date_handling"
        instructions: |
          - When users ask for "last month", use: WHERE transaction_date >= DATE('now', 'start of month', '-1 month') AND transaction_date < DATE('now', 'start of month')
          - When users ask for "year to date" or "YTD", use: WHERE transaction_date >= DATE('now', 'start of year') AND transaction_date <= DATE('now')
```

---

## 3. `examples.yml`: The Playbook of How to Act

**Purpose:** LLMs excel at pattern matching. This file provides high-quality, "few-shot" examples to show the agent: "When you see a question like *this*, produce an output *exactly like this*."

### Basic Structure

```yaml
examples:
  - module_name: "retrieval_worker"
    examples:
      - query: "What was our YoY revenue growth for auto loans?"
        example:
          question: "What was our YoY revenue growth for auto loans?"
          reasoning: |
            1. The user wants Year-over-Year (YoY) revenue growth.
            2. This requires comparing current period to same period last year.
            3. I will use the LAG() window function partitioned by month.
            4. The product filter should be 'Auto Loan'.
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

### Best Practices

#### 1. **The `reasoning` Block is Your Teaching Moment**

The `reasoning` text teaches the agent *how to think*. It should be a clear, step-by-step breakdown:

```yaml
reasoning: |
  1. The user wants "active customers" - I need to check the schema for how active is defined.
  2. From the schema, I see account_status = 'ACTIVE' means active.
  3. The user also wants "last 30 days" - I need to filter by last_transaction_date.
  4. I'll need to join customers table with transactions table to get the transaction date.
  5. The final query should SELECT customer_id, customer_name FROM customers WHERE account_status = 'ACTIVE' AND EXISTS (SELECT 1 FROM transactions WHERE transactions.customer_id = customers.customer_id AND transaction_date >= DATE('now', '-30 days'))
```

#### 2. **The `code` Block Must Be Perfect**

The SQL in `code` must be syntactically correct and produce the right answer. The agent will learn to mimic this structure, style, and logic.

#### 3. **Cover Common and Complex Cases**

- **Ambiguous Terms:** If "active user" is ambiguous, provide an example showing the canonical definition
- **Difficult Joins:** Create examples for tricky but common join paths
- **Complex Calculations:** Show examples for YoY growth, moving averages, etc.

### Complete Example

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

## Asset File Format Reference

Complete reference for the structure and fields of knowledge asset files.

### `schema.yml` Format

Describes the structure of your database.

**Root Structure:**
```yaml
tables:
  - table_name: string                  # REQUIRED
    description: string                  # REQUIRED
    primary_keys: list[string]          # Optional
    foreign_keys: list[object]          # Optional
    columns: list[object]                # REQUIRED
```

**Table Fields:**
- **`table_name`**: The name of the table as used in SQL queries
- **`description`**: Detailed description of what the table contains
- **`primary_keys`**: List of column names forming the primary key
- **`foreign_keys`**: List of foreign key definitions
- **`columns`**: List of column definitions

**Column Fields:**
- **`name`**: Column name (REQUIRED)
- **`type`**: SQL data type (REQUIRED)
- **`description`**: Detailed description (REQUIRED)
- **`example_values`**: Sample values (Optional)
- **`values`**: For categorical columns, list of possible values (Optional)

**Foreign Key Definition:**
```yaml
foreign_keys:
  - column: string                      # Column in this table
    reference_table: string              # Referenced table
    reference_column: string             # Column in referenced table
```

**Categorical Value Definition:**
```yaml
values:
  - value: any                          # The actual value
    description: string                  # Human-readable meaning
```

---

### `rules.yml` Format

Defines business logic and instructions.

**Root Structure:**
```yaml
rules:
  - module_name: string                 # REQUIRED
    rules: list[object]                 # REQUIRED
```

**Rule Fields:**
- **`module_name`**: Component to apply rule to (e.g., "retrieval_worker", "planner")
- **`rules`**: List of rule definitions

**Rule Definition:**
- **`rule_name`**: Unique name for the rule (REQUIRED)
- **`instructions`**: Instruction text for the LLM (REQUIRED, multi-line string)

**Module Names:**
- `"planner"`: Rules for the planning component
- `"replanner"`: Rules for the replanning component
- `"retrieval_worker"`: Rules for SQL generation (most common)
- `"analytics_worker"`: Rules for data analysis
- `"plot_worker"`: Rules for visualization generation

---

### `examples.yml` Format

Provides few-shot examples for the agent.

**Root Structure:**
```yaml
examples:
  - module_name: string                 # REQUIRED
    examples: list[object]              # REQUIRED
```

**Example Fields:**
- **`query`**: The user-like query (REQUIRED)
- **`example`**: The example content (REQUIRED)
  - **`question`**: The user's question (typically same as query)
  - **`reasoning`**: Step-by-step logic explanation (REQUIRED, multi-line string)
  - **`code`**: The perfect SQL code (REQUIRED, multi-line string)

**Module Names:**
- `"retrieval_worker"`: Examples for SQL generation (most common)
- `"analytics_worker"`: Examples for data analysis
- `"plot_worker"`: Examples for visualization

---

## Next Steps

- **[Knowledge Asset Tools](knowledge_asset_tools.md)**: Generate and enhance assets with DataScanner and Rule Inference
- **[Evaluate Your Agent](evaluating_your_agent.md)**: Test your agent's performance
- **[Configure Your Agent](configuring_your_agent.md)**: Set up your agent configuration

