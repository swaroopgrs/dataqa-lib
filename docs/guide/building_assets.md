# Building Your Agent's Knowledge Base

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

### Detailed Guide & Best Practices

#### 1. **Descriptions are Everything**

The `description` fields for tables and columns are what the LLM reads to understand your data and map user questions to the correct fields.

-   **Bad Column Description:** `acct_st` → `"Account Status"`
-   **Good Column Description:** `acct_st` → `"The current status of the account, also referred to as 'Account State'. Use this column to filter for active or inactive accounts."`

**Why it matters:** When a user asks "show me active accounts," the agent needs to know that `acct_st` contains this information and what values represent "active."

#### 2. **Explain the "What" and the "How"**

-   For a **table description**, explain what a single row represents (e.g., `"Each row represents a single credit card transaction for a customer."`).
-   For a **column description**, mention common synonyms or business jargon (e.g., `"The `xref_c1` column is the unique customer identifier, often called 'ECID' in reports."`).

#### 3. **Use `values` for Categorical Columns**

This is extremely powerful for columns with a fixed set of codes or categories. It prevents the agent from guessing values and helps it map user language (e.g., "open accounts") to the correct code (`'A1'`).

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

#### 4. **Define Relationships with `primary_keys` and `foreign_keys`**

Explicitly defining keys helps the agent construct correct `JOIN` statements between tables.

```yaml
tables:
  - table_name: customers
    primary_keys: ["customer_id"]
    columns:
      - name: customer_id
        type: integer
        description: "Unique customer identifier"
      # ... more columns ...
      
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
      # ... more columns ...
```

#### 5. **Complete Example**

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

**Purpose:** This file injects explicit instructions and business logic that cannot be inferred from the schema alone, like complex calculations or company policies.

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

### Detailed Guide & Best Practices

#### 1. **Target the Right Component**

Rules are injected into specific agent components. For users, the most important one is `retrieval_worker`, which influences **SQL generation**.

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

Vague rules are ignored.

-   **Bad Rule:** `"Handle dates correctly."`
-   **Good Rule:** `"When a user asks for data 'year to date' or 'YTD', filter the date column from January 1st of the current year up to today's date using: WHERE sales_date >= DATE('now', 'start of year') AND sales_date <= DATE('now')."`

#### 3. **Define Business KPIs**

The `rules.yml` file is the perfect place to define how Key Performance Indicators (KPIs) are calculated. This ensures consistency and accuracy.

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

#### 4. **Complete Example**

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
          
  - module_name: "planner"
    rules:
      - rule_name: "multi_step_queries"
        instructions: |
          - For comparison queries (e.g., "compare this month to last month"), always create separate tasks:
            1. Retrieve data for period A
            2. Retrieve data for period B
            3. Calculate the difference or ratio
          - Never try to do comparisons in a single SQL query.
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

### Detailed Guide & Best Practices

#### 1. **Focus on SQL Generation**

Provide examples for the `retrieval_worker` to teach it how to write perfect SQL for your use case.

#### 2. **The `<reasoning>` Block is Your Teaching Moment**

The text inside `reasoning` is crucial. It teaches the agent *how to think*. It should be a clear, step-by-step breakdown of how you get from the user's question to the final SQL query.

**Good reasoning:**
```yaml
reasoning: |
  1. The user wants "active customers" - I need to check the schema for how active is defined.
  2. From the schema, I see account_status = 'ACTIVE' means active.
  3. The user also wants "last 30 days" - I need to filter by last_transaction_date.
  4. I'll need to join customers table with transactions table to get the transaction date.
  5. The final query should SELECT customer_id, customer_name FROM customers WHERE account_status = 'ACTIVE' AND EXISTS (SELECT 1 FROM transactions WHERE transactions.customer_id = customers.customer_id AND transaction_date >= DATE('now', '-30 days'))
```

#### 3. **The `code` Block Must Be Perfect**

The SQL inside `code` must be syntactically correct and produce the right answer. The agent will learn to mimic this structure, style, and logic.

#### 4. **Cover Common and Complex Cases**

-   **Ambiguous Terms:** If "active user" is an ambiguous term, provide an example that shows the canonical definition in the reasoning and the correct `WHERE` clause in the code.
-   **Difficult Joins:** If there's a tricky but common join path across multiple tables, create an example for it.
-   **Complex Calculations:** Show examples for common but complex calculations like Year-over-Year growth, moving averages, etc.

#### 5. **Complete Example**

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
            
      - query: "What are the top 3 products by sales volume?"
        example:
          question: "What are the top 3 products by sales volume?"
          reasoning: |
            1. "Sales volume" means units_sold, not revenue.
            2. I need to aggregate by product_id and sum units_sold.
            3. Then order by total units descending and limit to 3.
          code: |
            SELECT
              product_id,
              SUM(units_sold) as total_units
            FROM sales_report
            GROUP BY product_id
            ORDER BY total_units DESC
            LIMIT 3;
            
      - query: "Compare revenue this quarter to last quarter"
        example:
          question: "Compare revenue this quarter to last quarter"
          reasoning: |
            1. This is a comparison query that requires two separate calculations.
            2. I'll create a query that calculates revenue for both periods in one result.
            3. I'll use CASE statements to separate current quarter and last quarter.
            4. Current quarter: Q1 = Jan-Mar, Q2 = Apr-Jun, Q3 = Jul-Sep, Q4 = Oct-Dec
            5. I'll need to determine which quarter we're in and calculate accordingly.
          code: |
            SELECT
              CASE 
                WHEN CAST(STRFTIME('%m', sales_date) AS INTEGER) BETWEEN 1 AND 3 THEN 'Q1'
                WHEN CAST(STRFTIME('%m', sales_date) AS INTEGER) BETWEEN 4 AND 6 THEN 'Q2'
                WHEN CAST(STRFTIME('%m', sales_date) AS INTEGER) BETWEEN 7 AND 9 THEN 'Q3'
                ELSE 'Q4'
              END as quarter,
              SUM(revenue) as total_revenue
            FROM sales_report
            WHERE sales_date >= DATE('now', 'start of year', '-3 months')
              AND sales_date < DATE('now', 'start of month')
            GROUP BY quarter
            ORDER BY quarter;
```

---

## Asset File Format Reference

For complete field-level reference, see [Asset File Formats](../reference/assets.md).

---

## Next Steps

- **[Configuration Reference](../reference/agent_config.md)**: See the detailed guide for the main `agent.yaml` file.
- **[Troubleshooting](troubleshooting.md)**: Tips for when your agent isn't behaving as expected.
- **[Benchmarking](benchmarking.md)**: Learn how to test and evaluate your agent's performance.


