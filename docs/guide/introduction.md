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

3.  **Replan/Dispatch:** After each step, they check their progress. "Okay, I have the top products. Now for the next step." Once all tasks are done, they compile the final answer for you.

The CWD Agent works in exactly the same way, using your YAML files as its domain knowledge.

---

## Why This Matters for Configuration

Understanding this workflow helps you configure the agent effectively:

-   **`schema.yml`** is the map the `RetrievalWorker` uses to find data. If the map is wrong, it gets lost.
-   **`rules.yml`** provides the `RetrievalWorker` with special instructions, like "When calculating revenue, always exclude taxes." It also guides the `Planner` on high-level strategy.
-   **`examples.yml`** gives the `RetrievalWorker` a playbook of perfect queries to learn from, making it much more accurate.

---

## The Three Core Assets

Your agent's intelligence comes from three YAML files:

1. **`schema.yml`** - The Map
   - Describes your database structure (tables, columns, relationships)
   - Tells the agent what data exists and what it means
   - **Most important:** Clear, detailed descriptions are critical

2. **`rules.yml`** - The Rulebook
   - Defines business logic and special instructions
   - Guides SQL generation and planning strategy
   - Ensures consistency in calculations and definitions

3. **`examples.yml`** - The Playbook
   - Provides perfect examples of how to handle specific query patterns
   - Teaches the agent through few-shot learning
   - Most powerful for complex or ambiguous queries

---

## How Assets Flow Through the Agent

```
User Query
    ↓
Retriever → Loads schema.yml, rules.yml, examples.yml
    ↓
Planner → Uses all assets to create a plan
    ↓
RetrievalWorker → Uses schema.yml + rules.yml + examples.yml to generate SQL
    ↓
Replanner → Checks if goal is met, updates plan if needed
    ↓
Final Answer
```

Each component receives the relevant assets it needs to make intelligent decisions.

---

## Next Steps

- **[Building Assets](building_assets.md)**: A deep dive into creating the `schema`, `rules`, and `examples` files that power the agent's brain.
- **[Configuration Reference](../reference/agent_config.md)**: Learn how to configure the `agent.yaml` file.
