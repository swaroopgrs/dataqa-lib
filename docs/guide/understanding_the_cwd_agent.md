# Understanding the CWD Agent

This guide introduces the core concepts behind the DataQA CWD Agent to help you understand how it works and how to configure it effectively.

---

## What is the CWD Agent?

The CWD (Plan, Worker, Dispatcher) Agent is an autonomous system designed for complex, multi-step data analysis. It mimics how a human analyst approaches a problem: **first you make a plan, then you execute the steps.**

### The Plan-Work-Dispatch Loop

The agent follows a continuous loop:

1. **Planner:** Receives your question and, based on its knowledge (`schema.yml`, `rules.yml`, `examples.yml`), creates a step-by-step plan.

2. **Worker:** A specialized tool executes the first task in the plan.
   - **RetrievalWorker**: Fetches data by generating and running SQL queries
   - **AnalyticsWorker**: Performs data analysis using tools like Pandas
   - **PlotWorker**: Generates charts and graphs

3. **Replanner (Dispatcher):** After a task is done, the Replanner evaluates the result:
   - If the question is answered → formulates the final answer
   - If more work is needed → updates the plan and sends the next task to a worker

This loop continues until the question is fully answered or a maximum number of tasks is reached.

---

## An Analogy: The Research Assistant

Imagine you ask a research assistant, "What's the performance difference between our top two products last quarter?"

1. **Plan:** The assistant thinks:
   - *Task 1:* "I need to identify the top two products by sales volume." (Requires SQL)
   - *Task 2:* "Then, I need to get the sales data for just those two products for last quarter." (Requires SQL)
   - *Task 3:* "Finally, I need to calculate the sales difference and summarize it." (Requires Analysis)

2. **Work:** They execute each task in order, running SQL queries and performing calculations.

3. **Replan/Dispatch:** After each step, they check progress. "Okay, I have the top products. Now for the next step." Once all tasks are done, they compile the final answer.

The CWD Agent works in exactly the same way, using your YAML files as its domain knowledge.

---

## The Three Core Assets

Your agent's intelligence comes from three YAML files:

### 1. `schema.yml` - The Map
- Describes your database structure (tables, columns, relationships)
- Tells the agent what data exists and what it means
- **Most important:** Clear, detailed descriptions are critical

### 2. `rules.yml` - The Rulebook
- Defines business logic and special instructions
- Guides SQL generation and planning strategy
- Ensures consistency in calculations and definitions

### 3. `examples.yml` - The Playbook
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

## Why This Matters

Understanding this workflow helps you configure the agent effectively:

- **`schema.yml`** is the map the `RetrievalWorker` uses to find data. If the map is wrong, it gets lost.
- **`rules.yml`** provides the `RetrievalWorker` with special instructions, like "When calculating revenue, always exclude taxes."
- **`examples.yml`** gives the `RetrievalWorker` a playbook of perfect queries to learn from, making it much more accurate.

---

## Next Steps

- **[Try the Examples](trying_the_examples.md)**: Try the included examples
- **[Configure Your Agent](configuring_your_agent.md)**: Learn to configure your agent
- **[Knowledge Asset Tools](knowledge_asset_tools.md)**: Generate and enhance assets with DataScanner and Rule Inference
- **[Create Knowledge Assets Manually](creating_knowledge_assets.md)**: Master creating schema, rules, and examples

