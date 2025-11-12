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

**Q: Where do I put my schema, rules, and example files?**
A: In the directory specified by `asset_directory` in your `resource_manager_config` (commonly `data/`). The path can use `<CONFIG_DIR>` as a placeholder.

---

## Agent Behavior

**Q: The agent generated SQL for a column that doesn't exist. Why?**
A: Your `schema.yml` is likely out of date or the column descriptions are misleading. Ensure your schema perfectly reflects your database and that descriptions are unambiguous.

**Q: The agent fails on a query involving a complex calculation. How do I fix it?**
A: This is a perfect use case for `examples.yml`. Create a new example showing a similar question, a detailed `reasoning` block explaining the steps, and the perfect `code` block with the correct SQL.

**Q: Can the agent work with multiple CSV files or database tables?**
A: Yes. List all your CSV files under `data_files` in the `sql_execution_config` section of your `agent.yaml`. For each file, provide a `table_name` that matches an entry in your `schema.yml`. The agent can then perform `JOIN`s between these tables as if they were in a real database.

**Q: How do I make the agent understand business-specific terms?**
A: Add detailed descriptions in your `schema.yml` that include synonyms and business jargon. For categorical values, use the `values` field to map codes to meanings. You can also add rules in `rules.yml` that define how to interpret specific terms.

**Q: The agent keeps asking for clarification instead of answering. What's wrong?**
A: This usually means there's ambiguity in your schema or the query. Review your column descriptions - do multiple columns match the user's question? Add more specific descriptions or create an example in `examples.yml` that shows how to handle that type of query.

---

## Asset Files

**Q: How detailed should my schema descriptions be?**
A: Very detailed! The descriptions are what the LLM uses to map user questions to your data. Include synonyms, business terms, usage examples, and what each row/column represents. See the [Building Assets](building_assets.md) guide for examples.

**Q: Should I put everything in rules or examples?**
A: Use rules for business logic and constraints (e.g., "always exclude test accounts"). Use examples for complex query patterns and to teach the agent how to think through problems. Both are valuable.

**Q: How many examples do I need?**
A: Start with 3-5 high-quality examples covering your most common and complex query patterns. You can add more as you identify gaps in the agent's performance.

---

## Support

**Q: Where can I get more help?**
A:
- Review the [Troubleshooting Guide](troubleshooting.md).
- Re-read the [Building Assets](building_assets.md) guide, as most issues can be solved by improving the asset files.
- Check the [Configuration Reference](../reference/agent_config.md) for detailed field descriptions.
