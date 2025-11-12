# Troubleshooting Guide

This guide covers common issues you might encounter while configuring and running your DataQA CWD Agent, and how to resolve them.

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
2.  **Add an `examples.yml`:** For complex or common query patterns, provide a perfect example with clear `reasoning` and `code` blocks. This is the most powerful way to guide the agent.
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

**Problem: The agent returns empty results or wrong data.**
**Solution:**
- Check that your CSV files are loaded correctly and the `table_name` in `data_files` matches the `table_name` in `schema.yml`.
- Verify that your SQL queries are actually returning data by testing them manually.
- Review your `rules.yml` - you might have overly restrictive filters.

---

## 4. Asset File Issues

**Problem: The agent doesn't seem to use my rules or examples.**
**Solution:**
- Check that your `retriever_config` includes the correct `resource_types` (e.g., `["rule", "schema", "example"]`).
- Verify that the `module_names` in `retriever_config` include the components you want to affect (e.g., `["planner", "retrieval_worker"]`).
- Ensure your YAML files are valid and in the correct directory.

**Problem: Column descriptions aren't helping the agent find the right columns.**
**Solution:**
- Make descriptions more detailed. Include synonyms, business jargon, and usage examples.
- Add `example_values` or `values` for categorical columns.
- Consider adding an example in `examples.yml` that shows how to use that specific column.

---

## 5. SQL Execution Issues

**Problem: SQL syntax errors or "function not found" errors.**
**Solution:**
- Check your `dialect` setting in `agent.yaml`. The default executor uses DuckDB, which supports SQLite-compatible syntax.
- If you're using custom functions, list them in the `dialect.functions` field.
- Review the generated SQL in error messages - the agent might be using functions not available in your SQL engine.

**Problem: "Table not found" errors.**
**Solution:**
- Ensure the `table_name` in your `data_files` configuration exactly matches a `table_name` in your `schema.yml`.
- Check that your CSV files are being loaded correctly (verify the file paths).

---

## 6. Still Stuck?

- Check the [FAQ](faq.md) for more common questions.
- Review the [Building Assets](building_assets.md) guide - most issues can be solved by improving your asset files.
- Ensure your `agent.yaml` matches the structure in the [Configuration Reference](../reference/agent_config.md).
