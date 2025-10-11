# Customizing Agents

Once you've built your first agent, you can customize and extend it to fit your data, workflows, and use cases.
This guide covers the most common ways to tailor your DataQA agent.

---

## 1. Changing the LLM Backend

You can use different LLMs for different agent components (planner, replanner, workers, etc.) by updating the `llm_configs` and `llm` sections in your YAML config.

**Example: Use GPT-4o for planning, GPT-4.1 for everything else**
```yaml
llm_configs:
  gpt-4.1:
    type: "dataqa.llm.openai.AzureOpenAI"
    config:
      model: "gpt-4.1-2025-04-14"
      api_version: "2024-08-01-preview"
      api_type: "azure_ad"
      temperature: 0
  gpt-4o:
    type: "dataqa.llm.openai.AzureOpenAI"
    config:
      model: "gpt-4o-2024-08-06"
      api_version: "2024-08-01-preview"
      api_type: "azure_ad"
      temperature: 0
llm:
  default: gpt-4.1
  planner: gpt-4o
  replanner: gpt-4.1
  retrieval_worker: gpt-4.1
  analytics_worker: gpt-4.1
  plot_worker: gpt-4.1
```

---

## 2. Adding or Modifying Tools

You can add custom analytics or plotting tools by extending the built-in tool registry.

**Example: Register a custom Pandas tool**
```python
from dataqa.core.tools.analytics.tool_generator import DEFAULT_ANALYTICS_TOOLS

def my_custom_tool(memory):
    # ... define your tool ...
    pass

DEFAULT_ANALYTICS_TOOLS["MyCustomTool"] = my_custom_tool
```
Then, reference your tool in the agent config or prompt.

---

## 3. Customizing Prompts

You can override the default prompts for planning, SQL generation, analytics, or plotting by editing the prompt templates in your config or by providing your own Jinja2 templates.

**Example: Custom planner prompt in YAML**
```yaml
prompts:
  planner_prompt: |
    You are a super planner. Always break down the user query into the smallest possible steps.
    {{use_case_description}}
    {{use_case_schema}}
    {{query}}
```

---

## 4. Adding More Data Assets

- **Schema:** Add more tables, columns, or metadata to your schema YAML.
- **Rules:** Write new rules for edge cases, business logic, or compliance.
- **Examples:** Add more in-context learning examples to improve LLM accuracy.

---

## 5. Using Custom SQL Executors or Data Sources

You can implement your own SQL executor (e.g., for a different database) and reference it in your config.

**Example: Use a custom SQL executor**
```yaml
workers:
  retrieval_worker:
    sql_execution_config:
      type: mypackage.executors.MySQLExecutor
      config:
        connection_string: "postgresql://user:pass@host/db"
```

---

## 6. Advanced: Adding a New Worker

To add a new type of worker (e.g., for a new analytics task):
1. Implement a new worker class (subclassing the appropriate base).
2. Register it in your agent config under `workers`.
3. Add any required tools or prompts.

---

## 7. Tips for Effective Customization

- **Iterate:** Start simple, then add rules/examples as you see gaps.
- **Test:** Use the benchmarking suite to evaluate changes.
- **Document:** Keep your YAML and asset files well-commented for future users.

---

## Next Steps

- [Configuration Deep Dive](configuration.md)
- [Extending DataQA (Plugins, Adapters)](extending.md)
- [API Reference](../reference/agent.md)
- [Need Help?](../faq.md)
- [FAQ](faq.md)