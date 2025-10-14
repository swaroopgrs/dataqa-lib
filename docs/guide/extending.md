# Extending DataQA

DataQA is designed to be extensible.
You can add your own tools, components, LLM adapters, SQL executors, and more-without modifying the core library.

---

## 1. Adding a Custom Analytics or Plot Tool

You can register new tools for analytics or plotting by adding them to the tool registry.

**Example: Register a custom Pandas tool**
```python
from dataqa.core.tools.analytics.tool_generator import DEFAULT_ANALYTICS_TOOLS

def my_custom_sum_tool(memory):
  # define your tool logic here
  pass

DEFAULT_ANALYTICS_TOOLS["MyCustomSumTool"] = my_custom_sum_tool
```
Now, you can reference `"MyCustomSumTool"` in your agent config or prompt.

---

## 2. Creating a Custom LLM Adapter

To use a new LLM backend (e.g., a different API or a local model), subclass the base LLM class and implement the required methods.

**Example: Custom LLM Adapter Skeleton**
```python
from dataqa.core.llm.base_llm import BaseLLM, LLMConfig

class MyLLMAdapter(BaseLLM):
    config_base_model = LLMConfig

    def _get_model(self, **kwargs):
        # return your LLM client here
        pass

    async def ainvoke(self, messages, **kwargs):
        # Implement async LLM call
        pass
```
Reference your adapter in your YAML config:
```yaml
llm_configs:
  my_llm:
    type: "myproject.llms.MyLLMAdapter"
    config: {...}
```

---

## 3. Implementing a Custom SQL Executor

To support a new database or execution engine, subclass the base SQL executor.

**Example: Custom SQL Executor Skeleton**
```python
from dataqa.core.components.code_executor.base_code_executor import CodeExecutor, CodeExecutorOutput

class MySQLExecutor(CodeExecutor):
    def __init__(self, config):
        super().__init__(config)
        # Set up your DB connection here

    async def run(self, input_data, config=None):
        # Execute SQL and return results as CodeExecutorOutput
        pass
```
Reference it in your agent config:
```yaml
workers:
  retrieval_worker:
    sql_execution_config:
      type: myproject.executors.MySQLExecutor
      config:
        connection_string: "postgresql://user:pass@host/db"
```

---

## 4. Adding a New Component or Worker

You can create new components (e.g., for data enrichment, validation, or custom analytics) by subclassing the appropriate base class.

**Example: Custom Worker Skeleton**
```python
from dataqa.core.components.base_component import Component, ComponentConfig

class MyCustomWorker(Component):
    config_base_model = ComponentConfig
    # Define input_base_model and output_base_model as needed

    async def run(self, input_data, config):
        # Your custom logic here
        pass
```
Register your worker in the agent config under `workers`.

---

## 5. Plugin System

For large projects, consider using Python entry points or a plugin registry to auto-discover and load custom tools/components.

---

## 6. Best Practices

- **Keep custom code in a separate package or directory.**
- **Document your custom tools and components.**
- **Write tests for your extensions.**
- **Contribute back!** If your extension is broadly useful, consider submitting a PR.

---

## Next Steps

- [Benchmarking & Evaluation](benchmarking.md)
- [API Reference](../reference/agent.md)
- [FAQ](faq.md)
