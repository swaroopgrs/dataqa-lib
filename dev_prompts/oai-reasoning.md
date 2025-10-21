Below is a **detailed, step-by-step plan** you can feed into an LLM (or break into smaller prompts) to generate the **config-driven** `dataqa` library. It’s organized by feature area, with every module, config schema, component class, and workflow/agent hook you’ll need.

---

## 1. Project & Packaging Setup

1. **Initialize with Poetry**
   - `poetry init` → define name `dataqa`, version, dependencies placeholder.
   - Add dev-dependencies: `pytest`, `ruff`, `sphinx`, `mypy`.

2. **Repo Layout**
   ```
   dataqa/
     ├── dataqa/
     │    ├── __init__.py
     │    ├── config/        # config schema & loaders
     │    ├── components/    # base + concrete components
     │    ├── agents/        # agent orchestrators
     │    ├── workflows/     # predefined workflow graphs
     │    ├── utils/         # logging, errors, retries
     │    └── entrypoint.py  # CLI or Python API
     ├── tests/
     ├── docs/               # Sphinx docs
     ├── pyproject.toml
     └── README.md
   ```

3. **Documentation & CI**
   - Sphinx: `docs/conf.py`, autodoc.
   - GitHub Actions: lint, test, build docs.

---

## 2. Configuration Schema (YAML/JSON)

Define a **single source of truth** for everything:

```yaml
# dataqa/config/schema.yaml

llm:
  provider: "openai"              # e.g. openai, cohere, etc.
  model: "gpt-4"
  max_tokens: 2048

code_executor:
  type: "api"                     # "api" or "in_memory"
  api_endpoint: "https://..."     # if type == api
  timeout: 30

retrieval:
  index_type: "in_memory"         # in_memory, opensearch, etc.
  schema_store: "schemas.yml"
  examples_store: "examples.yml"
  business_rules: "rules.yml"

workflow_defaults:
  retry_on_error: true
  max_retries: 2

agents:
  high_level:
    type: "chain_of_agents"
    sub_agents:
      - data_query
      - data_analysis
      - data_visualization
```

- **Loader**: write a `ConfigLoader` that reads/validates this against a Pydantic schema.

---

## 3. Core Component Abstractions

### 3.1 BaseComponent
```python
class BaseComponent(ABC):
    def __init__(self, config: dict): ...
    @abstractmethod
    def run(self, **kwargs) -> Any: ...
```

### 3.2 Code Executors
- **CodeExecutorBase** ← `BaseComponent`
- **InMemoryExecutor** → executes Python code via `exec`/`eval`
- **ApiExecutor** → sends code to user’s API endpoint, handles HTTP, timeouts, retries

### 3.3 LLM Component
- **LlmComponentBase** ← `BaseComponent`
- **OpenAIComponent**, **CohereComponent**, etc.
- Methods: `generate(prompt: str, **kwargs) -> str`

### 3.4 Retrieval Component
- **RetrievalBase** ← `BaseComponent`
- **InMemoryRetrieval**, **OpensearchRetrieval**
- Methods:
  - `get_schema(table_names: List[str]) -> Dict`
  - `get_examples(query: str) -> List[Example]`
  - `get_business_rules(context: str) -> List[str]`

---

## 4. Data-Querying Pipeline

1. **Query Rewriting**
   - Component: `QueryRewriter` (LLM-based)
   - Input: raw user question → Output: normalized/focused query

2. **Asset Retrieval**
   - Use `RetrievalBase` to fetch:
     - schema snippets
     - query→code examples
     - business rules
   - Assemble into context bundle

3. **Prompt Composition**
   - Template driven (Jinja2):
     ```
     You are a SQL generation assistant.
     Schema:
     {{ schema }}
     Business rules:
     {{ rules }}
     Examples:
     {{ examples }}
     Question:
     {{ rewritten_query }}
     Generate code only.
     ```

4. **Code Generation**
   - Call `LlmComponent.generate(prompt)`

5. **Code Execution**
   - Pass to `CodeExecutor.run(code)`
   - Handle success: return dataframe or JSON
   - On error: capture exception, if `retry_on_error` then automatically regenerate with “error context” appended.

6. **Result Wrapping**
   - Always return a standardized `QueryResult` object:
     ```python
     class QueryResult(NamedTuple):
         data: Any
         code: str
         logs: List[str]
     ```

---

## 5. Data-Analysis & Visualization

- **Analysis**
  - Two modes:
    1. **Prebuilt functions**: e.g. `mean, groupby, pivot_table` wrappers
    2. **LLM-generated code**: similar pipeline to querying, but prompt tuned for pandas/matplotlib

- **Visualization**
  - Prebuilt chart functions (e.g. `bar_chart(data, x, y)`)
  - Or generate code via LLM (prompt includes “Use matplotlib to plot…”)

- **Component Hooks**
  - `AnalysisComponent`, `VisualizationComponent` ← `BaseComponent`
  - Each takes config to choose “mode: prebuilt|llm” plus parameters

---

## 6. Workflow Graphs (LangGraph-based)

1. **Define Nodes**
   - Each component (QueryRewriter, Retrieval, LLM, Executor, Analyzer, Visualizer) is a node.

2. **Define Edges**
   - Query → Rewrite → Retrieve → Generate → Execute → Analyze → Visualize

3. **Config-Driven Graph**
   ```yaml
   workflows:
     simple_query:
       nodes:
         - name: rewrite
           component: QueryRewriter
         - name: retrieve
           component: InMemoryRetrieval
         - name: generate
           component: OpenAIComponent
         - name: execute
           component: ApiExecutor
       edges:
         - [rewrite, retrieve]
         - [retrieve, generate]
         - [generate, execute]
   ```

4. **Workflow Runner**
   - Reads config, instantiates components, runs graph.

5. **Pluggability**
   - To swap in CrewAI’s orchestrator, write an adapter that takes the same graph config.

---

## 7. Agent Definitions

- **High-Level Agent**
  - Delegates to sub-agents:
    - **DataQueryAgent**
    - **DataAnalysisAgent**
    - **DataVizAgent**

- **Agent API**
  ```python
  class Agent(ABC):
      def __init__(self, tools: Dict[str, BaseComponent]): ...
      @abstractmethod
      def run(self, input: str) -> Any: ...
  ```

- **Multi-Agent Flow**
  1. User question → HighLevelAgent
  2. Based on analysis: call DataQueryAgent or DataAnalysisAgent, etc.
  3. Chained back-and-forth until final output.

- **Tool Registration**
  - Each Agent constructor takes a `tools` dict populated from components.

---

## 8. Error Handling & Retries

- **Centralized Error Types**
  - `CodeExecutionError`, `LLMGenerationError`, `RetrievalError`
- **Retry Decorator**
  - Applies to any component with `retry_on_error` from config
- **User Feedback Loop**
  - On repeated failures, Agent can ask the user for clarification.

---

## 9. Extensibility & Adapter Pattern

- **Interface Definitions**
  - Every component interface in `components/interfaces.py`
- **Registration Mechanism**
  - Entry points or a simple factory:
    ```python
    COMPONENT_REGISTRY = {
      "openai": OpenAIComponent,
      "in_memory": InMemoryRetrieval,
      ...
    }
    ```
- **Adding New Framework**
  1. Implement the interface
  2. Register in `COMPONENT_REGISTRY`
  3. Update config schema to accept new type

---

## 10. Asset Ingestion & Retrieval Index

1. **Parsers**
   - YAML/TXT → normalized `SchemaDoc`, `ExampleDoc`, `RuleDoc`
2. **Indexers**
   - For local: simple key-value store
   - For Opensearch: bulk ingest API wrapper
3. **Retriever**
   - Embedding-based or keyword matching

---

## 11. Testing & CI

- **pytest**
  - Unit tests for each component interface
  - End-to-end: config → workflow run on toy CSV
- **Linting**
  - `ruff --fix`
- **Type Checking**
  - `mypy dataqa/`

---

## 12. Documentation & Examples

- **README**
  - Quickstart:
    ```python
    from dataqa import DataQALibrary
    lib = DataQALibrary(config="config.yml")
    result = lib.query("Show me total sales by region")
    lib.visualize(result.data, kind="bar")
    ```
- **Sphinx**
  - Auto-generate API, plus how-to guides for:
    - Adding a new LLM provider
    - Writing custom components
    - Building an Agent vs. Workflow

---

### Next Steps

1. **Feed this plan** (or chunks of it) to your LLM, asking it to scaffold each module.
2. **Iterate**: once code is generated, run tests and refine configs.
3. **Extend**: drop in new orchestrators (CrewAI, Autogen) by writing thin adapters.

With this blueprint, your LLM has a clear map—every package, module, config file, class hierarchy, and workflow definition it needs to generate the full `dataqa` library in a **configuration-driven**, **extensible** way.
