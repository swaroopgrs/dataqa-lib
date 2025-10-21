Okay, acting as an amazing software architect, I've reviewed your idea and the diverse, detailed responses from the LLMs. Each response offers valuable perspectives and details. My goal is to synthesize these into a single, coherent, and highly detailed architectural blueprint for the `dataqa` library. This unified plan incorporates the best ideas and structures from the provided responses, aiming for clarity, robustness, and extensibility, suitable for guiding an LLM in code generation.

**Unified Architectural Blueprint for `dataqa` Library**

**I. Introduction & Vision**

`dataqa` will be a modular, configuration-driven Python library enabling natural language interaction (querying, analysis, visualization) with data sources accessible via SQL or APIs/functions. It supports both predefined `Workflows` for repeatable tasks and dynamic, LLM-powered `Agents` for complex, exploratory interactions. The architecture prioritizes loose coupling, extensibility (especially regarding agent frameworks), and reliance on a user-provided external API for code execution, insulating `dataqa` from direct database dependencies. The initial target orchestration framework is LangGraph, but the design facilitates future integration of CrewAI, Autogen, Pydantic AI, etc.

**II. Core Architectural Principles**

1.  **Configuration-Driven (Pydantic):** All behavior (components, LLMs, KB, retrieval, execution API, workflows, agents) is defined via YAML configuration. Pydantic (`BaseSettings`) will be used for schema definition, validation, and loading (from files, environment variables), ensuring type safety and clarity. (Ref: R2, R6, R7)
2.  **Modularity & Componentization:** Functionality is encapsulated in reusable `DataQAComponent` instances. These components handle specific tasks (e.g., query rewrite, code gen, execution) and can be used within workflows or as tools by agents. (Ref: R1, R2, R5, R7)
3.  **Dependency Injection (DI):** Components receive dependencies (config, LLM clients, other components) externally, promoting loose coupling and testability. A simple factory or DI container can manage instantiation based on configuration. (Ref: R2, R6)
4.  **Extensibility via Adapters:** An `Adapter` pattern isolates core components from specific orchestration framework implementations (LangGraph, CrewAI). Adapters translate the generic `DataQAComponent` interface into framework-specific constructs (nodes, tools). (Ref: R2, R6, R7)
5.  **Loose Coupling:** Minimize direct dependencies between components and frameworks. The separation between code generation (`dataqa`) and execution (user's API) is paramount. (Ref: R2, R6)
6.  **Framework Agnosticism:** While starting with LangGraph, core logic resides in components, independent of the orchestration framework. Adding new frameworks primarily involves creating new Adapters. (Ref: User Idea, R2, R7)

**III. Project Structure & Development Standards**

1.  **Project Layout (`src/` layout recommended):** (Ref: R1, R5, R6, R7)
    ```
    dataqa/
    ├── docs/              # Sphinx documentation source
    ├── src/
    │   └── dataqa/        # Main library package
    │       ├── __init__.py
    │       ├── config/      # Pydantic models, loader
    │       ├── core/        # Base classes, interfaces (Component, Adapter, KB, LLM, Executor)
    │       ├── components/  # Concrete component implementations (Querying, Analysis, Viz, etc.)
    │       ├── knowledge/   # KB implementations (Local, OpenSearch), Asset Parsers, Retrievers
    │       ├── execution/   # CodeExecutor interface & implementations (API Client, In-Memory)
    │       ├── llm/         # LLM provider interface & implementations (AzureOpenAI)
    │       ├── orchestration/ # Workflow/Agent runners & framework adapters
    │       │   ├── adapters/  # LangGraphAdapter, BaseAdapter interface
    │       │   └── runners/   # WorkflowRunner, AgentRunner
    │       └── utils/       # Logging, error handling, helpers
    ├── tests/             # Pytest suite (unit, integration)
    ├── examples/          # Sample configs, assets, usage scripts
    ├── .github/           # CI/CD workflows (GitHub Actions)
    ├── .gitignore
    ├── .pre-commit-config.yaml
    ├── pyproject.toml     # Poetry config, dependencies, ruff config
    ├── poetry.lock
    └── README.md
    ```
2.  **Dependency Management (Poetry):** Manage dependencies and packaging using Poetry. Define core and development (`--group dev`) dependencies in `pyproject.toml`. Commit `poetry.lock`. (Ref: R1, R2, R6, R7)
3.  **Formatting & Linting (Ruff):** Use Ruff for unified formatting and linting. Configure in `pyproject.toml`. Enforce via pre-commit hooks. (Ref: R1, R2, R6, R7)
4.  **Testing (Pytest):** Comprehensive unit and integration tests using Pytest. Leverage mocking (`unittest.mock`, `pytest-mock`) extensively for external dependencies (LLMs, Execution API, KB). Aim for high coverage. (Ref: R1, R2, R6, R7)
5.  **Documentation (Sphinx):** Generate documentation using Sphinx. Use Napoleon extension for Google/NumPy style docstrings. Include: Installation, Quick Start, Configuration Guide, Component Reference, KB Guide, Execution API Spec, Workflow/Agent Guides, API Reference (autodoc), Developer Guide. Host publicly (e.g., ReadTheDocs). (Ref: R1, R2, R6, R7)
6.  **README.md:** Comprehensive entry point. Include:
    * **User Section:** Overview, Features, Installation (`poetry install dataqa`), Quick Start Example (running a simple query workflow/agent), Link to full docs.
    * **Developer Section:** Setup (`poetry install --with dev`), Running tests (`pytest`), Building docs, Contribution guidelines, Architecture overview, **Execution API Contract**. (Ref: R1, R2, R6, R7)
7.  **Versioning (Semantic Versioning):** Follow SemVer 2.0.0. (Ref: R6)
8.  **Continuous Integration (CI):** Use GitHub Actions (or similar) to run linting, tests, and doc builds on commits/PRs. (Ref: R3, R4, R6)

**IV. Configuration System (YAML + Pydantic)**

1.  **Technology:** Use Pydantic `BaseSettings` for defining hierarchical configuration schemas, validation, and loading from YAML files, environment variables, and defaults. (Ref: R2, R6)
2.  **Structure:** Define a top-level `DataQAConfig(BaseSettings)` model nesting specific settings:
    * `LLMSettings`: Provider, model, API key (`SecretStr`), parameters.
    * `ExecutionAPISettings`: Type (e.g., `async_rest`), URLs, auth (`SecretStr`), method, request/response format hints.
    * `KnowledgeBaseSettings`: Provider (`local`, `opensearch`), connection details, default retrieval methods, embedding model config.
    * `AssetSources`: List defining paths/locations for schema, examples, rules files/directories, including tags.
    * `ComponentSettings`: Configuration overrides for specific component instances (e.g., prompt template paths for `PromptComposer`).
    * `WorkflowDefinitions`: Dictionary mapping workflow names to their definitions (e.g., LangGraph structure referencing component instances).
    * `AgentDefinitions`: Configuration for agents (framework, coordinator settings, specialized agent roles, tools mapping to component instances).
3.  **Loading:** Leverage `BaseSettings` prioritized loading (init args > env vars > `.env` file > defaults). Use `env_prefix` (e.g., `DATAQA_`) and `env_nested_delimiter` (e.g., `__`). (Ref: R6)
4.  **Validation:** Rely on Pydantic's built-in validation. Add custom validators if needed. (Ref: R6)
5.  **Secrets Management:** Use `SecretStr` for sensitive values (API keys, tokens), loaded from environment variables or secret files, *not* hardcoded. (Ref: R6)
6.  **Example Snippet (`config.yaml`):** (Ref: R2, R7)
    ```yaml
    llm:
      provider: azure_openai
      model: gpt-4o
      api_key: ${DATAQA_AZURE_API_KEY} # Env var substitution
      endpoint: ${DATAQA_AZURE_ENDPOINT}
      api_version: "..."

    execution_api:
      type: async_rest
      submit_url: "http://user-api.internal/execute"
      status_url_template: "http://user-api.internal/status/{job_id}"
      auth_token: ${DATAQA_EXEC_AUTH_TOKEN}

    knowledge_base:
      provider: local # or opensearch
      # local_storage_path: /path/to/kb_data (if local)
      # opensearch_config: {...} (if opensearch)
      default_retrieval_methods: [hybrid]
      embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

    asset_sources:
      - name: sales_schema
        type: schema
        path: assets/schema/sales.yaml
        tags: [sales, core]
      - name: sales_examples
        type: examples
        path: assets/examples/sales/
        tags: [sales]
      - name: finance_rules
        type: business_rules
        path: assets/rules/finance.txt
        tags: [finance]

    components:
      default_query_rewriter:
        class: dataqa.components.querying.LLMQueryRewriter
        config:
          llm_ref: llm # Reference the main LLM config
          prompt_template: "Rewrite this query for clarity: {query}"
      # Define other named component instances

    workflows:
      simple_sql_query:
        engine: langgraph # Specify engine per workflow
        graph: # LangGraph definition using component refs
          start_node: rewrite_query
          nodes:
            rewrite_query:
              component_ref: components.default_query_rewriter
              next: retrieve_context
            # ... other nodes ...
          edges: # Optional explicit edges if not linear

    agents:
      framework: langgraph # Default agent framework adapter
      coordinator:
        agent_type: dataqa.orchestration.agents.CoordinatorAgent # Specific class
        llm_ref: llm
        prompt_template_path: prompts/coordinator_prompt.txt
        tools: [DataQueryAgent, DataAnalysisAgent] # Calls other agents
      specialized_agents:
        - name: DataQueryAgent
          agent_type: dataqa.orchestration.agents.ToolUsingAgent
          llm_ref: llm # Can use a different model if needed
          prompt_template_path: prompts/query_agent_prompt.txt
          tools: # Maps tool names to component instances
            - query_rewriter: components.default_query_rewriter
            - asset_retriever: components.default_retriever # Assumes a retriever component exists
            - prompt_composer: components.default_composer
            - code_generator: components.default_sql_generator
            - code_executor: components.api_executor
    ```

**V. Core Component Design (`DataQAComponent` Interface)**

1.  **Base Interface (`core/interfaces.py` or `core/component.py`):** (Ref: R2, R5, R6)
    ```python
    from abc import ABC, abstractmethod
    from pydantic import BaseModel
    from typing import Any, Dict, Type, Tuple, Optional

    class DataQAComponent(ABC):
        def __init__(self, config: Dict[str, Any], dependencies: Dict[str, Any] = None):
            """Initialize with component-specific config and resolved dependencies (like LLM clients)."""
            self.config = self._validate_config(config)
            self.dependencies = dependencies or {}
            # Example: self.llm_client = dependencies.get("llm_client")

        def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
            """Validate config, potentially using a Pydantic model specific to the component."""
            # Default: return config; subclasses should implement specific validation
            return config

        @abstractmethod
        async def execute(self, input_data: BaseModel) -> Tuple[BaseModel, Dict[str, Any]]:
            """
            Executes the component's logic asynchronously.
            Args:
                input_data: Pydantic model containing validated input.
            Returns:
                A tuple containing:
                1. output_data: Pydantic model with the component's output.
                2. state_updates: Dictionary of changes to be merged into the orchestration state.
            """
            pass

        # --- Metadata for Agent Tool Usage ---
        def get_name(self) -> str:
            """Returns a unique name for the component/tool."""
            # Default implementation or abstract method
            return self.__class__.__name__

        def get_description(self) -> str:
            """Returns a natural language description for the agent LLM."""
            # Default implementation or abstract method
            return self.__doc__ or "No description provided."

        @abstractmethod
        def get_input_schema(self) -> Type[BaseModel]:
            """Returns the Pydantic model class for expected input."""
            pass

        @abstractmethod
        def get_output_schema(self) -> Type[BaseModel]:
            """Returns the Pydantic model class for the output."""
            pass
    ```
2.  **Key Components (`components/`):**
    * **Query Rewriter (`LLMQueryRewriter`):** Uses LLM to refine input query. (Ref: R1, R2, R4, R6, R7)
    * **Asset Retriever (`KnowledgeRetriever`):** Interfaces with the Knowledge Base to fetch schema, examples, rules based on query context using configured retrieval methods. (Ref: User Idea, R1, R7)
    * **Prompt Composer (`TemplatePromptComposer`):** Uses configured templates (e.g., Jinja2) to build the final LLM prompt from query and retrieved assets. (Ref: R1, R4, R7)
    * **Code Generator (`LLMCodeGenerator`):** Calls LLM with composed prompt to generate SQL or Python code. (Ref: R1, R2, R7)
    * **Analysis Component (`AnalysisExecutor`):** Performs analysis via code generation (using `LLMCodeGenerator` + `CodeExecutor`) or pre-built functions. Configurable mode. (Ref: R2, R5, R7)
    * **Visualization Component (`VisualizationGenerator`):** Generates visualizations via code generation or pre-built functions. Configurable mode. (Ref: R2, R5, R7)
    * **Error Analysis Component (`LLMErrorAnalyzer`):** (Optional, could be part of agent logic) Takes execution error, analyzes it, suggests corrections for retry loop.

**VI. Knowledge Base & Asset Management (`knowledge/`)**

1.  **Asset Types:** Schema (YAML/JSON), Examples (YAML/JSON with query, code, optional reasoning), Business Rules (TXT/MD). Allow tagging. (Ref: User Idea, R2, R6)
2.  **Ingestion:** Component/script to parse asset files based on configuration (`asset_sources`) and load them into the chosen KB provider. Handle structure, metadata, and tags. (Ref: R1, R2, R5)
3.  **KB Providers (`knowledge/providers/`):**
    * `BaseKnowledgeBase` (ABC): Interface defining methods like `add_document`, `retrieve`.
    * `LocalKnowledgeBase`: Implementation using local file storage (e.g., JSON/Parquet for metadata) and an in-memory or file-based vector store (e.g., FAISS, ChromaDB). Suitable for testing/small scale. (Ref: R1, R2, R5)
    * `OpenSearchKnowledgeBase`: Implementation interfacing with OpenSearch (requires `opensearch-py`). Uses k-NN for dense, text search for sparse. (Ref: User Idea, R1, R2, R5)
4.  **Retrieval Methods (`knowledge/retrievers/`):**
    * `BaseRetriever` (ABC): Interface defining `retrieve` method.
    * Implementations for Dense (vector similarity), Sparse (BM25), Hybrid (RRF or weighted combination), Tag-based filtering/boosting. Configurable via `KnowledgeBaseSettings` and retrieval calls. Need embedding models (configurable). (Ref: User Idea, R1, R2, R5, R6)

**VII. External Code Execution (`execution/`)**

1.  **Decoupling:** `dataqa` *generates* code, the user's external service *executes* it.
2.  **Recommended Pattern:** Asynchronous REST API with polling. `dataqa` POSTs code, gets `job_id`, polls status endpoint. (Ref: R6)
3.  **API Contract Definition:** Clearly document the expected request/response JSON structure for the `POST /execute` and `GET /status/{job_id}` endpoints, including structured error reporting (`status`, `error_type`, `error_message`, `logs`). (Ref: R2, R6, R7)
4.  **`CodeExecutor` Interface (`execution/interface.py`):**
    * `BaseCodeExecutor` (ABC): Defines `execute(code: str, language: str, context: Dict) -> ExecutionResult`.
    * `ExecutionResult` (Pydantic Model): Contains `success: bool`, `result_data: Any`, `error_type: Optional[str]`, `error_message: Optional[str]`, `logs: Optional[List[str]]`.
5.  **Implementations (`execution/providers/`):**
    * `AsyncRestClientExecutor`: Implements the client-side logic for the recommended Async REST + Polling pattern using `httpx`. Handles auth, timeouts, polling, response parsing. (Ref: R6)
    * `(Optional) InMemoryExecutor`: Executes Python code directly (use `exec` carefully in a sandboxed environment) for testing or simple Pandas operations *if explicitly configured and security implications understood*. (Ref: R3, R4)
    * `(Optional) PythonInterfaceExecutor`: Allows users to provide a Python class implementing the `BaseCodeExecutor` interface directly, bypassing the network API. (Ref: R6 Insight 5)
6.  **Error Handling & Retry Loop:**
    * `AsyncRestClientExecutor` captures network errors and execution errors from the API response.
    * In **Workflows**, error handling is defined by graph structure (conditional edges). Simple retries possible.
    * In **Agents**, the `CodeExecutor` tool returns the `ExecutionResult` (including errors). The agent LLM analyzes the error (using description and `error_type`/`message`) and decides the next action:
        * Re-invoke `LLMCodeGenerator` with error context for correction.
        * Retry execution if transient.
        * Fail or escalate. (Ref: R2, R6, R7)

**VIII. Orchestration Layer (`orchestration/`)**

1.  **Workflows (`orchestration/runners/WorkflowRunner.py`):**
    * Reads workflow definitions from config.
    * Instantiates required `DataQAComponent` instances.
    * Uses the configured engine (initially LangGraph via `LangGraphAdapter`) to build and run the graph.
    * Manages state passing between components based on graph definition.
    * Uses LangGraph's Graph API style. (Ref: User Idea, R1, R2, R4)
2.  **Agents (`orchestration/runners/AgentRunner.py`):**
    * Reads agent definitions from config.
    * Instantiates `DataQAComponent` instances to be used as tools.
    * Loads the configured `AgentAdapter` (e.g., `LangGraphAdapter`).
    * Uses the adapter to initialize and run the agent(s) (Coordinator, specialized).
3.  **Agent Framework Abstraction (`orchestration/adapters/`):**
    * `BaseAgentAdapter` (ABC): Defines the interface for interacting with different agent frameworks. Key methods: `initialize_agent(config, tools)`, `run_agent(agent_instance, input)`. (Ref: R2, R7)
    * `LangGraphAdapter`: Implementation using LangGraph. Translates `dataqa` config and components into LangGraph `StateGraph`, nodes (wrapping component `execute`), tools, and agent executors (`create_react_agent`, etc.). (Ref: R1, R2, R7)
    * (Future): `CrewAIAdapter`, `AutoGenAdapter`, etc. implementing the `BaseAgentAdapter` interface.
4.  **Multi-Agent System (`agents/` section in config, potentially `orchestration/agents/` for base classes):**
    * Implement the Hierarchical Supervisor pattern. (Ref: R6)
    * `CoordinatorAgent`: Defined in config, uses other agents as tools. Orchestrates the overall task. (Ref: R3, R6, R7)
    * `DataQueryAgent`, `DataAnalysisAgent`, `DataVisualizationAgent`: Defined in config, use `DataQAComponent` instances as tools. Focus on their specific tasks. (Ref: R3, R6, R7)
    * Communication primarily through the Coordinator. State managed by the orchestrator/adapter.

**IX. Operational Considerations**

1.  **Real-time vs. Batch:** The component design is suitable for both. Real-time interaction depends on LLM/API latency. Batch processing involves scripting `dataqa` to run over multiple inputs, disabling interactive elements (like HIL) via config. Log errors robustly but continue batch. (Ref: R1, R2, R6)
2.  **Human-in-the-Loop (HIL):** Design components and orchestration with optional hooks/states where execution can pause for human review (e.g., query clarification, code approval, error resolution). Make HIL configurable. (Ref: R2, R3, R6)

**X. Conclusions & Recommendations**

This unified blueprint provides a robust foundation for `dataqa`.

* **Key Strengths:** Modularity, configurability, extensibility (framework-agnostic design via Adapters), clear separation of concerns (generation vs. execution).
* **Recommendations:**
    * **Prioritize Core:** Focus initially on the `Data Querying` component, `AsyncRestClientExecutor`, `LocalKnowledgeBase`, `LangGraphAdapter`, and the core Pydantic configuration system.
    * **API Contract First:** Finalize and clearly document the External Code Execution API contract early. Consider providing the optional Python interface adapter.
    * **Iterative Development:** Build workflows first, then the more complex agent system. Implement basic RAG before advanced techniques.
    * **Rigorous Testing:** Emphasize integration tests for the query->generate->execute->refine loop, heavily mocking external services.
    * **Documentation is Crucial:** Maintain thorough documentation, especially for configuration and the Execution API.

By following this detailed plan, leveraging the strengths identified in the LLM responses, an LLM can be effectively guided to generate the `dataqa` library, fulfilling your vision for a flexible and powerful natural language data interaction framework.
