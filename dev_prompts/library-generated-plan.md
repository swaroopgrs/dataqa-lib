Okay, acting as an amazing software architect, I've reviewed all the provided LLM responses based on your original idea. The responses offer excellent, detailed perspectives, especially Gemini Deep Research v2 and Gemini Pro v2 providing strong structural and phased plans, while others highlight specific patterns (Adapters, RAG strategies, API contracts, Agent roles).

My goal is to synthesize these into a **unified, actionable plan** that captures the core requirements and architectural decisions, making it suitable (and more concise than the combined input) to feed to an LLM for code generation.

This unified plan prioritizes:

1.  **Clarity and Structure:** Logical organization for the LLM.
2.  **Key Architectural Decisions:** Emphasizing modularity, configurability, and extensibility.
3.  **Core Functionality:** Focusing on the essential components and workflows.
4.  **Conciseness:** Removing redundancy while retaining necessary detail.

---

## Unified Plan for Generating the `dataqa` Library

**Objective:** Generate the Python source code for the `dataqa` library, a modular, config-driven framework for natural language querying, analysis, and visualization of data via a user-provided execution API.

**Core Principles:**

1.  **Config-Driven:** Use YAML parsed into Pydantic models for all configurations (LLMs, Components, KB, Execution API, Workflows, Agents). Use Pydantic `BaseSettings` for loading/validation.
2.  **Component-Based Architecture:** Define reusable, modular components for specific tasks (query rewrite, RAG, code gen, execution, analysis, viz).
3.  **Framework Extensibility (Orchestration):** Abstract the orchestration layer (Workflows/Agents). Implement initially for LangGraph, but use an Adapter pattern (`BaseOrchestrationAdapter`) to allow easy integration of CrewAI, Autogen, etc., later without core changes.
4.  **External Code Execution:** Generate code (SQL, Python) but execute it via a user-provided external API (Code Execution API). Define a clear contract.
5.  **Dual Mode (Workflows & Agents):** Support both predefined, graph-based Workflows (static orchestration) and dynamic, LLM-driven Agents (multi-agent, supervised pattern). Components must serve as building blocks for both.
6.  **Standard Dev Practices:** Use Poetry, Ruff, Pytest, Sphinx (with specified style), and maintain a clear project structure (`src/dataqa/...`).

**Phase 1: Project Setup & Core Abstractions**

1.  **Initialize Project:** Use Poetry (`pyproject.toml`), set up Ruff, Pytest, Sphinx. Define core dependencies (`pyyaml`, `pydantic`, `pydantic-settings`, `langchain-core`, `langgraph`, `langchain-openai`).
2.  **Directory Structure:**
    ```
    dataqa/
    ├── src/
    │   └── dataqa/
    │       ├── __init__.py
    │       ├── config/             # Pydantic models, loader
    │       ├── core/               # Base interfaces/classes
    │       ├── components/         # Reusable components (querying, analysis, viz, execution...)
    │       ├── knowledge_base/     # KB interfaces, implementations (local, opensearch), RAG logic
    │       ├── execution/          # Code Execution API client logic
    │       ├── orchestration/      # Workflow/Agent logic, Adapters (LangGraph, Base)
    │       └── utils/              # Helpers, logging
    ├── tests/
    ├── docs/
    ├── examples/           # Sample configs, assets, usage scripts
    ├── pyproject.toml
    ├── README.md
    └── .gitignore
    ```
3.  **Define Core Interfaces (`dataqa/core/interfaces.py` or similar):**
    * `BaseComponent(ABC)`: `execute(self, context: Dict) -> Dict` method. Must be adaptable as an agent tool.
    * `BaseLLMService(ABC)`: Interface for LLM interaction (`generate`, `generate_streaming`).
    * `BaseKnowledgeBase(ABC)`: Interface for asset storage/retrieval (`add_asset`, `retrieve_assets`).
    * `BaseRetrievalStrategy(ABC)`: Interface for retrieval logic (dense, sparse, hybrid, tag).
    * `BaseCodeExecutorClient(ABC)`: Interface for interacting with the *user's* external execution API.
    * `BaseOrchestrationAdapter(ABC)`: Interface to abstract agent/workflow frameworks (`build_workflow`, `run_workflow`, `build_agent`, `run_agent`).
4.  **Define Configuration Models (`dataqa/config/models.py`):** Use Pydantic for:
    * `LLMConfig`, `KnowledgeBaseConfig`, `ExecutionAPIConfig`, `ComponentConfig`, `WorkflowConfig`, `AgentConfig`, `DataQAConfig` (main). Use `SecretStr` for sensitive data.

**Phase 2: Configuration & Foundational Services**

5.  **Implement Config Loader (`dataqa/config/loader.py`):** Load YAML, validate with Pydantic models, handle environment variables/`.env` via `BaseSettings`.
6.  **Implement LLM Service (`dataqa/core/llm_service.py`):** Implement `AzureOpenAIService(BaseLLMService)`. Load config via Pydantic.
7.  **Implement Knowledge Base (`dataqa/knowledge_base/`):**
    * Implement `LocalKnowledgeBase(BaseKnowledgeBase)` (in-memory/JSON for testing).
    * Define interface/structure for `OpenSearchKnowledgeBase` (implementation later if needed).
    * Implement Asset Ingestion logic (parsing schema YAML, rule TXT, example YAML/JSON).

**Phase 3: Core Component Implementation (`dataqa/components/`)**

8.  **Implement Retrieval Strategies (`dataqa/knowledge_base/retrieval.py`):** Implement `DenseRetrieval`, `SparseRetrieval` (BM25), `HybridRetrieval`, `TagBasedFilter` strategies conforming to `BaseRetrievalStrategy`. These interact with the `BaseKnowledgeBase`.
9.  **Implement `AssetRetrieverComponent(BaseComponent)`:** Uses configured `BaseKnowledgeBase` and `BaseRetrievalStrategy` to fetch context (schema, examples, rules) based on input query and config.
10. **Implement `QueryRewriterComponent(BaseComponent)`:** Uses configured `BaseLLMService` to rewrite the input query for clarity/retrieval.
11. **Implement `PromptComposerComponent(BaseComponent)`:** Takes rewritten query, retrieved assets, task type, and configured templates (Jinja2 recommended) to build the final LLM prompt.
12. **Implement `CodeGeneratorComponent(BaseComponent)`:** Takes the composed prompt, uses `BaseLLMService` to generate code (SQL/Python).
13. **Implement `ApiCodeExecutorClient(BaseCodeExecutorClient)` (`dataqa/execution/client.py`):**
    * Implements the client side for the *recommended* **Asynchronous REST + Polling** pattern for the user's external API.
    * Takes API config (URLs, auth) from Pydantic model.
    * Methods: `submit_code(code, language) -> job_id`, `get_status(job_id) -> status_info` (including results or structured error).
    * Handles HTTP requests, responses, and errors robustly.
    * *Alternative:* Also define a `PythonExecutorAdapter(BaseCodeExecutorClient)` interface for users who prefer direct Python integration instead of a network API.
14. **Implement `CodeExecutionComponent(BaseComponent)`:** Acts as a facade. Takes generated code, uses the configured `BaseCodeExecutorClient` (either API or Python adapter) to execute it, and handles the submit/poll loop. Manages retries based on structured errors.
15. **Implement `AnalysisComponent(BaseComponent)` & `VisualizationComponent(BaseComponent)`:**
    * Support two modes (configurable):
        * `code_generation`: Use `CodeGeneratorComponent` -> `CodeExecutionComponent`.
        * `predefined_functions`: Map request to pre-built Python functions.
    * Define clear input (data, request) and output structures (`AnalysisResult`, `VisualizationResult`).

**Phase 4: Orchestration (LangGraph Implementation)**

16. **Implement LangGraph Adapter (`dataqa/orchestration/adapters/langgraph_adapter.py`):**
    * Implement `LangGraphAdapter(BaseOrchestrationAdapter)`.
    * `build_workflow`: Takes workflow config (nodes mapped to `dataqa` components, edges, state schema), builds and compiles a LangGraph `StateGraph`. Handles node execution by calling the respective component's `execute` method. Implement conditional edges for error handling/branching.
    * `run_workflow`: Executes `compiled_graph.invoke()`.
    * `build_agent`: Takes agent config (type, LLM, tools mapping to `dataqa` components), builds a LangGraph agent executor (e.g., using agent toolkit functions). Adapts `BaseComponent` to be used as LangGraph tools.
    * `run_agent`: Executes `agent_executor.invoke()`.
17. **Define Sample Workflow/Agent Configs (`examples/`):** Create YAML examples for:
    * A data querying workflow (Rewrite -> Retrieve -> Compose -> Generate -> Execute -> Handle Error). Include state schema and conditional logic for the error/retry loop.
    * A multi-agent system (Coordinator -> DataQueryAgent -> DataAnalysisAgent -> DataVizAgent). Define agent roles, tools (mapping to components), and prompts.

**Phase 5: Execution Handling, Iteration & API Contract**

18. **Define Code Execution API Contract (Documentation):** Clearly document the recommended **Asynchronous REST + Polling** contract for the user's external API endpoint:
    * `POST /execute` (Request: `{code, language}`, Response: 202 Accepted + `{job_id}`).
    * `GET /status/{job_id}` (Response: `{status: pending|running|success|error, results: ..., error_type: ..., error_message: ..., logs: ...}`).
    * Emphasize the need for **structured error reporting** (`error_type`, `error_message`).
    * Also document the *alternative* `PythonExecutorAdapter` interface for direct integration.
19. **Implement Error Handling Loop (within Orchestration):**
    * **Workflows (LangGraph):** Use conditional edges based on `CodeExecutionComponent`'s output status (`execution_success`). If error, transition to an error analysis node (could be a simple retry counter or an LLM call to analyze the error) which might loop back to `CodeGeneratorComponent` with error context.
    * **Agents (LangGraph):** The `CodeExecutionComponent` (acting as a tool) returns the structured error. The agent's LLM decides the next step (retry component, inform user, etc.) based on the error details and its prompt instructions.

**Phase 6: Development Standards, Testing & Documentation**

20. **Implement Linting/Formatting:** Configure `ruff` in `pyproject.toml`. Use pre-commit hooks.
21. **Write Tests (`tests/`):** Use `pytest`. Create unit tests (mocking dependencies like LLM calls, KB, Execution API client) and integration tests (testing component interactions within workflows/agents, focusing on the error handling loop). Use fixtures extensively.
22. **Write Documentation (`docs/`):** Use `sphinx`.
    * Docstrings (Sphinx style) for all public modules/classes/functions.
    * Pages: Introduction, Installation, Quickstart (using examples), Configuration Guide (YAML reference), Components Guide, Knowledge Base Guide, Retrieval Guide, Workflow Guide, Agent Guide, **Code Execution API Contract** (critical), Developer Guide (contributing, adding components/adapters), API Reference (autodoc).
23. **Create README (`README.md`):** Include overview, installation, quick start (simple config + run command), link to docs, developer setup, contribution guidelines.

**LLM Instructions:**

* Generate the Python code based on the structure and phases above.
* Implement classes and interfaces as described.
* Use Pydantic extensively for configuration and data models.
* Ensure components conform to `BaseComponent` and are usable in both workflows and as agent tools.
* Implement the `LangGraphAdapter` as the initial orchestration mechanism.
* Implement the client side for the asynchronous Code Execution API pattern.
* Include robust error handling and propagation, especially around code execution.
* Adhere to specified development standards (Poetry, Ruff, Pytest, Sphinx).
* Generate necessary boilerplate (e.g., `__init__.py` files).
* Provide clear type hints and docstrings.

---

This unified plan is significantly shorter than the combined input while retaining the essential structure, requirements, and key design decisions needed to guide the LLM effectively.
