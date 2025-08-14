Great, I’ll prepare a detailed step-by-step breakdown for building the `dataqa` library as a fully modular, YAML-configurable Python framework. It will support LangGraph-based workflows and agents with future extensibility in mind. I’ll include guidelines for component structure, config schema, knowledge base ingestion, LLM interaction, and execution feedback handling.

I’ll let you know once the blueprint is ready.

# dataqa Framework Design

**Overview:** The `dataqa` library will be a modular, YAML-driven Python framework for answering user questions about data. It will use multiple LLM-based agents (a coordinator plus specialized sub-agents) organized in a LangGraph-style workflow.  Users provide YAML or text assets (database schema, business rules, example Q&A, etc.) in a local or OpenSearch-backed “Asset Store.”  The system will generate SQL or Python code via LLM prompts (using Azure OpenAI by default) and execute it via a pluggable CodeExecution API.  The architecture follows modern agentic AI patterns: each agent has defined tools and roles, and a high-level coordinator orchestrates them ([What is crewAI? | IBM](https://www.ibm.com/think/topics/crew-ai#:~:text=Agentic%20frameworks%20are%20AI%20agent,gen%20AI%20applications%20to%20function)) ([What is crewAI? | IBM](https://www.ibm.com/think/topics/crew-ai#:~:text=crewAI%20is%20an%20open%20source,1)).  All components are configurable in YAML, and the design is extensible to future frameworks (e.g. CrewAI, AutoGen, Pydantic-AI) through abstract interfaces. 

## 1. Configuration and Assets 
- **YAML Config Files:** Use a hierarchical YAML configuration for all logic. For example, one might use [Hydra](https://hydra.cc) or a similar YAML loader to manage config namespaces ([Getting started | Hydra](https://hydra.cc/docs/intro/#:~:text=Hydra%20is%20an%20open,a%20Hydra%20with%20multiple%20heads)).  A top-level `dataqa.yaml` could define global settings (e.g. LLM model parameters, execution settings, agent roles) and reference other files. Sub-files describe domain assets: 
  - `schema.yaml` – database schema (tables, columns, datatypes).  
  - `rules.yaml` – business rules, constraints, calculation definitions.  
  - `examples.yaml` – example questions and answers.  
  - `prompts.yaml` – prompt templates for querying, analysis, and visualization.  
  Each of these assets is editable as YAML or text. The config loader should merge/override settings dynamically (e.g. via Hydra-style composition ([Getting started | Hydra](https://hydra.cc/docs/intro/#:~:text=Hydra%20is%20an%20open,a%20Hydra%20with%20multiple%20heads))).
- **Asset Store Abstraction:** Define a `AssetStore` interface for retrieving assets (by keyword or query). Provide at least two implementations: a LocalFileStore (reads YAML/txt from a directory structure) and an OpenSearchStore (searches a local OpenSearch index of the assets for relevant snippets). This allows retrieval of relevant schema or examples for a given user question (a RAG-style approach). For example, a tool might query the AssetStore to fetch the `schema.yaml` and relevant business rules. The YAML config would specify which store to use and index settings.

## 2. Core Modules and Interfaces 
- **LLM Client:** An abstract `LLMClient` interface (e.g. `generate(prompt:str) -> str`). Provide an `AzureOpenAIClient` implementation for Azure OpenAI GPT models. This client reads credentials from environment or YAML and handles rate-limiting, retries, and logging. The LLM client will be used by all agents to generate SQL or analysis code from natural language prompts.
- **Prompt Component:** A reusable component that constructs LLM prompts from user input, context assets, and prompt templates (from `prompts.yaml`). This prompt-builder should be configurable in YAML (templates with placeholders for schema, rules, examples, etc.). For example, the Data Querying agent might load a “SQL generation” prompt template, fill in the user question plus schema snippet, and call the LLM client.
- **Agent Base Class:** Define a base `Agent` class or interface (e.g. with methods `run(task:Task)` and `on_message()` if using conversational structure). Each agent has a name, a set of tools/components it can use, and access to the shared environment (assets, execution API).  
- **Tool Abstraction:** Agents use “tools” to perform actions. Define a `Tool` base class (e.g. `perform(**kwargs)`). Tools can be things like:  
  - `SQLExecutorTool` – generates or runs SQL (uses LLM or direct DB query).  
  - `PythonExecutorTool` – executes Python code via the CodeExecution API.  
  - `DataFrameTool` – runs queries on pandas dataframes.  
  - `PlotTool` – generates data visualizations (e.g. produces Python plotting code).  
  - `AssetRetrievalTool` – fetches schema/rules/examples from the AssetStore.  
  Each tool implementation is registered in YAML (e.g. under `tools:` in the config), and the agent picks from these to complete tasks.
- **Code Executor:** Abstract a `CodeExecutor` component with `execute(code:str) -> ExecutionResult`. Provide at least: 
  - **InMemoryExecutor** for testing (executes Python or SQL in-process).  
  - **APIExecutor** which sends code to a user-defined execution endpoint (e.g. via HTTP) and returns output.  
  The CodeExecutor handles execution, captures stdout/stderr, enforces timeouts, and can retry on failure. Execution errors trigger the error-handling logic (see section 5).
- **Base Interfaces:** All of the above (AssetStore, LLMClient, Agent, Tool, CodeExecutor) should be abstract base classes or protocols. Concrete implementations are selected via YAML config (for example, `agents: data_query_agent: type: DataQueryAgent, llm_client: azure, executor: api_based`).

## 3. Agents and Workflows 
- **Coordinator Agent:** A top-level agent that orchestrates the multi-agent workflow. Its responsibilities include: interpreting the user question at a high level, deciding which sub-agents to invoke (querying, analysis, visualization), and combining their results. It may maintain global state or intermediate results. For example, given a question, the coordinator may send it to the Data Query Agent first. 
- **Data Query Agent:** Specialized agent that focuses on retrieving the data. It: 
  1. Fetches the schema and rules from the AssetStore.  
  2. Uses the LLM (with a “generate SQL” prompt) to produce a SQL query or pandas code answering the question.  
  3. Validates or reviews the generated query (optionally re-run LLM if results seem invalid).  
  4. Passes the query code to the CodeExecutor for execution, obtaining raw data.  
- **Data Analysis Agent:** Takes the raw data result and performs any additional analysis. This could include aggregations, statistical calculations, or data cleaning. It can either call predefined analytics functions (e.g. Pandas, NumPy) or generate code via LLM (using an “analysis” prompt template). Output is an analysis result (e.g. a summary table).
- **Data Visualization Agent:** Given the (possibly filtered or aggregated) data, this agent generates visual output. It might either: 
  - Use LLM to propose a chart (with a prompt like “generate matplotlib code to plot this data”).  
  - Or call a built-in plotting function.  
  The output should be code for a chart (e.g. Python/Matplotlib/Plotly) which is then executed by the CodeExecutor to produce an image or chart data.
- **Graph-Style Workflow:** The agents are connected in a directed graph (LangGraph-style). For example: User → Coordinator → (DataQuery → Executor → DataAnalysis → Executor → DataVisualization → Executor) → final answer. Implement this as a workflow definition (could use a library like [networkx] or a simple DAG runner). Each node in the graph is an agent or tool action. This flow can be defined in code or even in YAML (e.g. listing steps and dependencies).
- **YAML-Driven Design:** All agent/task flows and tool chains should be configurable via YAML. For example, the YAML might specify:
  ```yaml
  agents:
    coordinator:
      role: orchestrator
      tools: [prompt_builder]
    data_query:
      role: query
      tools: [AssetRetrieval, SQLExecutor, LLMClient]
    data_analysis:
      role: analysis
      tools: [DataFrameTool, PythonExecutor]
    data_visualization:
      role: visualization
      tools: [PlotTool, PythonExecutor]
  ```
  This allows adding new agents or tools by editing the config, without code changes.

## 4. Tool and Component Hierarchy 
- **Pluggable Tool Modules:** Organize tools in a hierarchical package structure (e.g. `dataqa/tools/code_executor`, `dataqa/tools/dataframe`, `dataqa/tools/plotting`). Each sub-package can have multiple implementations. For example:
  - `dataqa/tools/code_executor/base.py` – defines the interface.  
    `dataqa/tools/code_executor/in_memory.py` and `api_based.py` – two implementations.  
  - `dataqa/tools/asset_store/base.py` – interface.  
    `local.py` (reads YAML), `opensearch.py`.  
  - `dataqa/tools/llm/base.py` – interface.  
    `azure_openai.py`.  
  - `dataqa/tools/prompt/base.py` – templating logic.
- **Generic LLM Generation Component:** A reusable prompt wrapper that can be used by any agent. For example, a `LangGraphLLMTool` that takes input variables (question, schema, rules, examples) and a template, then returns generated text. This ensures prompt logic is standardized across agents.
- **LangGraph Graph API Style Execution:** Implement a `GraphExecutor` that can run a LangGraph (workflow) of agent steps. Each node is a function call (agent or tool), edges define data flow. This mirrors LangGraph’s style of defining workflows as graphs of steps, making the execution order clear.
- **Batch vs. Real-time Modes:** Support two modes:
  - *Real-time mode:* Single-question, interactive. The user submits a question and the system runs the coordinator agent pipeline immediately and returns results. Use synchronous LLM calls with low latency.
  - *Batch mode:* Multiple questions or large jobs. The system can queue queries, possibly run agents in parallel or sequence without user waiting. Here, agents might save intermediate outputs. Configurations (in YAML) specify mode, batch size, parallelism. This affects how the LangGraph is instantiated (e.g. enabling a for-loop in the graph).

## 5. Error Handling and Human-in-the-Loop
- **Retry Logic:** All external calls (LLM generation, code execution, OpenSearch queries) should have retry wrappers. For example, the CodeExecutor can retry a failed execution up to *N* times (configurable) with exponential backoff. Similarly, LLM calls should catch timeouts or API errors and retry. This follows best practices for production AI systems ([Pydantic AI: Agent Framework. PydanticAI is a Python Agent Framework… | by Bhavik Jikadara | AI Agent Insider | Medium](https://medium.com/ai-agent-insider/pydantic-ai-agent-framework-02b138e8db71#:~:text=,retries%20and%20structured%20exception%20management)).
- **Validation Checks:** After code execution (SQL or Python), automatically check results for anomalies (e.g. empty result when not expected, SQL syntax errors, runtime exceptions). If the validation fails, the agent should either: 
  1. Automatically retry generation with revised prompt (e.g. append “The previous attempt failed, please correct errors.” to the prompt), or 
  2. Failover to a human reviewer.
- **Human-in-the-Loop Fallback:** If retries exceed threshold, the framework should escalate. For example, it could log the failure, halt execution, and alert a human. In a deployed UI, this might pop up a message like “An error occurred. Please review.”. The design should allow plugging in a “Human Review” step in the graph when needed.
- **Structured Exception Handling:** Use custom exception types (e.g. `LLMGenerationError`, `ExecutionError`, `ValidationError`) so that the workflow can branch on error type. For example, a `CodeExecutor` might raise `SyntaxErrorExecution`, and the coordinator can catch that and decide to ask the DataQueryAgent to regenerate the SQL.
- **Audit Logging:** Every agent action, prompt, and result should be logged (with correlation IDs). This helps in debugging errors and for manual review. Logging configuration (verbose vs info) should be in YAML.

## 6. Multi-Agent Coordination Patterns 
- **Specialized Agent Roles:** Follow the “crew” concept: each agent has a clear role and complementary expertise ([What is crewAI? | IBM](https://www.ibm.com/think/topics/crew-ai#:~:text=crewAI%20is%20an%20open%20source,1)) ([What is crewAI? | IBM](https://www.ibm.com/think/topics/crew-ai#:~:text=multiagent%20systems%20,distinct%20execution%20paths%20are%20required)). The DataQueryAgent handles schema and SQL; the DataAnalysisAgent handles transformations; the VisualizationAgent handles charts. The CoordinatorAgent plays a “manager” role, delegating tasks and collecting results.
- **Tool-Oriented Agents:** Agents use tools rather than doing all work themselves. For example, DataQueryAgent doesn’t parse schema itself but calls `AssetRetrievalTool` for schema and then `LLMClient` to generate SQL. This decouples functionality and makes it easy to swap components (tool-based orchestration ([What is crewAI? | IBM](https://www.ibm.com/think/topics/crew-ai#:~:text=Agentic%20frameworks%20are%20AI%20agent,gen%20AI%20applications%20to%20function))).
- **Planning and Iteration:** Implement simple planning: e.g., allow an agent to loop: “If the previous action failed or result incomplete, try a different tool or re-prompt.” This reflects agentic frameworks’ iterative refinement ([What is crewAI? | IBM](https://www.ibm.com/think/topics/crew-ai#:~:text=Agentic%20frameworks%20are%20AI%20agent,gen%20AI%20applications%20to%20function)). For instance, if DataAnalysisAgent’s first LLM attempt at analysis code is wrong, have it retry with extra context.
- **Shared Memory or Context:** Agents share context through the workflow graph. For example, the raw query result (dataframe) is passed from the DataQuery step to the DataAnalysis agent. Use a shared data structure or graph edge payloads to pass intermediate results.
- **Agent Communication (Optional):** If implementing as true “agents”, allow them to send messages to each other via the Coordinator. For example, if DataVisualization needs clarification (“should I highlight trends or absolute values?”), it can “ask” the Coordinator, which then could prompt the user or use a system prompt. (Frameworks like CrewAI allow agent chat; we can design a simplified interaction pattern.)

## 7. Extensibility and Future Framework Support 
- **Framework-Agnostic Design:** Structure the code so that agent orchestration is decoupled from specific libraries. For example, do not hardcode calls to CrewAI or AutoGen APIs. Instead, define abstract `Agent` and `Workflow` interfaces. That way, one could later implement a backend that uses CrewAI’s orchestration under the hood, or swap to AutoGen agents, by writing adapter classes that conform to our interfaces.
- **Plugin Hooks:** Allow third-party tool packages. For example, an extension pack might add new analysis tools or support a new LLM provider. The config can list these plugins. Use entry points or a plugin registry to dynamically load them.
- **Minimal Changes for New Frameworks:** To “support additional frameworks with minimal changes”, rely on interfaces. E.g. if integrating CrewAI, one could create a `CrewAIAdapter` that converts our Agent calls into CrewAI crew tasks. By keeping business logic (data QA workflows) separate, the core doesn’t need to change when plugging in a new orchestration engine.
- **Code Reuse:** Ensure tools and components are reusable in agents or as standalone workflows. For instance, the SQLExecutorTool could be used by any agent, and the LLM prompt builder is generic.

## 8. Data Querying, Analysis, Visualization Workflow 
- **Data Querying Flow:** The YAML config can define how a natural language question is turned into data retrieval:
  - Load `schema.yaml` and relevant rules/examples via AssetStore.  
  - Format a prompt like: “Given this database schema and business rules, write a SQL query to answer: ‘…question…’.”  
  - Call LLMClient → get SQL.  
  - Execute SQL via CodeExecutor → get result table.  
  - If SQL fails, re-run with additional instruction.  
  - Return table.
- **Data Analysis Flow:** If the question requires calculations beyond direct query:
  - DataQueryAgent can output initial results to a Pandas DataFrame.  
  - DataAnalysisAgent loads the DataFrame (via a DataFrameTool) and either calls known analysis functions (configured in YAML) or asks the LLM to generate Python to process the DataFrame.  
  - Example: “Analyze the data to compute averages by category.” The prompt could include the DataFrame schema. The agent runs the code and returns new table/metrics.
- **Visualization Flow:** 
  - DataVisualizationAgent takes the final data (or a subset) and generates a chart. The YAML can specify preferred chart types. The agent might prompt: “Generate Python code to plot this table with Matplotlib.”  
  - Execute code to produce an image or embed code.  
  - The final answer returned to the user includes the chart and any textual summary.
- **Workflow Patterns:** Implement common patterns like “Query → Analyze → Visualize” chains. The LangGraph API style means each step’s output is the next step’s input.

## 9. Configuration Format Example 
A sample `dataqa.yaml` could look like:
```yaml
# dataqa.yaml
global:
  mode: real-time        # or batch
  llm_model: "gpt-4o"
  max_retries: 3
  executor: api

agents:
  coordinator:
    class: CoordinatorAgent
    tools: [PromptTool]
  data_query_agent:
    class: DataQueryAgent
    tools: [AssetRetrievalTool, PromptTool, CodeExecutorTool]
  data_analysis_agent:
    class: DataAnalysisAgent
    tools: [DataFrameTool, PromptTool, CodeExecutorTool]
  data_viz_agent:
    class: DataVisualizationAgent
    tools: [PlotTool, PromptTool, CodeExecutorTool]

assets:
  schema: "assets/schema.yaml"
  rules: "assets/rules.yaml"
  examples: "assets/examples.yaml"
```
This shows how agents, tools, and assets are wired in YAML. Every name (class or file path) is configurable, enabling easy extension.

## 10. Development Environment and Best Practices 
- **Dependency Management:** Use Poetry to create an isolated Python environment and manage dependencies. The `pyproject.toml` should list dependencies like `openai`, `PyYAML` (or `hydra-core`), `requests` (for API calls), and any optional libs (`pandas`, `matplotlib`, etc.).
- **Linting and Formatting:** Enforce style with Ruff (or Black/Flake8). Configure a `ruff.toml` to lint Python code and optionally check YAML schema files. This ensures code consistency and catches errors early.
- **Testing:** Write comprehensive Pytest suites. For example, tests for: YAML config loading, LLM prompt generation (mocking LLM calls), code execution (mock or use a dummy API), and end-to-end workflows (mock agents and check data flows). Use fixtures to simulate the AssetStore and external APIs. 
- **Continuous Integration:** Set up GitHub Actions (or similar) to run `ruff --fix`, `pytest`, and build Sphinx docs on each commit/pull request.
- **Documentation:** Use Sphinx for docs generation. Document all modules, classes, and YAML config schema. Provide a `docs/usage.rst` with examples: e.g. “How to ask a question”, showing sample YAML and agent flow. Include a high-level architecture diagram (optional).
- **README:** The README should summarize `dataqa` features, show installation (via Poetry), and provide a quickstart example YAML and code snippet. It should also detail how to define the YAML assets and run a query. Developer docs should explain how to extend the system with new tools or agents.

## 11. Summary of Key Abstractions 
- **Interfaces:** 
  - `IConfigLoader` (parses YAML), 
  - `IAssetStore`, 
  - `ILLMClient`, 
  - `ICodeExecutor`, 
  - `ITool`, 
  - `IAgent`, 
  - `IWorkflowExecutor`.  
- **Implementations:** 
  - `YamlConfigLoader`, 
  - `LocalAssetStore`/`OpenSearchAssetStore`, 
  - `AzureOpenAIClient`, 
  - `InMemoryCodeExecutor`/`ApiCodeExecutor`, 
  - Various tools (SQLExecutor, DataFrameTool, etc.), 
  - Agents (CoordinatorAgent, DataQueryAgent, etc.), 
  - `LangGraphExecutor`.  
- **Config Structure:** Key YAML sections (`global`, `agents`, `assets`, etc.), illustrated above.  
- **Workflow Patterns:** Sequence of “understand → query → retrieve → analyze → visualize” with fallback loops. Each pattern corresponds to an agent chain.  
- **Error Handling:** Use retries and escalation. Inspired by Pydantic AI’s focus on production robustness (type-safety and retries) ([Pydantic AI: Agent Framework. PydanticAI is a Python Agent Framework… | by Bhavik Jikadara | AI Agent Insider | Medium](https://medium.com/ai-agent-insider/pydantic-ai-agent-framework-02b138e8db71#:~:text=,retries%20and%20structured%20exception%20management)).  
- **Extensibility:** The design keeps core logic separate from any one multi-agent library (CrewAI, AutoGen, Pydantic AI), enabling future plugging-in. For example, CrewAI and AutoGen are both multi-agent orchestration frameworks ([What is crewAI? | IBM](https://www.ibm.com/think/topics/crew-ai#:~:text=crewAI%20is%20an%20open%20source,1)) ([GitHub - microsoft/autogen: A programming framework for agentic AI  PyPi: autogen-agentchat Discord: https://aka.ms/autogen-discord Office Hour: https://aka.ms/autogen-officehour](https://github.com/microsoft/autogen#:~:text=AutoGen)); our framework only assumes abstract agent calls, so it can adapt to them with adapters.

In summary, `dataqa` will consist of well-defined Python modules (config loader, asset store, LLM client, code executor, tools, agents) connected by YAML-configured workflows.  It will follow agentic design principles (tool use and coordination) ([What is crewAI? | IBM](https://www.ibm.com/think/topics/crew-ai#:~:text=Agentic%20frameworks%20are%20AI%20agent,gen%20AI%20applications%20to%20function)) ([What is crewAI? | IBM](https://www.ibm.com/think/topics/crew-ai#:~:text=multiagent%20systems%20,distinct%20execution%20paths%20are%20required)), provide robust error handling with retries and human fallback ([Pydantic AI: Agent Framework. PydanticAI is a Python Agent Framework… | by Bhavik Jikadara | AI Agent Insider | Medium](https://medium.com/ai-agent-insider/pydantic-ai-agent-framework-02b138e8db71#:~:text=,retries%20and%20structured%20exception%20management)), and use modern development practices (Poetry, Ruff, Pytest, Sphinx).  This structure will guide an LLM to generate the actual code for each component, ensuring a modular and extensible QA system for databases.  

**Sources:** Agentic multi-agent design patterns ([What is crewAI? | IBM](https://www.ibm.com/think/topics/crew-ai#:~:text=Agentic%20frameworks%20are%20AI%20agent,gen%20AI%20applications%20to%20function)) ([What is crewAI? | IBM](https://www.ibm.com/think/topics/crew-ai#:~:text=multiagent%20systems%20,distinct%20execution%20paths%20are%20required)), examples of multi-agent frameworks (CrewAI ([What is crewAI? | IBM](https://www.ibm.com/think/topics/crew-ai#:~:text=crewAI%20is%20an%20open%20source,1)), AutoGen ([GitHub - microsoft/autogen: A programming framework for agentic AI  PyPi: autogen-agentchat Discord: https://aka.ms/autogen-discord Office Hour: https://aka.ms/autogen-officehour](https://github.com/microsoft/autogen#:~:text=AutoGen))), and production AI tooling practices (Pydantic AI’s focus on retries and type safety ([Pydantic AI: Agent Framework. PydanticAI is a Python Agent Framework… | by Bhavik Jikadara | AI Agent Insider | Medium](https://medium.com/ai-agent-insider/pydantic-ai-agent-framework-02b138e8db71#:~:text=,retries%20and%20structured%20exception%20management)), Hydra for YAML configs ([Getting started | Hydra](https://hydra.cc/docs/intro/#:~:text=Hydra%20is%20an%20open,a%20Hydra%20with%20multiple%20heads))).