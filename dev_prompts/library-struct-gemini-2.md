Okay, here's a detailed proposal for the folder structure and a common config structure for your `dataqa` library.

## 1. Detailed Folder Structure

This folder structure is designed to be modular, extensible, and to support the requirements of building config-driven agents and workflows for natural language querying of databases.

```
dataqa/
├── __init__.py
├── configs/                  # Configuration handling
│   ├── __init__.py
│   ├── base_config.py        # Base Pydantic models for configurations (e.g., LLMConfig, KBConfig)
│   ├── graph_config_schema.py # Pydantic model for the main graph/agent/workflow config (validates the YAML)
│   └── loader.py             # Utilities to load, validate, and parse YAML configurations
├── core/                     # Core abstractions and base classes
│   ├── __init__.py
│   ├── base_component.py     # Abstract base class for all components
│   ├── base_agent.py         # Abstract base class for agents
│   ├── base_workflow.py      # Abstract base class for workflows (if distinct from agents)
│   ├── base_state.py         # Base Pydantic models for graph/agent state (e.g., merging BasePipelineState and CwdState ideas)
│   └── errors.py             # Custom error classes (e.g., ConfigError, KBError, ExecutionError)
├── components/               # Reusable building blocks for agents and workflows
│   ├── __init__.py
│   ├── llm/                  # LLM interaction components
│   │   ├── __init__.py
│   │   ├── llm_call.py         # Generic LLM call component (invokes LLM with prompt, returns response)
│   │   ├── prompt_builder.py   # Component for dynamically building prompts from templates and retrieved assets
│   │   └── output_parser.py    # Component for parsing LLM outputs (e.g., to Pydantic models, JSON)
│   ├── data_querying/
│   │   ├── __init__.py
│   │   ├── query_rewriter.py   # Rewrites user query based on context/history
│   │   ├── asset_retriever.py  # Retrieves relevant assets (schema, examples, rules) from Knowledge Base
│   │   ├── code_generator.py   # Generates code (SQL, Python) based on query and assets
│   │   └── code_executor/      # Executes generated code
│   │       ├── __init__.py
│   │       ├── base_executor.py  # Base code executor interface
│   │       ├── sql_executor.py   # Executes SQL against various DBs (Snowflake, Databricks)
│   │       ├── python_executor.py# Executes Python code (e.g., for in-memory pandas operations)
│   │       ├── function_executor.py # Executes predefined functions
│   │       └── api_executor.py     # Executes API calls
│   ├── data_analysis/
│   │   ├── __init__.py
│   │   ├── analysis_generator.py # Generates code or textual analysis
│   │   └── prebuilt_analyzers.py # Wrappers for pre-built analysis functions/tools
│   ├── data_visualization/
│   │   ├── __init__.py
│   │   ├── viz_code_generator.py # Generates visualization code (e.g., Matplotlib, Plotly)
│   │   └── prebuilt_visualizers.py# Wrappers for pre-built visualization functions/tools
│   └── general/                # General utility components
│       ├── __init__.py
│       ├── router.py           # Conditional routing logic (e.g., for LangGraph conditional edges)
│       ├── gather_node.py      # Gathers inputs for final output (similar to provided example)
│       └── state_modifier.py   # Component for direct state manipulation if needed
├── knowledge_base/           # Knowledge base management
│   ├── __init__.py
│   ├── base_kb_store.py      # Abstract interface for a knowledge base store
│   ├── ingester.py           # Ingests assets (rules, examples, schema) from files (YAML, TXT) into a KB store
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── base_retriever.py # Base class for retrieval strategies
│   │   ├── dense_retriever.py  # Vector-based retrieval
│   │   ├── sparse_retriever.py # BM25/TF-IDF based retrieval
│   │   ├── hybrid_retriever.py # Combination of dense and sparse
│   │   └── tag_retriever.py    # Tag-based retrieval
│   └── stores/               # Concrete KB store implementations
│       ├── __init__.py
│       ├── local_store.py      # Simple local file/in-memory store
│       ├── opensearch_store.py # OpenSearch client and store logic
│       └── vector_db_store.py  # E.g., FAISS, ChromaDB, Weaviate
├── llms/                     # LLM provider abstractions and implementations
│   ├── __init__.py
│   ├── base_llm_provider.py  # Base LLM provider interface (like your BaseLLM)
│   └── providers/
│       ├── __init__.py
│       ├── azure_openai.py     # Azure OpenAI client and wrapper
│       └── ...                 # Other providers like Anthropic, Google, HuggingFace
├── models/                   # Pydantic models for data structures
│   ├── __init__.py
│   ├── common.py             # Common enums (e.g., MessageRole, TaskStatus), basic types
│   ├── assets.py             # Models for business rules, examples, schema definitions, retrieved assets
│   ├── agent_primitives.py   # Models for agent-specific concepts (e.g., Task, Plan, Observation)
│   └── execution.py          # Models for code execution results, dataframe summaries, viz outputs
├── orchestration/            # Framework-specific orchestration logic
│   ├── __init__.py
│   ├── base_builder.py       # Abstract base class for building graphs/agents from config
│   ├── langgraph/            # LangGraph-specific implementation
│   │   ├── __init__.py
│   │   ├── builder.py          # Builds LangGraph StateGraph from the common config
│   │   ├── adapter.py          # Adapts dataqa.core.BaseComponent to LangGraph nodes
│   │   └── utils.py            # LangGraph specific utilities
│   ├── crewai/               # Placeholder for CrewAI integration
│   │   ├── __init__.py
│   │   └── builder.py
│   ├── autogen/              # Placeholder for Autogen integration
│   │   ├── __init__.py
│   │   └── builder.py
│   └── ...                   # Other orchestration frameworks
├── agents/                   # Definitions of pre-built or configurable agents
│   ├── __init__.py
│   ├── planner_agent.py      # Agent responsible for planning tasks
│   ├── data_query_agent.py   # Agent specialized in data querying
│   ├── data_analysis_agent.py # Agent specialized in data analysis
│   ├── data_visualization_agent.py # Agent specialized in data visualization
│   └── master_agent.py       # High-level orchestrator agent (like CWDAgent example)
├── workflows/                # Definitions of pre-built or configurable workflows
│   ├── __init__.py
│   ├── nl_to_sql_workflow.py # Example: A complete NL-to-SQL workflow
│   └── data_processing_workflow.py
├── utils/                    # General utility functions for the library
│   ├── __init__.py
│   ├── import_utils.py       # For dynamic class loading (cls_from_str)
│   ├── file_utils.py         # For loading data from files (YAML, JSON, TXT)
│   └── general_utils.py      # Other miscellaneous utilities
└── examples/                 # Example configurations, data, and scripts
    ├── assets/               # Example business rules, schemas, query-code pairs
    │   ├── schema.yaml
    │   ├── rules.txt
    │   └── examples.yaml
    ├── configs/              # Example YAML configurations for agents/workflows
    │   ├── simple_sql_query_workflow.yaml
    │   └── multi_agent_analysis.yaml
    └── run_example.py        # Script to run an example using the library
```

**Key Rationale for this Structure:**
*   **Separation of Concerns:** Clear separation between core logic, components, LLM/KB interactions, orchestration, and data models.
*   **Extensibility:** Adding new components, LLM providers, KB stores, or orchestration frameworks involves adding new modules in their respective directories without major refactoring. The `orchestration` directory is key for this.
*   **Config-Driven:** The `configs` directory is central. `components` are designed to be configurable.
*   **Abstraction:** Base classes in `core`, `llms`, `knowledge_base` define interfaces, promoting loose coupling.
*   **Reusability:** Components in `components/` are designed to be reused across different agents and workflows.

## 2. Common Config Structure (YAML)

This YAML structure aims to be comprehensive enough to define complex agents and workflows, while remaining readable and adaptable to different orchestration frameworks.

```yaml
# --- Metadata ---
version: "1.0"                 # Config schema version
name: "nlq_sales_analyzer"     # Unique name for this agent/workflow
description: "Analyzes sales data using natural language queries."

# --- Orchestration Framework ---
orchestration:
  framework: "langgraph"       # e.g., langgraph, crewai, autogen
  # Framework-specific settings:
  # settings:
  #   recursion_limit: 20 # For LangGraph

# --- State Schema ---
# Path to the Pydantic model defining the shared state of the graph/agent.
# This is crucial for frameworks like LangGraph.
state_schema: "dataqa.core.base_state.DynamicGraphState" # Or a custom state like CwdState
# For DynamicGraphState, its fields could be partly inferred from component outputs.

# --- Global Resources ---
# Define LLMs, KBs, DB connections etc., that can be referenced by components/agents.
resources:
  llms:
    - name: "default_llm_gpt4o"
      type: "dataqa.llms.providers.azure_openai.AzureOpenAI"
      config: # Actual LLM provider config
        model: "gpt-4o-2024-08-06"
        # api_key, base_url can be set via env vars or here
        # temperature, max_tokens, etc.
  knowledge_bases:
    - name: "sales_kb"
      type: "dataqa.knowledge_base.stores.local_store.LocalStore" # Example
      config:
        store_path: "./kb_data/sales_assets.json" # Path to ingested data
        # Asset types this KB contains (e.g., schema, business_rules, examples)
        asset_types: ["db_schema", "sales_rules", "query_examples"]
  # Databases (for SQL Executor, etc.)
  databases:
    - name: "sales_db_snowflake"
      type: "snowflake" # Internal type recognized by SQLExecutor
      connection_details: # Could be a connection string or structured details
        # account, user, password (use secrets management), warehouse, database, schema
        # ...

# --- Component Definitions ---
# These are the building blocks (nodes in a graph, tools for an agent).
components:
  - name: "query_rewriter_node"
    type: "dataqa.components.data_querying.QueryRewriter"
    config:
      llm: "resource:llms.default_llm_gpt4o" # Reference a defined LLM
      prompt_template: "Rewrite the query '{query}' using history: {history}"
    inputs: # Map from state to component's input fields
      query: "state:current_query"
      history: "state:conversation_history"
    # Outputs are implicitly stored in state, e.g., as state.query_rewriter_node_output.rewritten_query

  - name: "schema_retriever_node"
    type: "dataqa.components.data_querying.AssetRetriever"
    config:
      knowledge_base: "resource:knowledge_bases.sales_kb"
      asset_type_to_retrieve: "db_schema"
      retriever_config: # Config for the specific retriever (e.g., dense, tag)
        type: "dense_retriever" # Matches a retriever in dataqa.knowledge_base.retriever
        top_k: 3
    inputs:
      query: "state:query_rewriter_node_output.rewritten_query" # Accessing output of a previous node from state

  - name: "sql_code_generator_node"
    type: "dataqa.components.data_querying.CodeGenerator"
    config:
      llm: "resource:llms.default_llm_gpt4o"
      code_type: "sql"
      # Prompt template can be complex, loaded from file, or built using PromptBuilder component
      prompt_template_path: "file:./prompts/sql_generation_prompt.txt" # Using FILE_ prefix for loading
    inputs:
      query: "state:query_rewriter_node_output.rewritten_query"
      db_schema: "state:schema_retriever_node_output.retrieved_assets" # Assuming asset retriever outputs structured data
      # business_rules, examples could also be inputs here from other retrievers

  - name: "sql_executor_node"
    type: "dataqa.components.data_querying.code_executor.SQLExecutor"
    config:
      database_connection: "resource:databases.sales_db_snowflake" # Reference a defined DB
    inputs:
      code: "state:sql_code_generator_node_output.generated_code"

  - name: "result_analyzer_node"
    type: "dataqa.components.data_analysis.AnalysisGenerator"
    config:
      llm: "resource:llms.default_llm_gpt4o"
      analysis_prompt: "Summarize this data: {data}. What are the key insights related to {query}?"
    inputs:
      data: "state:sql_executor_node_output.result_dataframe_summary" # Assuming executor provides a summary
      query: "state:query_rewriter_node_output.rewritten_query"

  # Example of a router component for conditional logic
  - name: "execution_router_node"
    type: "dataqa.components.general.Router"
    config:
      # Rules for routing based on state values
      # Each rule defines a condition (e.g., Python expression on state) and a target node name
      routes:
        - condition: "state:sql_executor_node_output.error is not None"
          target: "error_handler_node" # Name of another component/agent
        - condition: "state:sql_executor_node_output.result_dataframe is not None"
          target: "result_analyzer_node"
      default_target: "__END__" # Or another node
    inputs: # Fields from state needed to evaluate conditions
      error: "state:sql_executor_node_output.error"
      result_dataframe: "state:sql_executor_node_output.result_dataframe"


# --- Graph/Workflow Definition ---
# Defines the structure if this is primarily a workflow.
# Can be adapted by the specific orchestrator's builder.
graph:
  entry_point: "query_rewriter_node"
  # Nodes can be implicitly all components defined above, or listed explicitly for clarity.
  # nodes: [ "query_rewriter_node", "schema_retriever_node", ... ]
  edges:
    - source: "query_rewriter_node"
      target: "schema_retriever_node"
    - source: "schema_retriever_node" # Multiple components can be triggered in parallel if supported
      target: "sql_code_generator_node"
    - source: "sql_code_generator_node"
      target: "sql_executor_node"
    - source: "sql_executor_node"
      target: "execution_router_node" # Leads to a conditional router

  # Conditional edges are handled by the router component in this design.
  # The router ("execution_router_node") itself will output the name of the next node.
  # The orchestration builder (e.g., LangGraph builder) will use this output for routing.
  # For LangGraph, the `builder.py` would wire up the Router component's output to conditional logic.

  # Terminal nodes or how the graph ends.
  # Can be specific node names or a special "__END__" keyword.
  # If a node is terminal, it might not have outgoing edges, or it could explicitly point to __END__.
  # result_analyzer_node could lead to __END__ or another node like a "response_formatter_node".
  # error_handler_node could also lead to __END__ or a retry mechanism.

# --- Agent Definitions ---
# Used if building a multi-agent system.
# An agent can itself be a graph (defined above) or have simpler logic.
# agents:
#   - name: "master_query_agent"
#     type: "dataqa.agents.MasterAgent" # Path to the agent class
#     llm: "resource:llms.default_llm_gpt4o"
#     # Agent's role, goals, constraints
#     config:
#       system_prompt: "You are a master AI assistant..."
#       # This agent might orchestrate a graph defined above,
#       # or delegate to sub-agents.
#       # For example, its 'run' logic could invoke the 'nlq_sales_analyzer' graph.
#
#   - name: "sql_tool_agent" # A more specialized agent, perhaps used as a tool by another.
#     type: "dataqa.agents.ToolUsingAgent"
#     llm: "resource:llms.default_llm_gpt4o"
#     tools: # List of components this agent can use
#       - "component:sql_code_generator_node" # Reference components by name
#       - "component:sql_executor_node"
#     config:
#       system_prompt: "You are an SQL expert. Given a query, generate and execute SQL."

# --- Output Configuration ---
# Defines what from the final state should be returned as the result.
# Could be handled by a final "gather" node or specified here.
# output:
#   final_response: "state:result_analyzer_node_output.analysis_summary"
#   generated_sql: "state:sql_code_generator_node_output.generated_code"
```

**Explanation of Config Structure:**
*   **`orchestration`**: Specifies the backend framework (LangGraph, CrewAI, etc.), allowing the library to adapt.
*   **`state_schema`**: Defines the Pydantic model for the graph's state, essential for type safety and state management.
*   **`resources`**: Centralizes definitions for LLMs, KBs, DBs. Components reference these via `resource:<type>.<name>`. This promotes DRY and easy configuration updates.
*   **`components`**: Each component has a `name`, `type` (Python class path for dynamic loading), `config` (its specific parameters), and `inputs`.
    *   **`inputs`**: Maps state fields to the component's input arguments. `state:path.to.field` syntax is used. Outputs of components are assumed to update the state (e.g., `component_name_output` becomes a field in the state).
    *   `file:./path/to/file.txt` syntax can be used in config values to load content from files.
*   **`graph`**: Defines the workflow structure with `entry_point`, `edges`. Conditional logic is handled by `Router` components, making the graph definition more declarative. The builder for the chosen orchestration framework will interpret this section.
*   **`agents`**: Allows defining agents, their types, LLMs, tools (referencing components), and specific configurations. This supports building multi-agent systems.
*   This structure is designed to be parsed by `dataqa.configs.loader.py`, which would instantiate components, resources, and then pass the structure to the appropriate builder in `dataqa.orchestration`.

This detailed folder structure and common config structure should provide a solid foundation for your `dataqa` library, enabling flexibility, extensibility, and a config-driven approach for both agents and workflows.
