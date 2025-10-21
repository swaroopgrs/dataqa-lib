Okay, this is an exciting project! Let's break this down.

## Task 1: Folder Structure

Your proposed new folder structure is a good starting point and already shows a better organization than the initial "Pipeline related sample code" structure. The refactoring you've started with `dataqa/components/base.py` and its tests is a solid foundation.

Here's an enhanced version of your proposed folder structure, incorporating ideas to further improve clarity, modularity, and extensibility, especially for supporting different orchestration frameworks and distinguishing between Workflows and Agents:

```
dataqa_project_root/
├── dataqa/  # The main library source code
│   ├── __init__.py
│   ├── errors.py               # Common library-wide custom errors
│
│   ├── core_models/            # Pydantic models for fundamental data structures used across the library
│   │   ├── __init__.py
│   │   └── common.py           # E.g., ExecutionResult, RetrievedAsset, DataFrameReference
│
│   ├── llm_providers/          # Abstraction for different LLM providers
│   │   ├── __init__.py
│   │   ├── base.py             # BaseLLMProvider interface (e.g., defining ainvoke, stream)
│   │   └── azure_openai.py     # Implementation for Azure OpenAI
│   │   # ... other providers like anthropic.py, vertexai.py
│
│   ├── knowledge_base/         # For managing and accessing structured knowledge assets
│   │   ├── __init__.py
│   │   ├── assets.py           # Pydantic models for BusinessRule, SchemaMetadata, QueryExample, etc.
│   │   ├── ingestion/          # Logic for ingesting assets from various sources
│   │   │   ├── __init__.py
│   │   │   ├── base_ingestor.py
│   │   │   └── file_ingestor.py # For YAML/TXT files
│   │   ├── stores/             # Abstraction for different knowledge base storage backends
│   │   │   ├── __init__.py
│   │   │   ├── base_store.py
│   │   │   ├── opensearch_store.py
│   │   │   └── local_store.py  # Simple local file/in-memory store
│   │   └── retrieval_engine.py # Core logic for different retrieval strategies (dense, sparse, hybrid, tag-based)
│
│   ├── components/             # Reusable building blocks for Workflows and Agent Tools
│   │   ├── __init__.py
│   │   ├── base.py             # Your new Component base class (Generic[InputType, OutputType])
│   │   ├── schemas.py          # Common Pydantic schemas *used by* components if any, or typevar definitions.
│   │                           # Often, input/output schemas are specific to component types.
│   │   ├── code_execution/
│   │   │   ├── __init__.py
│   │   │   ├── base.py         # BaseCodeExecutorComponent
│   │   │   ├── schemas.py      # Input (e.g., CodeInput) & Output (e.g., CodeExecutionResult)
│   │   │   ├── in_memory.py    # InMemoryCodeExecutorComponent
│   │   │   └── remote_api.py   # ApiBasedCodeExecutorComponent
│   │   ├── llm_interaction/    # Components that directly interact with LLMs (using llm_providers)
│   │   │   ├── __init__.py
│   │   │   ├── base.py         # BaseLLMInteractionComponent
│   │   │   ├── schemas.py
│   │   │   ├── structured_generator.py # Generates structured output (e.g., SQL, JSON)
│   │   │   └── chat_responder.py       # For conversational interactions
│   │   ├── prompt_engineering/ # Components for building/formatting prompts
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── schemas.py
│   │   │   └── template_formatter.py
│   │   ├── knowledge_retrieval/ # Components that use the knowledge_base.retrieval_engine
│   │   │   ├── __init__.py
│   │   │   ├── base.py         # BaseKnowledgeRetrieverComponent
│   │   │   ├── schemas.py      # Input (e.g., RetrievalQuery), Output (e.g., RetrievedKnowledge)
│   │   │   └── asset_retriever.py # Retrieves assets using configured strategies
│   │   ├── data_analysis/      # Components for performing data analysis
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── schemas.py
│   │   │   └── pandas_analyzer.py # Example using pandas
│   │   ├── visualization/      # Components for generating visualization specs or code
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── schemas.py      # Output (e.g., VegaLiteSpec, MatplotlibCode)
│   │   │   └── chart_generator.py
│   │   └── output_processing/  # Components for formatting/aggregating final outputs
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── schemas.py
│   │       └── result_gatherer.py
│
│   ├── framework_adapters/     # Key for decoupling from specific orchestration libraries
│   │   ├── __init__.py
│   │   ├── base_adapter.py     # Defines interface for adapting components/graphs
│   │   ├── langgraph_adapter.py # Helpers to wrap Components as LangGraph nodes, build graphs
│   │   # ... other adapters: crewai_adapter.py, autogen_adapter.py
│
│   ├── workflows/              # For predefined execution paths (e.g., LangGraph graphs)
│   │   ├── __init__.py
│   │   ├── builder.py          # Constructs workflows (e.g., LangGraph CompiledGraph) from config
│   │                           # Uses framework_adapters
│   │   ├── config_schemas.py   # Pydantic models for workflow definition in YAML
│   │   ├── state_models.py     # Pydantic models for workflow states (e.g., CwdState)
│   │   └── errors.py
│
│   ├── agents/                 # For dynamic, LLM-directed execution
│   │   ├── __init__.py
│   │   ├── base_agent.py       # Base class for an agent
│   │   ├── tool_adapters.py    # Adapts Components into Tools usable by agent frameworks
│   │   ├── builder.py          # Constructs agent systems from config (uses framework_adapters)
│   │   ├── config_schemas.py   # Pydantic models for agent definition in YAML
│   │   ├── standard_agents/    # Pre-built agent types
│   │   │   ├── __init__.py
│   │   │   ├── planner_agent.py
│   │   │   ├── data_querying_agent.py
│   │   │   # ... data_analysis_agent.py, visualization_agent.py
│   │   └── orchestrator_agent.py # High-level agent managing sub-agents
│
│   └── utils/                  # General, low-level utilities
│       ├── __init__.py
│       ├── import_utils.py     # e.g., cls_from_str
│       ├── file_utils.py       # e.g., load_file
│       └── pydantic_utils.py   # e.g., dynamic model creation helpers if needed beyond component_utils
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   # Mirror the dataqa structure for unit tests
│   │   ├── components/
│   │   │   └── test_base.py
│   │   # ...
│   ├── integration/
│   │   # ...
│
├── examples/                   # Example configurations and run scripts
│   ├── common_prompts/         # Reusable prompt templates
│   │   └── query_rewriter.txt
│   ├── data_query_workflow/
│   │   ├── config.yaml
│   │   ├── data/
│   │   │   └── sample_db.csv
│   │   └── run.py
│   ├── multi_step_agent/
│   │   ├── config.yaml
│   │   └── run.py
│
├── pyproject.toml
├── README.md
└── MANIFEST.in # If including non-Python files in package
```

**Key Rationale for Changes/Additions:**

*   **`core_models/`**: For Pydantic models like `RetrievedAsset` that are fundamental and used across many modules.
*   **`llm_providers/`**: Your idea, very good for abstraction.
*   **`knowledge_base/`**: More focused. `assets.py` for what is stored, `ingestion/` for how it gets there, `stores/` for where it lives, and `retrieval_engine.py` for the core "how to get it" logic. The `components/knowledge_retrieval/` will *use* this engine.
*   **`components/` Subdirectories**:
    *   Renamed `llm` to `llm_interaction` to be clearer.
    *   Added `data_analysis` and `visualization` as they are top-level steps.
    *   Each component type has its own `schemas.py` for its specific `InputType` and `OutputType` Pydantic models. This keeps component definitions self-contained. `components/schemas.py` could be for truly generic typevars or abstract base schemas if any.
*   **`framework_adapters/`**: This is crucial for your goal of not being tightly coupled to LangGraph. This module will contain the "glue" code. For instance, `langgraph_adapter.py` would have functions like `component_to_langgraph_node(component_instance, input_mappings, output_key_name)`.
*   **`workflows/` vs `agents/`**: Clear separation as per your requirements and Anthropic's definitions. Each has its own `builder.py` (which would use `framework_adapters`) and `config_schemas.py`.
*   **`agents/tool_adapters.py`**: To explicitly convert your `Components` into a format that agent frameworks (LangGraph agent tools, CrewAI tools, etc.) can consume.
*   **`examples/`**: Better organized to showcase different use-cases and keep example data/configs separate from library code.

This structure aims for high cohesion within modules and low coupling between them, facilitating easier maintenance and extension.

## Task 2: Config YAML Refactoring

The provided `config.yaml` has several redundancies and areas for simplification. The goal is to make it more intuitive and DRY (Don't Repeat Yourself). Your new `ComponentConfig` base class is a good step.

**Analysis of Redundancies/Issues in Original YAML:**

1.  **`name` field**: `name` appears at the top level of a component definition (e.g., `name: query_rewriter`), inside `params: { name: query_rewriter }`, and sometimes inside `params: { config: { name: query_rewriter } }`. This is highly redundant. The top-level key should be the canonical name.
2.  **`params: { config: { ... } }` nesting**: This extra `config` key seems unnecessary. The `params` (or a better-named key like `config_params` or just directly under the component key) should directly hold the configuration for that component's specific config schema.
3.  **`type` vs `component_class_path`**: Mismatch in naming. Let's standardize on `class_path` as it's more descriptive.
4.  **LLM Configuration in Components**: Components like `query_rewriter` re-specify `model: gpt-4o-2024-05-13`. If an LLM provider instance (like `gpt_4o_model`) is already configured with a model and passed to the component, the component shouldn't need to know the model name again unless it's for a specific override mechanism (which adds complexity).
5.  **Input/Output Definitions**: The `input` and `output` lists defining variables are verbose. For many components, input/output schemas are fixed. For highly dynamic ones (like `BasePromptLLMChain`), this definition might be needed if the schema isn't known until config time.
6.  **Pipeline `edges` for input mapping**: The term `edges` inside a node definition (like in `query_rewriter`) is confusing as it's actually defining input sources/mappings, not graph edges. Graph edges should be defined separately.
7.  **Implicit State Object**: The syntax `START.query` or `query_rewriter.rewritten_query` implies a shared state object where outputs are stored. This is fine for LangGraph. The mapping needs to be clear.

**Proposed Refactored YAML Structure:**

This structure assumes:
*   The top-level key for a definition (e.g., `my_azure_llm` or `std_query_rewriter`) becomes its unique `name`.
*   `class_path` specifies the Python class.
*   Other keys are parameters for the component's *specific* config schema (which inherits from `dataqa.components.base.ComponentConfig`).
*   `ref:section.name` syntax for referencing other defined entities.
*   `file:path/to/file` syntax for loading content from files.

```yaml
version: "1.0"

# 1. LLM Provider Definitions
# These are instances of LLM provider classes (e.g., dataqa.llm_providers.azure_openai.AzureOpenAIProvider)
llm_providers:
  default_gpt4o_azure: # This is the 'name'
    class_path: dataqa.llm_providers.azure_openai.AzureOpenAIProvider
    description: "Default Azure OpenAI GPT-4o provider."
    # Parameters for AzureOpenAIProviderConfig (inherits ComponentConfig)
    model_deployment_name: "gpt-4o-2024-05-13"
    base_url: "https://YOUR_AZURE_ENDPOINT.openai.azure.com/"
    api_version: "2024-02-01"
    api_key_env: "AZURE_OPENAI_API_KEY" # Recommended: load from environment variable
    # api_key: "your_direct_api_key_if_not_using_env"
    temperature: 0.0
    max_tokens: 2000

# 2. Component Definitions / Blueprints
# These are configurations for your reusable components.
# The 'name' (key) and 'class_path' are used by the builder to instantiate them.
# Other keys are parameters for the component's specific <ComponentName>Config model.
components:
  # --- Query Rewriter Component ---
  main_query_rewriter:
    class_path: dataqa.components.llm_interaction.structured_generator.StructuredGeneratorComponent # Example path
    description: "Rewrites user query based on history, outputs structured rewrite and reasoning."
    llm_provider_ref: "ref:llm_providers.default_gpt4o_azure" # Reference an LLM provider
    prompt_template: "file:./prompts/query_rewriter_template.v1.txt"
    # Input variables expected by the prompt_template
    template_input_variables: ["query", "previous_rewritten_query", "current_datetime"]
    # Output structure expected from the LLM (Pydantic model name or schema def)
    # This component's 'output_schema' property would point to a Pydantic model like QueryRewriteOutput
    # defined in dataqa/components/llm_interaction/schemas.py
    # output_schema_name: "QueryRewriteOutput" # Or define fields if truly dynamic
    # output_parser_type: "pydantic_model" # Could be implicit if output_schema_name is given

  # --- Prompt Formatter for Code Generation ---
  code_generator_prompt_formatter:
    class_path: dataqa.components.prompt_engineering.template_formatter.TemplateFormatterComponent
    description: "Formats the prompt for the SQL code generator."
    template: "file:./prompts/code_generator_template.v1.txt"
    # template_input_variables are implicitly defined by the template itself if the component is smart enough
    # or can be explicitly listed for validation:
    template_input_variables: ["rewritten_query", "db_schema_summary", "few_shot_examples"]

  # --- SQL Code Generation Component ---
  sql_code_generator:
    class_path: dataqa.components.llm_interaction.structured_generator.StructuredGeneratorComponent
    description: "Generates SQL code and reasoning based on a formatted prompt."
    llm_provider_ref: "ref:llm_providers.default_gpt4o_azure"
    # This component expects 'messages' or a 'formatted_prompt' as input.
    # Its output_schema would be something like SQLCodeGenerationOutput (code, reasoning)
    # output_schema_name: "SQLCodeGenerationOutput"
    output_parser_type: "xml" # As per original example

  # --- In-Memory SQL Code Executor ---
  duckdb_code_executor:
    class_path: dataqa.components.code_execution.in_memory.InMemoryCodeExecutorComponent
    description: "Executes SQL queries on in-memory data using DuckDB."
    # Parameters for InMemoryCodeExecutorConfig
    data_sources:
      - table_name: "PROD_BD_TH_FLAT_V3"
        file_path: "file:./data/FAKE_PROD_BD_TH_FLAT_V3.csv"
        # load_options: { "format": "csv", "header": true } # More detailed options
      - table_name: "ETS_D_CUST_PORTFOLIO"
        file_path: "file:./data/FAKE_ETS_D_CUST_PORTFOLIO.csv"
    # This component's input_schema expects 'code_to_execute', output_schema provides 'execution_result'

  # --- Final Output Gatherer ---
  final_result_assembler:
    class_path: dataqa.components.output_processing.result_gatherer.ResultGathererComponent
    description: "Assembles the final output from various intermediate results."
    # This component's input_schema might define fields like:
    # rewritten_query: str, generated_code: str, execution_output: Any
    # Its output_schema would be the final PipelineOutput model.


# 3. Workflow Definitions
# Defines one or more workflows using the components above.
workflows:
  - name: "payments_nl_to_sql_workflow"
    description: "Converts natural language payment queries to SQL, executes, and returns results."
    # Defines the expected input structure for this workflow.
    # This corresponds to your `PipelineInput` model.
    workflow_inputs_schema:
      query: str
      context: list[str] # Conversation history
      previous_rewritten_query: str
      datetime: str # Defaulted by Pydantic model if not provided

    # Defines the graph structure for LangGraph (or other frameworks via adapters)
    graph:
      # Nodes in the execution graph.
      # Each key is a unique node_id within this workflow.
      nodes:
        node_rewrite_query: # Unique ID for this node in the graph
          component_ref: "ref:components.main_query_rewriter" # References a component definition
          # Maps fields from the workflow state to the input schema of the component.
          # 'workflow_inputs' is a conventional key for the initial workflow inputs.
          # Otherwise, it's '<source_node_id>.outputs.<field_name>'
          input_mapping:
            query: "workflow_inputs.query"
            previous_rewritten_query: "workflow_inputs.previous_rewritten_query"
            current_datetime: "workflow_inputs.datetime"
          # Output from this node will be available in state as:
          # state.node_rewrite_query.outputs.rewritten_query
          # state.node_rewrite_query.outputs.reasoning (assuming these are fields in QueryRewriteOutput)

        node_format_sql_prompt:
          component_ref: "ref:components.code_generator_prompt_formatter"
          input_mapping:
            rewritten_query: "node_rewrite_query.outputs.rewritten_query"
            # db_schema_summary: "node_retrieve_schema.outputs.summary" # If schema retrieval was a step
            # few_shot_examples: "node_retrieve_examples.outputs.examples" # If example retrieval was a step
          # Output: state.node_format_sql_prompt.outputs.formatted_prompt (or .messages)

        node_generate_sql:
          component_ref: "ref:components.sql_code_generator"
          input_mapping:
            # Assuming sql_code_generator expects 'messages' list or 'formatted_prompt' string
            formatted_prompt: "node_format_sql_prompt.outputs.formatted_prompt"
          # Output: state.node_generate_sql.outputs.code, .reasoning

        node_execute_sql:
          component_ref: "ref:components.duckdb_code_executor"
          input_mapping:
            code_to_execute: "node_generate_sql.outputs.code"
          # Output: state.node_execute_sql.outputs.execution_result

        node_assemble_output:
          component_ref: "ref:components.final_result_assembler"
          input_mapping:
            rewritten_query: "node_rewrite_query.outputs.rewritten_query"
            code: "node_generate_sql.outputs.code"
            execution_output: "node_execute_sql.outputs.execution_result" # This will be the CodeExecutorOutput model
          # Output: state.node_assemble_output.outputs (which should match PipelineOutput)

      # Defines the execution flow (edges between nodes)
      edges:
        - source: "__start__" # Special LangGraph START node
          target: "node_rewrite_query"
        - source: "node_rewrite_query"
          target: "node_format_sql_prompt"
        - source: "node_format_sql_prompt"
          target: "node_generate_sql"
        - source: "node_generate_sql"
          target: "node_execute_sql"
        - source: "node_execute_sql"
          target: "node_assemble_output"
        - source: "node_assemble_output"
          target: "__end__" # Special LangGraph END node

      # Example for conditional edges:
      # conditional_edges:
      #   - source_node: "node_generate_sql"
      #     # Logic to determine the next path, based on state.node_generate_sql.outputs
      #     # Could be a path to a Python function or a simple field check
      #     condition_evaluator: "dataqa.common.conditions.check_sql_validity"
      #     # condition_on_field: "outputs.validity_status"
      #     target_map:
      #       "VALID": "node_execute_sql"
      #       "INVALID_SYNTAX": "node_fix_sql_syntax" # Another node
      #       "NEEDS_REVIEW": "__end__" # End if needs manual review

    # Defines the final output structure of the workflow, mapping from the state.
    # This corresponds to your `PipelineOutput` model.
    workflow_outputs_schema:
      rewritten_query: "node_assemble_output.outputs.rewritten_query"
      code: "node_assemble_output.outputs.code"
      execution_output: "node_assemble_output.outputs.execution_output"
      text: "node_assemble_output.outputs.text" # If the gatherer also produces a summary text
```

**Explanation of Changes & Benefits:**

1.  **Clear Sections**: `llm_providers`, `components`, `workflows` (and later `agents`) provide logical grouping.
2.  **Canonical Naming**: The YAML key (e.g., `default_gpt4o_azure`) is the unique `name`. This name is passed to the `ComponentConfig` instance by the builder.
3.  **`class_path`**: Standardized for specifying the Python class.
4.  **`ref:section.name`**: Clear and explicit way to reference other definitions (e.g., an LLM provider used by a component, or a component used by a workflow node).
5.  **`file:path/to/file`**: Standardized for loading external file content (like prompts).
6.  **Flattened Config**: Component-specific parameters are now directly under the component's definition, not nested under `params` or `params.config`.
7.  **Workflow Structure**:
    *   `workflow_inputs_schema`: Explicitly defines the data contract for starting the workflow.
    *   `graph.nodes`: Each node clearly defines:
        *   `component_ref`: Which component blueprint to use.
        *   `input_mapping`: How the component's inputs are sourced from the workflow state (either initial inputs or outputs of other nodes). The dot notation `source_node_id.outputs.field_name` or `workflow_inputs.field_name` standardizes state access. The `framework_adapter` (e.g., for LangGraph) will translate this into state access logic for the `__call__` method of the component.
    *   `graph.edges`: Defines explicit control flow.
    *   `workflow_outputs_schema`: Explicitly defines the data contract for the workflow's final result.
8.  **Extensibility**: This structure makes it easier to add new types of components or providers. The builder logic will parse these sections and instantiate objects accordingly.
9.  **Component Reusability**: `components` are defined once and can be referenced multiple times in different workflows or by different nodes (if their configuration is identical or slightly overridden at the node level - though full override at node level adds complexity, simple refs are cleaner).

This refactored YAML structure, combined with your `Component` base class and the proposed folder structure, should provide a robust and maintainable foundation for `dataqa`. The key will be the implementation of the "builder" (e.g., `dataqa.workflows.builder.py`) that can parse this YAML and construct the executable graph using the appropriate `framework_adapter`. Your `Component.__call__` method, with its input/output mapping logic injected by the builder, will be central to this.

This refined structure should also make it easier to understand the system's configuration and to evolve it as new requirements emerge.
