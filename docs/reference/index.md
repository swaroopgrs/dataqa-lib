# API Reference

Complete reference documentation for DataQA's Python API, configuration files, and key classes.

---

## Clients: Main Entry Points

### LocalClient

The primary client for local development and usage. Operates on a local project configuration file and its associated assets.

**Location:** `dataqa.integrations.local.client.LocalClient`

**Initialization:**
```python
from dataqa.integrations.local.client import LocalClient

client = LocalClient(config_path="path/to/agent.yaml")
```

**Key Methods:**

#### `process_query(request, streaming=False, summarize=False)`

Processes a user query and returns a response generator.

**Parameters:**
- `request` (CoreRequest): The query request
- `streaming` (bool): If True, yields intermediate status updates
- `summarize` (bool): If True, generates a summary of the execution

**Returns:** Generator yielding `CoreStatus` (if streaming) and `CoreResponse` (final)

**Example:**
```python
from dataqa.core.client import CoreRequest

request = CoreRequest(
    user_query="What is the total revenue?",
    conversation_id="conv-1",
    question_id="q1"
)

# Non-streaming
response_generator = client.process_query(request)
async for response in response_generator:
    if hasattr(response, 'text'):  # Final response
        print(response.text)
        print(response.output_dataframes)

# Streaming
response_generator = client.process_query(request, streaming=True)
async for chunk in response_generator:
    if hasattr(chunk, 'message'):  # Status update
        print(f"{chunk.name}: {chunk.message}")
    else:  # Final response
        print(chunk.text)
```

---

### DBCClient

Client for LLMSuite Database Connect (DBC) service integration. Used when deploying agents in a DBC environment.

**Location:** `dataqa.integrations.dbc.client.DBCClient`

**Note:** DBCClient requires specific DBC infrastructure and is typically used in production deployments. See your DBC administrator for setup details.

---

## Core Data Models

### CoreRequest

Represents a query request to the agent.

**Location:** `dataqa.core.client.CoreRequest`

**Fields:**
- `user_query` (str, required): The natural language query
- `question_id` (str, required): Unique identifier for this question
- `conversation_id` (str, required): Unique identifier for the conversation session
- `history` (List[CoreConversationTurn], optional): Previous conversation turns for context

**Example:**
```python
from dataqa.core.client import CoreRequest, CoreConversationTurn

request = CoreRequest(
    user_query="What was the revenue last month?",
    question_id="q1",
    conversation_id="conv-1",
    history=[
        CoreConversationTurn(
            query="What is total revenue?",
            output_text="The total revenue is $50,000."
        )
    ]
)
```

---

### CoreResponse

Represents the agent's response to a query.

**Location:** `dataqa.core.client.CoreResponse`

**Fields:**
- `text` (str): The main text response
- `output_dataframes` (List[pd.DataFrame]): Generated DataFrames
- `output_images` (List[bytes]): Generated images (as bytes)
- `steps` (List[CoreStep]): Intermediate processing steps for debugging

**Example:**
```python
async for response in client.process_query(request):
    if hasattr(response, 'text'):
        print(f"Answer: {response.text}")
        for i, df in enumerate(response.output_dataframes):
            print(f"DataFrame {i+1}:")
            print(df.head())
        for step in response.steps:
            print(f"{step.name}: {step.content}")
```

---

### CoreConversationTurn

Represents a single turn in a conversation history.

**Location:** `dataqa.core.client.CoreConversationTurn`

**Fields:**
- `query` (str): The user's query
- `output_text` (str): The agent's response

---

### CoreStep

Represents an intermediate processing step for debugging.

**Location:** `dataqa.core.client.CoreStep`

**Fields:**
- `name` (str): Name of the processing step
- `content` (str): Details of the step

---

## Knowledge Asset Tools

### DataScanner

Automatically extracts and enriches schema from data files.

**Location:** `dataqa.core.components.knowledge_extraction.data_scanner.DataScanner`

**Initialization:**
```python
from dataqa.core.components.knowledge_extraction.data_scanner import DataScanner

config = {
    "data_files": [
        {"path": "data/sales.csv", "table_name": "sales_report"}
    ],
    "output_path": "data/"
}

scanner = DataScanner(config)
```

**Key Methods:**

#### `extract_schema()`

Extracts schema structure from data files (no LLM required).

**Returns:** `DatabaseSchema` object

**Example:**
```python
schema = await scanner.extract_schema()
# Output: data/schema_extracted.yml
```

#### `infer_metadata()`

Infers descriptions and metadata using LLM (requires LLM credentials).

**Returns:** None (saves to `schema_inferred.yml`)

**Example:**
```python
await scanner.infer_metadata()
# Output: data/schema_inferred.yml with LLM-generated descriptions
```

---

### RuleInference

Generates business rules by comparing generated SQL with expected SQL.

**Location:** `dataqa.core.components.knowledge_extraction.rule_inference.RuleInference`

**Initialization:**
```python
from dataqa.core.components.knowledge_extraction.rule_inference import RuleInference
from dataqa.core.llm.openai import AzureOpenAI

llm = AzureOpenAI(...)
rule_inference = RuleInference(llm=llm, prompt=prompt)
```

**Key Methods:**

#### `__call__(query, generated_sql, expected_sql, config)`

Compares SQL and infers missing rules.

**Parameters:**
- `query` (str): User question
- `generated_sql` (str): SQL generated by agent
- `expected_sql` (str): Expected/correct SQL
- `config` (RunnableConfig): Runtime configuration

**Returns:** Dict with `rules` and `llm_output`

**Example:**
```python
result = await rule_inference(
    query="What is total revenue?",
    generated_sql="SELECT SUM(revenue) FROM sales;",
    expected_sql="SELECT SUM(revenue) FROM sales WHERE status = 'COMPLETED';",
    config=config
)
# Returns suggested rules explaining the difference
```

---

## Agent Classes

### CWDAgent

The core conversational agent implementing the Plan-Work-Dispatch loop.

**Location:** `dataqa.core.agent.cwd_agent.cwd_agent.CWDAgent`

**Note:** In practice, you typically use `LocalClient` or `DBCClient` (LLMSuite Database Connect) rather than instantiating `CWDAgent` directly. The clients handle agent creation and configuration.

---

### CWDState

Tracks the agent's progress through query execution.

**Location:** `dataqa.core.agent.cwd_agent.cwd_agent.CWDState`

**Key Fields:**
- `query`: The user's original question
- `plan`: The current plan with list of tasks
- `worker_response`: Results from executed tasks
- `output_dataframes`: DataFrames generated during execution
- `final_answer`: The agent's final response text

---

## Configuration Reference

### [Agent Configuration](agent_config.md)

Complete reference for the `agent.yaml` configuration file, including:
- LLM configuration and assignment
- Resource manager setup
- Retriever configuration
- Worker configuration (SQL execution, analytics, plotting)
- Use case context
- SQL dialect settings

---

## See Also

- [Quickstart](../quickstart.md): Get started quickly
- [Configure Your Agent](../guide/configuring_your_agent.md): Learn about agent configuration
- [Create Knowledge Assets Manually](../guide/creating_knowledge_assets.md): Includes asset file format reference
- [Knowledge Asset Tools](../guide/knowledge_asset_tools.md): Learn about DataScanner and Rule Inference
