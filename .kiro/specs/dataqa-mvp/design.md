# Design Document

## Overview

The DataQA MVP implements a modular, composable framework for building data agents that can interact with structured data through natural language. The architecture follows a layered approach with clear separation between configuration, orchestration, execution, and knowledge management components. The system is built on LangGraph for state management and orchestration, with Pydantic for configuration validation and type safety.

## Architecture

The system follows a plugin-based architecture with the following core layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Python API    │  │   CLI Tools     │  │   Config    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Agent Orchestration Layer                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   LangGraph     │  │  State Manager  │  │   HITL      │ │
│  │   Workflows     │  │                 │  │  Interface  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Core Components Layer                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Knowledge     │  │    Executor     │  │    LLM      │ │
│  │   Primitive     │  │   Primitive     │  │  Interface  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │    DuckDB       │  │     FAISS       │  │  Matplotlib │ │
│  │   (SQL Engine)  │  │ (Vector Store)  │  │ (Plotting)  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### Configuration System

**ConfigManager**
- Loads and validates YAML configuration files using Pydantic models
- Manages environment-specific settings and secrets
- Provides type-safe access to configuration parameters

**AgentConfig (Pydantic Model)**
```python
class AgentConfig(BaseModel):
    name: str
    llm: LLMConfig
    knowledge: KnowledgeConfig
    executor: ExecutorConfig
    workflow: WorkflowConfig
```

### Core Primitives

**KnowledgePrimitive**
- Abstract base class for knowledge retrieval systems
- Implementations: FAISSKnowledge (local), OpenSearchKnowledge (production)
- Methods: `ingest()`, `search()`, `update()`

**ExecutorPrimitive** 
- Abstract base class for code execution environments
- Implementations: InMemoryExecutor (DuckDB), APIExecutor (remote)
- Methods: `execute_sql()`, `execute_python()`, `get_schema()`

**LLMInterface**
- Abstraction layer for different LLM providers
- Handles prompt formatting, context injection, and response parsing
- Supports OpenAI, Anthropic, and local models

### Agent Orchestration

**DataAgent**
- Main orchestration class built on LangGraph
- Manages conversation state and workflow execution
- Coordinates between knowledge retrieval, code generation, and execution

**SharedState (Pydantic Model)**
```python
class SharedState(BaseModel):
    conversation_history: List[Message]
    current_query: str
    retrieved_context: List[Document]
    generated_code: Optional[str]
    execution_results: Optional[Dict]
    pending_approval: Optional[str]
```

**WorkflowNodes**
- `query_processor`: Analyzes user input and determines intent
- `context_retriever`: Searches knowledge base for relevant information
- `code_generator`: Creates SQL/Python code using LLM with context
- `approval_gate`: Handles human-in-the-loop approval for sensitive operations
- `executor`: Runs generated code in secure environment
- `response_formatter`: Formats results for user presentation

## Data Models

### Core Data Structures

**Message**
```python
class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]]
```

**Document**
```python
class Document(BaseModel):
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]]
    source: str
```

**ExecutionResult**
```python
class ExecutionResult(BaseModel):
    success: bool
    data: Optional[pd.DataFrame]
    error: Optional[str]
    execution_time: float
    code_executed: str
```

### Database Schema Management

**SchemaInfo**
```python
class SchemaInfo(BaseModel):
    tables: List[TableInfo]
    relationships: List[Relationship]
    business_rules: List[str]
```

**TableInfo**
```python
class TableInfo(BaseModel):
    name: str
    columns: List[ColumnInfo]
    description: Optional[str]
    sample_data: Optional[List[Dict]]
```

## Error Handling

### Error Categories

1. **Configuration Errors**: Invalid YAML, missing required fields
2. **Knowledge Errors**: Failed document ingestion, search failures
3. **Generation Errors**: LLM API failures, invalid code generation
4. **Execution Errors**: SQL errors, Python runtime errors, connection failures
5. **Security Errors**: Unauthorized access attempts, unsafe code detection

### Error Handling Strategy

**Graceful Degradation**
- System continues operating with reduced functionality when non-critical components fail
- Fallback mechanisms for knowledge retrieval and code generation

**User-Friendly Error Messages**
- Technical errors are translated to actionable user messages
- Sensitive system information is never exposed to end users

**Logging and Monitoring**
- Comprehensive logging at all system levels
- Error tracking and performance metrics collection

## Testing Strategy

### Unit Testing
- Individual component testing with mocked dependencies
- Pydantic model validation testing
- LLM interface mocking for consistent test results

### Integration Testing
- End-to-end workflow testing with real databases
- Knowledge base ingestion and retrieval testing
- Multi-component interaction validation

### Security Testing
- Code injection prevention testing
- Access control validation
- Sandbox escape attempt detection

### Performance Testing
- Query response time benchmarking
- Knowledge base search performance
- Memory usage and resource consumption monitoring

### Test Data Management
- Synthetic datasets for testing different data scenarios
- Schema variations for robustness testing
- Edge case data for error handling validation

## Implementation Phases

### Phase 1: Core Infrastructure
- Configuration system and Pydantic models
- Basic LangGraph workflow setup
- In-memory executor with DuckDB
- Simple knowledge primitive with FAISS

### Phase 2: Agent Capabilities  
- Natural language query processing
- Context-aware code generation
- Basic visualization support
- CLI tool implementation

### Phase 3: Production Features
- Human-in-the-loop approval system
- Enhanced error handling and logging
- Performance optimization
- Documentation and examples