---
inclusion: always
---

# DataQA Technical Standards

## Technology Stack
- **Python 3.10+** with modern features (async/await, type hints, pattern matching)
- **LangGraph** for state management and graph execution
- **Pydantic V2** for all data models and validation
- **Pandas** for tabular data representation
- **DuckDB** as default SQL engine
- **FAISS** for vector storage
- **uv** for dependency management
- **Pytest** for testing
- **Ruff** for formatting/linting

## Code Standards

### Type Safety & Validation
- Use comprehensive type hints on all functions and classes
- Implement Pydantic models for data validation and serialization
- Define clear Input/OutputContracts for components
- Validate all inputs through Pydantic schemas

### Architecture Patterns
- Build around LangGraph's state management paradigm
- Implement primitives as composable, stateless components
- Use dependency injection for external services (LLM, Executor, Knowledge)
- Strictly separate code generation from execution environments
- Never execute LLM-generated code directly

### Data & Configuration
- Represent all data as Pandas DataFrames
- Use DuckDB for local SQL operations
- Generate visualizations to in-memory buffers only
- Define agent configurations in YAML files parsed through Pydantic
- Support environment-specific configuration overrides

### Testing Requirements
- Write unit tests for all primitives with mocked external dependencies
- Test configuration parsing and validation
- Include integration tests for complete workflows
- Maintain high test coverage

## Development Guidelines

### Dependencies & Security
- Minimize external dependencies and pin versions
- Use optional dependencies for specialized features
- Use sandboxed execution environments
- Implement proper authentication for production
- Use custom exception classes from `dataqa.exceptions`

### Performance & Error Handling
- Leverage async/await for I/O operations
- Implement caching for expensive operations
- Provide clear, actionable error messages
- Log errors with appropriate context
- Implement graceful degradation