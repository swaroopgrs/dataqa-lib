---
inclusion: always
---

# DataQA Project Structure & Organization

## Core Architecture
DataQA follows a layered architecture with clear separation of concerns:

```
src/dataqa/
├── models/        # Pydantic data models and schemas
├── primitives/    # Core composable components (LLM, Executor, Knowledge)
├── config/        # Configuration loading and validation
├── agent/         # Agent implementations and workflows
├── cli/           # Command-line interface
└── exceptions.py  # Custom exception classes
```

## File Organization Patterns
- **Models**: Define all data structures using Pydantic V2 with comprehensive validation
- **Primitives**: Implement stateless, composable components with dependency injection
- **Config**: YAML-based agent configurations parsed through Pydantic schemas
- **Tests**: Mirror source structure with comprehensive unit and integration tests

## Naming Conventions
- Use `snake_case` for Python files, functions, and variables
- Use `PascalCase` for classes and Pydantic models
- Prefix test files with `test_` and mirror source structure
- Use descriptive names that reflect business domain concepts

## Import Standards
- Use absolute imports from `dataqa` package root
- Group imports: standard library, third-party, local modules
- Import specific classes/functions rather than entire modules when possible
- Use type imports with `from typing import TYPE_CHECKING` for circular dependencies

## Configuration Management
- Store agent configs in `config/` directory as YAML files
- Use Pydantic models for all configuration validation
- Support environment-specific overrides through environment variables
- Never hardcode sensitive values - use environment variables or secure vaults