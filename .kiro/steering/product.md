
---
inclusion: always
---

# DataQA Product Guidelines

## Core Principles

**Security-First Architecture** - Never execute LLM-generated code directly. Always use sandboxed execution environments with proper validation and approval gates.

**Composable Components** - Build modular primitives that can be combined into specialized agents. Avoid monolithic implementations.

**Configuration-Driven** - Define agent behavior through YAML configs validated by Pydantic models. Support environment-specific overrides.

## Agent Architecture Patterns

When implementing agents, follow these established patterns:

- **Workflow Agents** - Linear task execution with clear state transitions
- **ReAct Agents** - Iterative reasoning and action loops with LangGraph state management
- **Plan-and-Execute** - High-level planning followed by detailed execution steps
- **Hierarchical Delegation** - Manager agents coordinating specialist sub-agents

## Data Interaction Requirements

- Convert natural language to SQL/API calls with business context
- Chain operations for multi-step analysis and transformations
- Generate visualizations to in-memory buffers only
- Implement human-in-the-loop approval for sensitive operations
- Maintain conversational context across multi-turn interactions

## Security & Execution Model

**Critical**: Strictly separate code generation from execution:
- LLM generates code → Validation layer → Sandboxed execution backend
- Support pluggable backends: in-memory (dev) and API-based (prod)
- Log all agent decisions and data access for audit trails
- Require explicit user consent for data modifications

## Implementation Guidelines

- Use LangGraph for state management and graph execution
- Represent all data as Pandas DataFrames
- Implement dependency injection for external services (LLM, Executor, Knowledge)
- Ask users for clarification rather than making assumptions
- Provide progressive disclosure: start simple, reveal complexity as needed

## Target Use Cases

- **Data Platform Teams** - Secure data interaction services
- **Application Developers** - Embedded NL data features
- **Analytics Teams** - Self-service analytics tools
