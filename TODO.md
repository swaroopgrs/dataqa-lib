# 💎DataQA Library - TODOs & Improvements==

---

==**## ++1. Feature Additions & Fixes++**==

- [ ] ==**Streaming & Node Statuses**==
- [x] ~~Implement streaming of node start/completion status.~~
- [x] ~~**Streaming Final Output**~~
- [ ] ~~Stream the final output response.~~
- [ ] ~~**Streaming Non-JSON Output**~~
- [ ] ~~Ensure final output responses are consistently formatted in Markdown.~~
- [ ] ==**History & Memory**==
- [ ] ~~Add proper history and memory for follow-up queries to both DBC and Local clients.~~
- [ ] ==**Tag Retrievers**==
- [ ] ~~Fix and test the tag retriever module.~~
- [ ] ==**Vector Retrievers**==
- [ ] ~~Fix and test the vector retriever module.~~
- [ ] ==**Pipeline Core**==
- [ ] ~~Fix any refactor `pipelines/pipeline.py` and `pipelines/schema.py`.~~
- [ ] ~~Add recursion limit to create react agent~~
- [x] ~~**Improve AgentResponseParser Metadata Handling**~~
- [ ] ~~Fix `AgentResponseParser` should gracefully handle empty or missing metadata (especially in DBC client).~~
- [ ] ==**Asset Versioning**==
- [ ] ~~Add version fields and changelogs to all asset files (rules, schema, examples)~~
- [ ] ==**Asset Validation**==
- [ ] ~~Add static validation module.~~
- [ ] ~~Add LLM based reviewer/validator module; post SQL generation.~~
- [ ] ==**Benchmarking**==
- [ ] ~~Remove benchmarking scripts from the library and add them to the examples folder~~

---

==**## ++2. Documentation & Developer Experience++**==

- [ ] ==**Auto-Generated API Docs**==
- [ ] ~~Integrate [MkDocs](https://www.mkdocs.org/) for API documentation.~~
- [ ] ~~Ensure all public classes/functions have docstrings.~~
- [ ] ==**High-Level Architecture Docs**==
- [ ] ~~Add a Mermaid or PlantUML diagram to the docs and README.~~
- [ ] ==**Getting Started (User)**==
- [ ] ~~Write a step-by-step guide for users installing and running the library.~~
- [ ] ==**Getting Started Guide (Contributors)**==
- [ ] ~~Add a `CONTRIBUTING.md` with setup, style, and PR process.~~
- [ ] ==**Update Docs for Asset Format Changes**==
- [ ] ~~Revise documentation to reflect new asset formats and config structures.~~
- [ ] ==**Inline Comments**==
- [ ] ~~Add/expand inline comments in complex modules (e.g., agent workflow, retrievers).~~
- [ ] ==**Cookbook/Examples**==
- [ ] ~~Add cookbook + example usage, and FAQ sections.~~
- [ ] ==**Documentation Hosting**==
- [ ] ~~Deploy documentation using Jules on Static Content Hosting (SCH).~~

---

==**## ++3. Testing++**==

- [ ] ==**Unit Tests**==
- [ ] ~~Add a `tests/` directory with pytest-based unit tests for all major modules.~~
- [ ] ==**Integration Tests**==
- [ ] ~~Write end-to-end tests for agent workflows (local and DBC).~~
- [ ] ==**Test Data & Fixtures**==
- [ ] ~~Centralize mocks and test data for reuse.~~
- [ ] ==**CI/CD Integration**==
- [ ] ~~Add jules changes for automated test runs on PRs.~~

---

==**## ++4. Logging, Error Handling & Observability++**==

- [ ] ==**Standardize Error Handling**==
- [ ] ~~Use custom exception classes and propagate meaningful errors.~~
- [ ] ==**Centralized Logging**==
- [ ] ~~Implement a unified logging config (log levels, structured logs).~~
- [ ] ==**Monitoring & Metrics**==
- [ ] ~~Integrate with OpenTelemetry or Prometheus for metrics/tracing. This can be done via integration with LangFuse or Arize Phoenix.~~

---

==**## ++5. Configuration & Environment Management++**==

- [ ] ==**Consolidate Configuration Schemas**==
- [ ] ==**Reduce redundancy in YAML/config files.**==
- [ ] ==**Environment Variable Consistency**==
- [ ] ~~Ensure all configs support env var expansion.~~

---

==**## ++6. Extensibility & Customization++**==

- [ ] ==**Plugin System for Tools**==
- [ ] ~~Design a plugin registry for analytics and plot tools.~~
- [ ] ==**Document Adapter Interfaces**==
- [ ] ~~Provide templates and documentation for adding new LLMs or SQL executors.~~

---

==**## ++7. UI & Visualization++**==

- [ ] ==**Demo UI**==
- [ ] ~~Build a simple Streamlit UI for demo, debugging and benchmarking.~~
- [ ] ==**Interactive Plots**==
- [ ] ~~Add support for interactive visualizations (e.g., Plotly).~~

---

==**## ==Legend==**==
- [ ] ==**= To Do**==
- [x] ==**= Done**==
- [ ] ~~==**=Strikethrough== = Obsolete**==