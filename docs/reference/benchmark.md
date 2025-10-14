# Benchmarking API Reference

DataQA includes a benchmarking suite for evaluating the accuracy, latency, and performance of agents and pipelines.

---

## Test Pipeline

The `TestPipeline` is the main orchestrator for running a benchmark.

::: dataqa.benchmark.test_pipeline.TestPipeline

---

## Configuration Schema

These Pydantic models define the structure of the benchmark configuration files.

### BenchmarkConfig

The root configuration object for a benchmark run.

::: dataqa.benchmark.schema.BenchmarkConfig

### TestDataItem

Represents a single test case (question, expected answer, etc.).

::: dataqa.benchmark.schema.TestDataItem

---

## Evaluation

These models are used for storing and representing evaluation results.

### EvaluationLabel

Labels for evaluation outcomes (e.g., correct, wrong, reject).

::: dataqa.benchmark.schema.EvaluationLabel

### LLMJudgeOutput

Structured output for LLM-based evaluation.

::: dataqa.benchmark.schema.LLMJudgeOutput

---

## See Also

- [Benchmarking & Evaluation Guide](../guide/benchmarking.md)
- [API Reference: Agent](agent.md)
