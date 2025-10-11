# DataQA Benchmarking

This directory contains tools and configurations for benchmarking DataQA agents and pipelines.

## Overview

The benchmarking framework allows you to:
- Run tests against your DataQA agents and pipelines
- Evaluate performance using both automated metrics and LLM-based evaluation
- Compare different configurations and approaches

## Directory Structure

- `config/`: Configuration files for benchmark tests
- `log/`: Log files from benchmark runs
- `output/`: Output files and results from benchmark runs

## Usage

To run a benchmark test:

```python
from dataqa.benchmark.run_test import TestPipeline
from dataqa.benchmark.schema import BenchmarkConfig
import yaml

# Load your benchmark configuration
with open("path/to/benchmark_config.yml", "r") as f:
    config_data = yaml.safe_load(f)
test_config = BenchmarkConfig(**config_data)

# Create and run the test pipeline
test_pipeline = TestPipeline(config=test_config)
await test_pipeline.run()
```

## Creating Your Own Benchmark Tests

1.  Create a benchmark configuration file:
    ```yaml
    use_case_config:
      - name: "my_benchmark_test"
        cwd_config: "path/to/agent_config.yml"
        test_data_file: "path/to/test_data.yml"
    output: "path/to/output/directory"
    log: "path/to/log/directory"
    run_predictions: true
    run_llm_eval: true
    ```
2.  Create test data files following the schema defined in `schema.py`.
3.  Run your benchmark tests and analyze the results.

## Example

See the provided configuration files in the `config/` directory for examples of how to set up benchmark tests.