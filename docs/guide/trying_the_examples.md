# Try the Examples

The DataQA library comes with pre-packaged examples to help you get started. This guide shows you how to run them quickly.

---

## Prerequisites

Before running examples, ensure you have:

1. **Installed DataQA** (see [Quickstart](../quickstart.md#1-installation))
2. **Set up environment variables** for LLM access:
   ```bash
   export AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
   export OPENAI_API_BASE="https://your-azure-openai-endpoint.net/"
   ```

---

## Running the CWD Agent Example

The library includes a CIB Merchant Payments example that demonstrates the CWD Agent in action.

### Quick Run

```bash
python -m dataqa.examples.cib_mp.agent.cwd_agent
```

This will:
- Initialize the CWD Agent using the configuration in `dataqa/examples/cib_mp/agent/`
- Run a pre-defined query
- Print the agent's response and execution trace to your console

### Expected Output

You should see:
- A final text response
- Any output DataFrames
- A debug trace showing the agent's plan and execution steps

### Customizing the Example

You can modify the example script to try your own queries:

1. Navigate to the example directory:
   ```bash
   # Find the example file location
   python -c "import dataqa.examples.cib_mp.agent.cwd_agent; import os; print(os.path.dirname(dataqa.examples.cib_mp.agent.cwd_agent.__file__))"
   ```

2. Edit the `cwd_agent.py` file to change the query or configuration

---

## Running the Benchmarking Suite

The library includes a benchmarking suite to evaluate agent performance.

### Quick Run

```bash
python -m dataqa.benchmark.run_test \
    --config /path/to/benchmark_config.yml \
    --test-data /path/to/test_questions.yml
```

### Creating Your Own Benchmark

1. **Create a test data file** (`test_questions.yml`):
   ```yaml
   use_case: "My Agent Evaluation"
   data:
     - id: "test_001"
       question: "What is the total revenue?"
       solution:
         - function_arguments:
             sql: "SELECT SUM(revenue) FROM sales_report;"
       active: true
   ```

2. **Run the benchmark** pointing to your agent config and test data

3. **Review results** in the output Excel file

For more details, see [Evaluate Your Agent](evaluating_your_agent.md).

---

## Troubleshooting

**Authentication errors:**
- Double-check your environment variables
- Verify the API key and endpoint are correct

**Missing data files:**
- Ensure all referenced data files exist in the expected locations

**Import errors:**
- Make sure DataQA is installed: `pip install aicoelin-dataqa`
- Check that you're using Python 3.11 or higher

---

## Next Steps

- **[Configure Your Agent](configuring_your_agent.md)**: Create your own agent configuration
- **[Knowledge Asset Tools](knowledge_asset_tools.md)**: Generate and enhance assets
- **[Understanding the CWD Agent](understanding_the_cwd_agent.md)**: Understand how the CWD Agent works

