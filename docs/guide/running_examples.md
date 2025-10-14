# Running the Examples

The DataQA library comes with pre-packaged examples to help you get started quickly.
These examples demonstrate how to run both the `CWD`Agent` (conversational agent) and a data processing pipeline.

---

## Prerequisites

Before running the examples, make sure you have:
- Installed DataQA (see [Installation](installation.md))
- Set up your environment variables for LLM access (see below)

### Set Environment Variables

You must set your Azure OpenAI or other LLM credentials before running any example:

```bash
export AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
export OPENAI_API_BASE="https://your-azure-openai-endpoint.net/"
# (Add other variables as needed for your environment)
```

Or, place them in a `.env` file in your project root.

---

## Running the "CWD`Agent`" Example

This example demonstrates how to ask the conversational agent a question about merchant payment data.

**1. Open your terminal.**
**2. Run the following command:**
```bash
python -m dataqa.examples.cib_mp.agent.cwd_agent
```
This will:
- Initialize the `CWD`Agent` using the configuration found in `dataqa/examples/cib_mp/agent/`
- Run a pre-defined query
- Print the agent's response and a detailed execution trace to your console

**Expected Output:**
- You should see a final text response, any output dataframes, and a debug trace.
- If you see authentication errors, check your environment variables.

**Tip:**
- You can open and modify the script at `dataqa/examples/cib_mp/agent/cwd_agent.py` to try your own queries.

---

## Running the Benchmarking Suite

The library includes a powerful benchmarking suite to evaluate the performance of your agents.

**1. Navigate to the benchmark config directory** inside the installed package to see examples (for reference).
**2. Create your own benchmark configuration YAML file.** You can copy and modify an existing one, like `dataqa/benchmark/config/agent_2025_0515.yml`. Make sure the test data paths are correct.
**3. Run the benchmark test script** using the `-c` flag and pass your config file with `-c`:

```bash
python -m dataqa.benchmark.run_test -c /path/to/your/custom_benchmark_config.yml
```
This will execute the benchmark tests and save the results to the output directory specified in your configuration file.

---

## Next Steps

- [Building Your First Agent](building_your_first_agent.md)
- [Customizing Agents](customizing_agents.md)
- [API Reference](../reference/agent.md)

---

## Troubleshooting

- **Authentication error:**
  - Double-check your environment variables and credentials.
- **Missing data files:**
  - Ensure all referenced data files exist in the correct locations.
- **Other issues:**
  - See the [FAQ](faq.md) or send an email to the mailing list [TODO].