# Benchmarking & Evaluation

DataQA includes a built-in benchmarking suite to help you evaluate the accuracy, latency, and performance of your agents and pipelines.

---

## 1. What Can You Benchmark?

- **Agents:** Test how well your conversational agent answers a set of questions.
- **Pipelines:** Evaluate the correctness and speed of custom data processing pipelines.
- **LLM Output:** Compare different LLMs, prompts, or configuration.

---

## 2. Preparing a Benchmark Configuration

Create a YAML file describing your benchmark.
A typical config includes:
- The agent or pipeline to test
- The test data (questions, expected answers, etc.)
- Evaluation criteria (e.g., accuracy, latency)

**Example: `benchmark_config.yml`**
```yaml
agent_config_path: "agent.yaml"
test_data_path: "tests/test_questions.yml"
output_dir: "benchmark_results/"
metrics:
  - accuracy
  - latency
```

---

## 3. Preparing Test Data

Test data is usually a YAML file with a list of questions and expected outputs.

**Example: `test_questions.yml`**
```yaml
- id: test_001
  question: "What is the total sales for 2024?"
  expected_answer: "..."
- id: test_002
  question: "Show all active customers."
  expected_answer: "..."
```

---

## 4. Running the Benchmark

Use the benchmarking runner module:

```bash
python -m dataqa.benchmark.run_test -c benchmark_config.yml
```

- Results will be saved in the specified `output_dir`.
- You'll get a summary of accuracy, latency, and any errors.

---

## 5. Interpreting Results

- **Accuracy:** Percentage of questions answered correctly (matches expected output).
- **Latency:** Time taken per query or pipeline run.
- **Detailed Logs:** See which questions failed and why.

Results are typically saved as CSV or Excel files for easy analysis.

---

## 6. Advanced: Custom Evaluation Metrics

You can add your own evaluation logic by subclassing the evaluation classes in `dataqa/benchmark/schema.py`.

---

## 7. Best Practices

- **Start with a small test set** to validate your setup.
- **Iterate:** Add more questions and edge cases as your agent improves.
- **Automate:** Integrate benchmarking into your CI/CD pipeline for regression testing.

---

## Next Steps

- [Deployment Guide](deployment.md)
- [API Reference: Benchmarking](../reference/benchmark.md)
- [FAQ](faq.md)

---

## Need Help?

- [FAQ](faq.md)
