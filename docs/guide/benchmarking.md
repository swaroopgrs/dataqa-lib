# Evaluating Your Agent

DataQA includes a benchmarking suite to help you test and evaluate your agent's performance as you refine its knowledge base. This is crucial for tracking improvements and catching regressions.

---

## 1. Why Evaluate?

- **Measure Accuracy:** Objectively determine if your agent is answering questions correctly.
- **Track Progress:** See if changes to your `schema.yml`, `rules.yml`, or `examples.yml` are making the agent smarter.
- **Prevent Regressions:** Ensure that fixing one type of query doesn't break another.

---

## 2. Preparing Test Data

The foundation of any evaluation is a good test set. Create a YAML file containing a list of questions you expect your agent to answer correctly.

**Example: `test_questions.yml`**
```yaml
# A friendly name for your test suite
use_case: "Sales Agent Evaluation"

# A list of test items
data:
  - id: "sales_total_revenue"
    question: "What is the total revenue of all sales?"
    # The expected SQL query the agent should generate
    solution:
      - function_arguments:
          sql: "SELECT SUM(revenue) FROM sales_report;"
    # Optional: A ground truth text answer for LLM-based judging
    ground_truth_output: "The total revenue is $27,650."
    active: true

  - id: "sales_by_region"
    question: "Show me the total revenue per region."
    solution:
      - function_arguments:
          sql: "SELECT region, SUM(revenue) FROM sales_report GROUP BY region;"
    active: true
```
*   `id`: A unique identifier for the test case.
*   `question`: The user query to test.
*   `solution`: The ground truth, typically the exact SQL you expect the agent to generate.
*   `ground_truth_output`: An optional final text answer, used for more advanced LLM-based judging.

---

## 3. Running the Benchmark

The benchmarking suite is run from the command line. You point it to your agent's configuration and your test data file.

```bash
# Set your LLM environment variables first!
export AZURE_OPENAI_API_KEY="..."
export OPENAI_API_BASE="..."

# Run the test
python -m dataqa.core.components.knowledge_extraction.rule_inference_batch_test \
    --config /path/to/your/agent.yaml \
    --test-data /path/to/your/test_questions.yml
```

*(Note: The entry point `rule_inference_batch_test.py` is planned to be renamed to a more general test runner in a future release.)*

---

## 4. Interpreting Results

The benchmark script will output:
- **Execution Logs:** To your console, showing each test question and the agent's generated SQL.
- **Result Files:** An Excel (`.xlsx`) and a Pickle (`.pkl`) file in a `temp/` directory containing detailed results for each test case, including:
    - The original question.
    - The expected SQL.
    - The generated SQL.
    - An `llm_label` (Correct/Wrong) if `ground_truth_output` was provided and evaluated.

By comparing the "expected_sql" and "generated_sql" columns in the output Excel file, you can quickly identify where your agent is succeeding or failing.

---

## 5. Iterative Improvement

The typical workflow is:
1.  Run the benchmark and identify a failing test case.
2.  Analyze *why* it failed. Is a column description unclear? Is a business rule missing? Is there a complex query pattern that needs an example?
3.  Update your `schema.yml`, `rules.yml`, or `examples.yml` to fix the issue.
4.  Run the benchmark again to confirm the fix and ensure no other tests have broken.
5.  Repeat.

---

## Next Steps

- **[Building Assets](building_assets.md)**: Learn how to improve your asset files based on benchmark results.
- **[Troubleshooting](troubleshooting.md)**: Common issues and solutions.
