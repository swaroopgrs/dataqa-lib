# Tools API Reference

DataQA provides a suite of built-in analytics and plotting tools that can be used by agents and pipelines for data analysis and visualization.

---

## Analytics Tools

Analytics tools operate on Pandas DataFrames and provide common data analysis operations (aggregation, filtering, transformation, etc.).

### Built-in Analytics Tools

- **CalculatorTool:** Evaluate arithmetic expressions on DataFrames.
- **QueryDataframeTool:** Evaluate boolean expressions.
- **ColumnSelector:** Filter rows using boolean expressions.
- **GroupBy:** Group and aggregate data.
- **MergeDataframes:** Merge two DataFrames.
- **ConcatenateDataframes:** Concatenate DataFrames along rows or columns.
- **ValueCounts:** Count unique values.
- **Largest / Smallest:** Get top/bottom N rows.
- **GetUniqueValues:** Get unique values from a column.
- **AbsoluteTool:** Compute absolute values.
- **CalculateCorrelationMatrix:** Compute correlation matrix.

**See:**
::: dataqa.core.tools.analytics.tool_generator

---

## Plotting Tools

Plotting tools generate visualizations from DataFrames.

### Built-in Plotting Tools

- **Plots:** Generate scatter, bar, line, pie, hist, or box plots using matplotlib/seaborn.
- **See:**
::: dataqa.core.tools.plot.tool_generator

---

## Using Tools in Agents

Tools are used by the `AnalyticsWorker` and `PlotWorker` components.
You can reference built-in tools by name in your agent configuration or prompt templates.

---

## Extending Tools

You can add your own analytics or plotting tools by registering them in the tool registry.

**Example:**
```python
from dataqa.core.tools.analytics.tool_generator import DEFAULT_ANALYTICS_TOOLS

def my_custom_tool(memory):
  # Your tool logic here
  pass

DEFAULT_ANALYTICS_TOOLS["MyCustomTool"] = my_custom_tool
```

See [Extending DataQA](../guide/extending.md) for more details.

---

## See Also

- [Extending DataQA](../guide/extending.md)
- [API Reference: Components](components.md)
