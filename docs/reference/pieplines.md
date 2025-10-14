# Pipelines API Reference

DataQA's pipeline engine allows you to define and execute custom data workflows as directed acyclic graphs (DAGs) of components.

---

## Building a Pipeline from YAML

The primary way to create a pipeline is from a YAML configuration file.
The `build_graph_from_yaml` function is the entry point for this process.

::: dataqa.core.pipelines.pipeline.build_graph_from_yaml

**Example Usage:**
```python
from dataqa.core.pipelines.pipeline import build_graph_from_yaml

compiled_graph, pipeline_state_type = build_graph_from_yaml("path/to/pipeline.yaml")
```

---

## Pipeline Configuration Schema

These Pydantic models define the structure of the pipeline YAML files.

### PipelineConfig

The root configuration object for a pipeline.

::: dataqa.core.pipelines.schema.PipelineConfig

### Pipeline

Represents a pipeline (DAG) of nodes.

::: dataqa.core.pipelines.schema.Pipeline

### NodeEdge

Represents a node and its parent relationships in the pipeline graph.

::: dataqa.core.pipelines.schema.NodeEdge

---

## See Also

- [Building Your First Agent](../guide/building_your_first_agent.md)
- [Configuration Deep Dive](../guide/configuration.md)
- [API Reference: Components](components.md)
