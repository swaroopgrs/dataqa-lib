# Assets API Reference

DataQA uses structured asset files (YAML) to define your data schema, business rules, and in-context learning examples.
This section documents the asset file formats and their corresponding Pydantic models.

---

## Schema Assets

Schema files describe your database tables, columns, and metadata.

### DatabaseSchema

The root object for a schema asset.

::: dataqa.core.data_models.asset_models.DatabaseSchema

### TableSchema

Represents a single table in the schema.

::: dataqa.core.data_models.asset_models.TableSchema

### ColumnSchema

Represents a single column in a table.

::: dataqa.core.data_models.asset_models.ColumnSchema

---

## Rules Assets

Rules files define business logic, SQL generation guidelines, and other constraints.

### Rules

The root object for a rules asset.

::: dataqa.core.data_models.asset_models.Rules

### Rule

Represents a single business rule.

::: dataqa.core.data_models.asset_models.Rule

---

## Examples Assets

Examples files provide in-context learning examples for LLMs.

### Examples

The root object for an examples asset.

::: dataqa.core.data_models.asset_models.Examples

### Example

Represents a single in-context example.

::: dataqa.core.data_models.asset_models.Example

---

## See Also

- [Building Your First Agent](../guide/building_your_first_agent.md)
- [Configuration Deep Dive](../guide/configuration.md)
- [API Reference: Components](components.md)
