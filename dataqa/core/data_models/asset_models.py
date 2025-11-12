from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field


class ResourceType(Enum):
    Rule = "rule"
    Schema = "schema"
    Example = "example"
    VectorSchema = "vector_schema"


class VectorSchemaRecordType(Enum):
    Table = "table"
    Column = "column"
    Value = "value"


class CategoricalValue(BaseModel):
    value: Any
    description: str = Field(default="")


class ForeignKey(BaseModel):
    column: str
    reference_table: str
    reference_column: str


class ColumnSchema(BaseModel):
    name: str
    type: str
    description: str = Field(default="")
    values: list[CategoricalValue] | None = Field(default=[])
    foreign_key: str | None = None
    null_count: int | None = Field(default=None)
    distinct_count: int | None = Field(default=None)
    example_values: list[Any] | None = Field(default=[])
    is_primary_key: bool | None = Field(default=False)


class TableSchema(BaseModel):
    table_name: str
    description: str
    columns: list[ColumnSchema]
    row_count: int | None = Field(default=None)
    tags: list[str] = Field(default=[])
    primary_keys: list[str] | None = Field(default=[])
    foreign_keys: list[ForeignKey] | None = Field(default=[])


class DatabaseSchema(BaseModel):
    tables: list[TableSchema]


class Rule(BaseModel):
    rule_name: str
    module_name: str | None = None
    search_content: str | None
    tags: list[str] = Field(default=[])
    instructions: str


class Rules(BaseModel):
    rules: list[Rule]


class ExampleContent(BaseModel):
    code: str
    question: str
    reasoning: str | None = None


class Example(BaseModel):
    query: str
    module_name: str | None = None
    example: ExampleContent
    tags: list[str] = Field(default=[])
    search_content: str | None


class Examples(BaseModel):
    examples: list[Example]


class IngestionData(BaseModel):
    rules: Rules | None = None
    schema: DatabaseSchema | None = None
    examples: Examples | None = None


class VectorSchema(BaseModel):
    embedding_vector: list[float]
    search_content: str
    table_description: str
    table_name: str
    tags: list[Any]
    values: Dict[str, Any]
    record_type: VectorSchemaRecordType
    column_description: str
    column_name: str
    value: str
    value_description: str


class RetrievedAsset(BaseModel):
    """
    Data model for a retrieved knowledge asset at record level.
    """

    asset_type: str = Field(
        description="Type of the asset (e.g., 'schema', 'rule', 'example')"
    )
    content: Any = Field(
        description="Content of the retrieved asset (e.g., a Rule, Example, or TableSchema object)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the asset (e.g., source)",
    )
    relevance_score: float | None = Field(
        default=None,
        description="Optional relevant score assigned to the asset",
    )
    asset_id: str | None = Field(
        default=None, description="Optional unique identifier for the asset"
    )
