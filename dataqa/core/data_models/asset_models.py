from enum import Enum
from typing import Any, Dict, List, Optional

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
    value: str
    description: Optional[str] = None


class ForeignKey(BaseModel):
    column: str
    reference_table: str
    reference_column: str


class ColumnSchema(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    values: List[CategoricalValue] = Field(default_factory=list)
    example_values: List[Any] = Field(default_factory=list)
    distinct_count: Optional[int] = None
    null_count: Optional[int] = None
    is_primary_key: Optional[bool] = None
    foreign_key: Optional[str] = None


class TableSchema(BaseModel):
    table_name: str
    description: Optional[str] = None
    columns: List[ColumnSchema]
    row_count: Optional[int] = None
    tags: List[str] = Field(default_factory=list)
    primary_keys: List[str] = Field(default_factory=list)
    foreign_keys: List[ForeignKey] = Field(default_factory=list)


class DatabaseSchema(BaseModel):
    tables: List[TableSchema]


class Rule(BaseModel):
    rule_name: str
    module_name: Optional[str] = None
    instructions: str
    tags: List[str] = Field(default_factory=list)
    search_content: Optional[str] = None


class Rules(BaseModel):
    rules: List[Rule]


class ExampleContent(BaseModel):
    question: str
    code: str
    reasoning: Optional[str] = None


class Example(BaseModel):
    query: str
    module_name: Optional[str] = None
    example: ExampleContent
    tags: List[str] = Field(default_factory=list)
    search_content: Optional[str] = None


class Examples(BaseModel):
    examples: List[Example]


class IngestionData(BaseModel):
    rules: Optional[Rules] = None
    schema: Optional[DatabaseSchema] = None
    examples: Optional[Examples] = None


class VectorSchema(BaseModel):
    embedding_vector: List[float]
    search_content: str
    table_description: str
    table_name: str
    tags: List[Any]
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
    relevance_score: Optional[float] = Field(
        default=None,
        description="Optional relevant score assigned to the asset",
    )
    asset_id: Optional[str] = Field(
        default=None, description="Optional unique identifier for the asset"
    )