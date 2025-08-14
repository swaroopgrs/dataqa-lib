from enum import Enum
from typing import Any, Dict, List, Optional, Union

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


class ResourceRecordBase(BaseModel):
    tags = List[Any]
    search_content: str


class Rule(ResourceRecordBase):
    name: str
    instructions: str


class Example(ResourceRecordBase):
    query: str 
    example: Union[Dict, str]


class TableSchema(BaseModel):
    name: str
    description: str
    tags: List[Any]
    primary_key: str
    foreign_key: List[str]
    columns: List[Dict]


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


class Resource(BaseModel):
    data: List[Union[Rule, Example, TableSchema, VectorSchema]]
    type: ResourceType
    module_name: str
    module_type: str
    formatter: str


class RetrievedAsset(BaseModel):
    """
    Data model for a retrieved knowledge asset at record level.
    """

    asset_type: str = Field(
        description="Type of the asset (e.g., 'schema', 'rule', 'example')"
    )
    content: Any = Field(
        description="Content of the retrieved asset (e.g., schema definition, rule text)"
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
