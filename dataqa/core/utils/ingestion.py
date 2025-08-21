import logging
import os.path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field
from tqdm import tqdm

from dataqa.core.data_models.asset_models import (
    ColumnSchema,
    DatabaseSchema,
    TableSchema,
)
from dataqa.core.llm.openai import OpenAIEmbedding

DEFAULT_SEARCH_CONTENT_CONFIG = {
    "tables": ["table_name", "description"],
    "columns": ["name", "description"],
    "values": ["value", "description"],
    "include_key": False,
}


class TableRecord(BaseModel):
    """Record of table index"""

    table_name: str = Field(description="Physical name of the table")
    table_description: str = Field(
        description="Description of the table. May also contain custom information, such as synonym of table, example questions..."
    )
    tags: List[str] = Field(description="List of tags for the table")
    search_content: str = Field(
        description="Search content used to generate embedding from. Default: table name + table description without key."
    )
    values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Table record value that will be retrieved and used in downstream processing and/or in the prompt.",
    )
    embedding_vector: List[float] = Field(
        default_factory=list,
        description="Embedding vector created from search content",
    )


class ColumnRecord(BaseModel):
    """Record of column index"""

    table_name: str = Field(description="Physical name of the table")
    table_description: str = Field(
        description="Description of the table. May also contain custom information, such as synonym of table, example questions..."
    )
    column_name: str = Field(description="Physical name of the column")
    column_description: str = Field(
        description="Description of the column. May also contain custom information, such as synonym of column"
    )
    tags: List[str] = Field(description="List of tags for the table")
    search_content: str = Field(
        description="Search content used to generate embedding from. Default: column name + column description without key."
    )
    values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Table record value that will be retrieved and used in downstream processing and/or in the prompt.",
    )
    embedding_vector: List[float] = Field(
        default_factory=list,
        description="Embedding vector created from search content",
    )


class CategoricalValueRecord(BaseModel):
    """Record of categorical value index"""

    table_name: str = Field(description="Physical name of the table")
    table_description: str = Field(
        description="Description of the table. May also contain custom information, such as synonym of table, example questions..."
    )
    column_name: str = Field(description="Physical name of the column")
    column_description: str = Field(
        description="Description of the column. May also contain custom information, such as synonym of column"
    )
    value: str = Field(description="Unique value of a categorical column")
    value_description: str = Field(
        description="Description of the categorical value. May also contain custom information, such as synonym of the value"
    )
    tags: List[str] = Field(description="List of tags for the table")
    search_content: str = Field(
        description="Search content used to generate embedding from. Default: value + value description."
    )
    values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Table record value that will be retrieved and used in downstream processing and/or in the prompt.",
    )
    embedding_vector: List[float] = Field(
        default_factory=list,
        description="Embedding vector created from search content",
    )


def record_value_to_string(
    record_model: Union[BaseModel],
    include_fields: Optional[List[str]],
    display_field_name: bool = False,
) -> str:
    """Creates a concatenated string from specified fields of a Pydantic model."""
    if not include_fields:
        return ""

    field_strings = []
    record_dict = record_model.model_dump()
    for field in include_fields:
        value = record_dict.get(field)
        if value:
            if display_field_name:
                field_strings.append(f"{field}: {value}")
            else:
                field_strings.append(str(value))
    return "\n".join(field_strings)


class SchemaUtil:
    def __init__(self):
        self.schema: Optional[DatabaseSchema] = None
        self.parsed_schema = None

    def load_schema(
        self,
        schema_dict: Optional[Dict],
        schema_file_path: Optional[str],
    ) -> None:
        """
        Loads a schema from a dictionary or a YAML file into a Pydantic model.
        """
        if schema_dict:
            self.schema = DatabaseSchema(**schema_dict)
        elif schema_file_path and os.path.exists(schema_file_path):
            with open(schema_file_path, "r") as f:
                raw_schema = yaml.safe_load(f)
            self.schema = DatabaseSchema(**raw_schema)
        else:
            raise ValueError(
                "Please provide a schema dictionary or a valid YAML file path."
            )

    def parsed_schema_to_json(self) -> Dict:
        if not self.parsed_schema:
            return {}
        all_records_dict = {
            k: [r.model_dump() for r in v]
            for k, v in self.parsed_schema.items()
        }
        return all_records_dict

    def parse_schema(
        self,
        search_content_config: Optional[
            Dict[str, Union[List[str], bool]]
        ] = None,
    ) -> None:
        """
        Parse schema definition from the loaded DatabaseSchema model into records
        for vectorization.
        """
        if not self.schema:
            raise ValueError("Schema not loaded. Please call load_schema() first.")

        if search_content_config is None:
            search_content_config = DEFAULT_SEARCH_CONTENT_CONFIG

        table_records, column_records, value_records = [], [], []

        for table in self.schema.tables:
            table_search_content = record_value_to_string(
                table,
                search_content_config.get("tables"),
                search_content_config.get("include_key", False),
            )
            table_values = table.model_dump(include={'table_name', 'description', 'primary_keys', 'foreign_keys'})
            
            table_records.append(
                TableRecord(
                    table_name=table.table_name,
                    table_description=table.description or "",
                    tags=table.tags,
                    values=table_values,
                    search_content=table_search_content,
                )
            )

            for column in table.columns:
                col_search_content = record_value_to_string(
                    column,
                    search_content_config.get("columns"),
                    search_content_config.get("include_key", False),
                )
                column_values = column.model_dump(include={'name', 'type', 'description'})
                
                column_records.append(
                    ColumnRecord(
                        table_name=table.table_name,
                        table_description=table.description or "",
                        column_name=column.name,
                        column_description=column.description or "",
                        tags=table.tags,
                        values=column_values,
                        search_content=col_search_content,
                    )
                )

                if column.values:
                    for value in column.values:
                        val_search_content = record_value_to_string(
                            value,
                            search_content_config.get("values"),
                            search_content_config.get("include_key", False),
                        )
                        value_record_values = value.model_dump()

                        value_records.append(
                            CategoricalValueRecord(
                                table_name=table.table_name,
                                table_description=table.description or "",
                                column_name=column.name,
                                column_description=column.description or "",
                                value=value.value,
                                value_description=value.description or "",
                                tags=table.tags,
                                values=value_record_values,
                                search_content=val_search_content,
                            )
                        )

        self.parsed_schema = {
            "tables": table_records,
            "columns": column_records,
            "values": value_records,
        }
        msg = f"Schema parsing completed. {len(table_records)} tables, {len(column_records)} columns, {len(value_records)} categorical values."
        logging.info(msg)

    async def create_embedding(self, embedding_model_config: Dict) -> None:
        """
        Create embedding for parsed schema. This is part of the local mode toolkit.
        """
        import time

        start = time.time()
        if self.parsed_schema is None:
            raise ValueError(
                "Parsed schema not available. Please run parse_schema() function first."
            )
        
        embedding_model = OpenAIEmbedding()
        for schema_type, records in self.parsed_schema.items():
            for record in tqdm(
                records, desc=f"Create embedding for {schema_type} records."
            ):
                search_content = record.search_content
                if not search_content:
                    logger.warning(
                        f"Skipping embedding for {schema_type} record due to empty search content: {record.model_dump()}"
                    )
                    continue
                embedding = await embedding_model(
                    search_content, **embedding_model_config
                )
                record.embedding_vector = embedding
        msg = f"Embedding created for all records. Time taken: {round(time.time() - start, 2)} seconds."
        logging.info(msg)