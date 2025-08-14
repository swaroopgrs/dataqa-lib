import logging
import os.path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field
from tqdm import tqdm

from dataqa.llm.openai import OpenAIEmbedding

SCHEMA_REF_INFO = "Please see here for accepted format of schema. https://bitbucketdc-cluster07.jpmchase.net/projects/LLMCS/repos/dataqa-lib/browse/examples/ccb_risk/data/ccb_risk_schema.yml?at=refs%2Fheads%2Ffeature%2Fprompt-template"

DEFAULT_SEARCH_CONTENT_CONFIG = {
    "tables": ["name", "description"],
    "columns": ["name", "description"],
    "values": ["name", "description"],
    "include_key": False,
}

ACCEPTED_FIELDS = {
    "tables": [
        "name",
        "description",
        "tags",
        "primary_key",
        "foreign_key",
        "columns",
    ],
    "columns": [
        "name",
        "description",
        "type",
        "values",
    ],
    "values": [
        "value",
        "description",
    ],
}

REQUIRED_FIELDS = {
    "tables": [
        "name",
        "columns",
    ],
    "columns": [
        "name",
        "type",
    ],
    "values": [
        "value",
    ],
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
    embedding_vector: Optional[List[float]] = Field(
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
    embedding_vector: Optional[List[float]] = Field(
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
    embedding_vector: Optional[List[float]] = Field(
        default_factory=list,
        description="Embedding vector created from search content",
    )


def validate_fields(
    record_type: str, found_fields: List[str]
) -> Tuple[Union[bool, Set[str]]]:
    """
    validate fields of schema at different level tables, columns, and values
    :param record_type: record type: "tables", "columns", "values"
    :param found_fields: list of fields found in the schema definition
    :return: boolean (True: pass validation; False: fail validation), not supported fields, missing fields
    """
    validation_passed = True
    accepted_fields = ACCEPTED_FIELDS.get(record_type, None)
    if accepted_fields is None:
        raise ValueError(f"Record type {record_type} is not defined.")

    required_fields = REQUIRED_FIELDS.get(record_type, None)
    if required_fields is None:
        raise ValueError(f"Record type {record_type} is not defined.")

    accepted_fields = set(accepted_fields)
    required_fields = set(required_fields)
    found_fields = set(found_fields)

    not_supported_fields = found_fields - accepted_fields
    if len(not_supported_fields) > 0:
        validation_passed = False

    missing_fields = required_fields - found_fields
    if len(missing_fields) > 0:
        validation_passed = False

    return (validation_passed, not_supported_fields, missing_fields)


def record_value_to_string(
    value: dict,
    include_field: Optional[List[str]],
    display_field_name: bool = False,
) -> str:
    if display_field_name:
        if include_field is None:
            field_strings = [f"{k}: {v}" for k, v in value.items()]
        else:
            field_strings = [
                f"{k}: {v}" for k, v in value.items() if k in include_field
            ]
    else:
        if include_field is None:
            field_strings = value.values()
        else:
            field_strings = [
                f"{v}" for k, v in value.items() if k in include_field
            ]
    return "\n".join(field_strings)


class SchemaUtil:
    def __init__(self):
        self.schema = None
        self.parsed_schema = None

    def load_schema(
        self,
        schema_dict: Optional[
            Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]
        ],
        schema_file_path: Optional[str],
    ) -> None:
        """
        :param schema_dict: input schema definition
        :param schema_file_path: input yaml file with path that contains the schema definition
        """
        if schema_dict is not None:
            pass
        elif schema_file_path is not None:
            if os.path.exists(schema_file_path):
                schema_dict = yaml.safe_load(open(schema_file_path, "r"))
            else:
                raise ValueError(f"Schema file {schema_file_path} not found.")
        else:
            raise ValueError(
                "Please provide schema definition dictionary or the yaml file path that contains the schema definition."
            )
        self.schema = schema_dict

    def parsed_schema_to_json(self) -> Dict:
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
        Parse schema definition with nested structure into table, column, and categorical value records for OpenSearch indices
        Expected format of schema definition: https://bitbucketdc-cluster07.jpmchase.net/projects/LLMCS/repos/dataqa-lib/browse/examples/ccb_risk/data/ccb_risk_schema.yml?at=refs%2Fheads%2Ffeature%2Fprompt-template

        :param search_content_config: config
        """
        if search_content_config is None:
            search_content_config = DEFAULT_SEARCH_CONTENT_CONFIG

        table_records, column_records, value_records = [], [], []

        tables = self.schema.get("tables", None)
        if tables is None:
            raise ValueError(f"Tables not found. {SCHEMA_REF_INFO}")
        for table in tables:
            validated, not_supported, missing = validate_fields(
                "tables", list(table.keys())
            )
            if not validated:
                raise ValueError(
                    f"Table field error. Not supported: {not_supported}; Missing required: {missing}. {SCHEMA_REF_INFO}"
                )

            include_key = search_content_config.get("include_key")
            concat_fields = search_content_config.get("tables")
            search_content = record_value_to_string(
                table, concat_fields, include_key
            )

            table_record_value = {
                "table_name": table.get("name"),
                "table_description": table.get("description", ""),
                "primary_key": table.get("primary_key", None),
                "foreign_key": table.get("foreign_key", []),
            }
            table_record = {
                "table_name": table.get("name"),
                "table_description": table.get("description", ""),
                "tags": table.get("tags", []),
                "values": table_record_value,
                "search_content": search_content,
            }

            table_records.append(TableRecord(**table_record))

            for column in table.get("columns"):
                validated, not_supported, missing = validate_fields(
                    "columns", list(column.keys())
                )
                if not validated:
                    raise ValueError(
                        f"Column field error. Not supported: {not_supported}; Missing required: {missing}. {SCHEMA_REF_INFO}"
                    )

                include_key = search_content_config.get("include_key")
                concat_fields = search_content_config.get("columns")
                search_content = record_value_to_string(
                    table, concat_fields, include_key
                )

                column_record_value = {
                    "column_name": column.get("name"),
                    "column_type": column.get("type"),
                    "column_description": str(column.get("description", "")),
                }
                column_record_value.update(table_record_value)
                column_record = {
                    "table_name": table.get("name"),
                    "table_description": table.get("description", ""),
                    "column_name": column.get("name"),
                    "column_description": str(column.get("description", "")),
                    "tags": column.get("tags", []),
                    "values": column_record_value,
                    "search_content": search_content,
                }

                column_records.append(ColumnRecord(**column_record))

                if "values" in column:
                    for value in column.get("values"):
                        validated, not_supported, missing = validate_fields(
                            "values", list(value.keys())
                        )
                        if not validated:
                            raise ValueError(
                                f"Value field error. Not supported: {not_supported}; Missing required: {missing}. {SCHEMA_REF_INFO}"
                            )

                        include_key = search_content_config.get("include_key")
                        concat_fields = search_content_config.get("values")
                        search_content = record_value_to_string(
                            table, concat_fields, include_key
                        )

                        categorical_value = str(value.get("value"))
                        value_record_value = {
                            "value": categorical_value,
                            "value_description": value.get("description", ""),
                        }
                        value_record_value.update(column_record_value)
                        value_record = {
                            "table_name": table.get("name"),
                            "table_description": table.get("description", ""),
                            "column_name": column.get("name"),
                            "column_description": str(
                                column.get("description", "")
                            ),
                            "value": categorical_value,
                            "value_description": value.get("description", ""),
                            "tags": table.get("tags", []),
                            "values": table_record_value,
                            "search_content": search_content,
                        }
                        value_records.append(
                            CategoricalValueRecord(**value_record)
                        )
            all_records = {
                "tables": table_records,
                "columns": column_records,
                "values": value_records,
            }
        self.parsed_schema = all_records
        msg = f"Schema parsing completed. {len(table_records)} tables, {len(column_records)} columns, {len(value_records)} categorical values."
        logging.info(msg)
        print(msg)

    async def create_embedding(self, embedding_model_config: Dict) -> None:
        """
        Create embedding for parsed schema
        :param embedding_model_config: config file that contains api_key, api_version, azure_endpoint, model
        :return: None
        """
        import time

        start = time.time()
        if self.parsed_schema is None:
            raise ValueError(
                "Parsed schema not available. Please run parse_schema() function first."
            )
        else:
            embedding_model = OpenAIEmbedding()
            for schema_type, records in self.parsed_schema.items():
                for record in tqdm(
                    records, desc=f"Create embedding for {schema_type} records."
                ):
                    search_content = record.search_content
                    if search_content == "":
                        raise ValueError(
                            "Failed to create embedding. Empty search content."
                        )
                    embedding = await embedding_model(
                        search_content, **embedding_model_config
                    )
                    record.embedding_vector = embedding
        msg = f"Embedding is created for all records. Time taken: {round(time.time() - start, 2)} seconds."
        logging.info(msg)
        print(msg)

    def upload_schema_to_opensearch(self):
        pass


# if __name__ == "__main__":
#     schema_file = r"H:\Projects\jpmc_bitbucket\dataqa-lib\examples\ccb_risk\data\ccb_risk_schema.yml"
#     schema_util = SchemaUtil()
#     schema_util.load_schema(None, schema_file)
#     schema_util.parse_schema()

#     model_config = {
#         "azure_endpoint": "",
#         "openai_api_version": "2024-02-15-preview",
#         "openai_api_key": "",
#         "embedding_model_name": "text-embedding-ada-002-2",
#     }
#     asyncio.run(schema_util.create_embedding(model_config))

#     all_records = schema_util.parsed_schema_to_json()

#     yaml.safe_dump(
#         all_records,
#         open(
#             r"H:\Projects\jpmc_bitbucket\dataqa-lib\examples\ccb_risk\data\ccb_risk_schema_embedding.yml",
#             "w",
#         ),
#     )

