from typing import Dict, List, Optional, Union

from dataqa.core.data_models.asset_models import (
    RetrievedAsset,
    TableSchema,
    VectorSchema,
    VectorSchemaRecordType,
)


def get_vector_schema_record(
    resource_data: List[VectorSchema],
    record_type: VectorSchemaRecordType,
    table_name: str,
    column_name: Optional[str] = None,
    value: Optional[str] = None,
) -> Optional[VectorSchema]:
    for record in resource_data:
        match = (record.table_name == table_name) and (
            record.record_type == record_type
        )
        if record_type == VectorSchemaRecordType.Column:
            match = match and (record.column_name == column_name)
        if record_type == VectorSchemaRecordType.Value:
            match = match and (record.value == value)
        if match:
            return record
    return None


def reconstruct_table_schema(
    retrieved_vector_schema: List[RetrievedAsset],
    full_vector_schema_data: List[VectorSchema],
) -> List[TableSchema]:
    tables: Dict[str, TableSchema] = {}
    for record_asset in retrieved_vector_schema:
        record = record_asset.content
        table_name = record.table_name

        if table_name not in tables:
            matched_table_record = get_vector_schema_record(
                full_vector_schema_data,
                VectorSchemaRecordType.Table,
                table_name,
            )
            tables[table_name] = TableSchema(
                table_name=table_name,
                description=record.table_description,
                columns=[],
                tags=matched_table_record.values.get("tags", []),
                primary_keys=matched_table_record.values.get(
                    "primary_keys", []
                ),
                foreign_keys=matched_table_record.values.get(
                    "foreign_keys", []
                ),
            )

        table = tables[table_name]

        # Find or create column
        column = next(
            (c for c in table.columns if c.name == record.column_name), None
        )
        if not column and record.record_type != VectorSchemaRecordType.Table:
            column = {
                "name": record.column_name,
                "description": record.column_description,
                "type": record.values.get("column_type", "UNKNOWN"),
                "values": [],
            }
            table.columns.append(column)

        if record.record_type == VectorSchemaRecordType.Value:
            column["values"].append(
                {"value": record.value, "description": record.value_description}
            )

    return list(tables.values())


def convert_table_schema_to_sql_str(
    table_schema: Dict[str, Union[str, list]],
) -> str:
    """
    Converts a table schema dictionary (from TableSchema.model_dump()) to a
    descriptive SQL CREATE TABLE string for LLM prompts.
    """
    table_name = table_schema.get("table_name", "unknown_table")
    table_description = table_schema.get("description", "")
    columns_data = table_schema.get("columns", [])

    command_parts = []
    if table_description:
        command_parts.append(f"-- {table_description}")

    command_parts.append(f"CREATE TABLE {table_name} (")

    column_definitions = []
    for column in columns_data:
        col_def_parts = []

        # Add comment block with description and categorical values
        col_desc = column.get("description")
        col_values = column.get("values")
        if col_desc or col_values:
            col_def_parts.append("    /*")
            if col_desc:
                col_def_parts.append(f"    description: {col_desc}")
            if col_values:
                col_def_parts.append("    values:")
                for val in col_values:
                    val_desc_str = (
                        f" / {val['description']}"
                        if val.get("description")
                        else ""
                    )
                    col_def_parts.append(f"      {val['value']}{val_desc_str}")
            col_def_parts.append("    */")

        # Add the column name and type
        col_name = column.get("name", "unknown_col")
        col_type = column.get("type", "UNKNOWN_TYPE")
        col_def_parts.append(f"    {col_name} {col_type}")

        column_definitions.append("\n".join(col_def_parts))

    command_parts.append(",\n".join(column_definitions))
    command_parts.append(");")

    return "\n".join(command_parts)
