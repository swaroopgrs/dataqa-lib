from typing import Dict, List, Optional, Union

import yaml

from dataqa.data_models.asset_models import (
    Resource,
    ResourceType,
    RetrievedAsset,
    TableSchema,
    VectorSchema,
    VectorSchemaRecordType,
)


def get_vector_schema_record(
    resource: Resource,
    record_type: VectorSchemaRecordType,
    table_name: str,
    column_name: Optional[str],
    value: Optional[str],
) -> VectorSchema:
    for record in resource.data:
        match = (record.table_name == table_name) and (
            record.record_type == record_type
        )
        if record_type == VectorSchemaRecordType.Column:
            match = match and (record.column_name == column_name)
        if record_type == VectorSchemaRecordType.Value:
            match = match and (record.value == value)
        if match:
            return record


def reconstruct_table_schema(
    retrieved_vector_schema: List[RetrievedAsset], full_vector_schema: Resource
) -> Resource:
    tables = {}
    for record in retrieved_vector_schema:
        table_name = record.content.table_name
        table_description = record.content.table_description
        record_type = record.content.record_type
        if record_type == VectorSchemaRecordType.Table:
            matched_table = get_vector_schema_record(
                full_vector_schema, record_type, table_name, None, None
            )
            if table_name in tables:
                pass
            else:
                table = {}
                table["name"] = table_name
                table["description"] = table_description
                table["tags"] = []
                table["primary_key"] = matched_table.values["primary_key"]
                table["foreign_key"] = matched_table.values["foreign_key"]
                table["columns"] = []
                tables[table_name] = table
        elif record_type == VectorSchemaRecordType.Column:
            column_name = record.content.column_name
            column_description = record.content.column_description
            if table_name in tables:
                table = tables[table_name]
            else:
                matched_table = get_vector_schema_record(
                    full_vector_schema,
                    VectorSchemaRecordType.Table,
                    table_name,
                    None,
                    None,
                )
                table = {}
                table["name"] = table_name
                table["description"] = table_description
                table["tags"] = []
                table["primary_key"] = matched_table.values["primary_key"]
                table["foreign_key"] = matched_table.values["foreign_key"]
                table["columns"] = []
                tables[table_name] = table
            table["columns"].append(
                {
                    "name": column_name,
                    "description": column_description,
                    "type": record.content.values.get("column_type", ""),
                }
            )
        elif record_type == VectorSchemaRecordType.Value:
            column_name = record.content.column_name
            column_description = record.content.column_description
            value = record.content.value
            value_description = record.content.value_description
            if table_name in tables:
                table = tables[table_name]
            else:
                matched_table = get_vector_schema_record(
                    full_vector_schema,
                    VectorSchemaRecordType.Table,
                    table_name,
                    None,
                    None,
                )
                table = {}
                table["name"] = table_name
                table["description"] = table_description
                table["tags"] = []
                table["primary_key"] = matched_table.values["primary_key"]
                table["foreign_key"] = matched_table.values["foreign_key"]
                table["columns"] = []
                tables[table_name] = table
            if column_name in [c["name"] for c in table["columns"]]:
                column = next(
                   (c for c in table["columns"] if c["name"] == column_name),
                    None,
                )
                if "values" in column:
                    column["values"].append(
                        {"value": value, "description": value_description}
                    )
                else:
                    column["values"] = [
                        {"value": value, "description": value_description}
                    ]
            else:
                table["columns"].append(
                    {
                        "column_name": column_name,
                        "column_description": column_description,
                        "type": record.content.values.get("column_type", ""),
                        "values": [
                            {"value": value, "description": value_description}
                        ],
                    }
                )
        else:
            raise NotImplementedError
    data = []
    for table in tables.values():
        data.append(TableSchema(**table))
    reconstructed_tables = Resource(
        data=data,
        type=ResourceType.Schema,
        module_name="",
        module_type="",
        formatter="",
    )
    return reconstructed_tables


def convert_table_schema_to_sql_str(
        table_schema: Dict[str, Union[str, Dict]]
) -> str:
    """
    Converts an SQL table schema provided as a dictionary to an SQL table creation command
    For e.g. a table schema specified as below:
    {
        "name": "cpov_chase_deposit",
        "description": "This table contains deposit balances, outflows, cash buffers, etc.",
        "schema": [
            {
                "name": "ymonth",
                "type": "Integer",
                "description": "as of date"
            },
            {
                "name": "xref_cl",
                "type": "varchar",
                "description": "Unique identifier key for every customers same as experian_consumer_key in other tables"
            },
            {`
                "name": "cash_buffer",
                "type": "Numeric",
                "description": "cash buffer in months"
            }
        ],
        "tag": [
            "deposit"
        ]
    }
    gets converted to:
    -- This table contains deposit balances, outflows, cash buffers, etc.
    CREATE TABLE cpov_chase_deposit (
        /*
        description: as of date
        */
        ymonth Integer;
        /*
        description: Unique identifier key for every customers same as experian_consumer_key in other tables
        */
        xref_cl varchar;
        /*
        description: cash buffer in months
        */
        cash_buffer Numeric;
    )
    :param table_schema: Schema of the table in dictionary format
    :return: Table schema specified within the SQL table creation syntax
    """
    sql_table_creation_command =(
        "\n"
        "-- {table_description}\n"
        "\n"
        "CREATE TABLE {table_name} (\n"
        "{column_values}\n"
        ");\n"
    )
    sql_column_creation_command = """
    /*
    description: {description}{col_value_descriptions}
    */
    {column_name} {type, \n"""

    sql_column_value_description_specification = """    {value} {description}"""

    sql_columns: List[str] = []
    for column in table_schema["columns"]:
        column_value_description_list: List[str] = []
    for column_value in column.get("values", []):
        column_value_description_list.append(
            sql_column_value_description_specification.format(
                **{
                    "value": column_value["value"],
                    "description": "/ "
                    + column_value.get("description", "")
                    if "description" in column_value
                    and column_value["description"] != ""
                    else "",
                }
            )
        )
        assembled_column_value_descriptions = ""
        if column_value_description_list:
            value_string = "\n ".join(column_value_description_list)
            assembled_column_value_descriptions = (
                f"\n    values:\n  {value_string}"
            )        
        sql_columns.append(
            sql_column_creation_command.format(
                **{
                    "description": column["description"],
                    "col_value_descriptions": assembled_column_value_descriptions,
                    "column_name": column["name"],
                    "type": column["type"],
                }
            )
        )

        table_creation_syntax = sql_table_creation_command.format(
            **{
                "table_name": table_schema["name"],
                "table_description": table_schema["description"],
                "column_values": "".join(sql_columns),
            }
        )

        return table_creation_syntax


if __name__ == "__main__":
    schema_list = yaml.safe_load(
        open("examples/ccb_risk/data/ccb_risk_schema.yml")
    )
    for schema_dict in schema_list["tables"]:
        sql_schema = convert_table_schema_to_sql_str(schema_dict)
        print(sql_schema)
        