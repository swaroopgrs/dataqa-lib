import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from dataqa.core.data_models.asset_models import (
    IngestionData,
    ResourceType,
    RetrievedAsset,
    TableSchema,
    VectorSchema,
    VectorSchemaRecordType,
)


def get_vector_schema_record(
    resource: IngestionData,
    record_type: VectorSchemaRecordType,
    table_name: str,
    column_name: Optional[str],
    value: Optional[str],
) -> VectorSchema:
    """
    Retrieves a `VectorSchema` record from a `Resource` instance.

    Args:
        resource (Resource): The resource instance.
        record_type (VectorSchemaRecordType): The type of the record.
        table_name (str): The name of the table.
        column_name (Optional[str]): The name of the column (if applicable).
        value (Optional[str]): The value (if applicable).

    Returns:
        VectorSchema: The matching `VectorSchema` instance.

    Raises:
        ValueError: If no matching record is found.

    """
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
    retrieved_vector_schema: List[RetrievedAsset],
    full_vector_schema: IngestionData,
) -> IngestionData:
    """
    Reconstructs a table schema from a list of retrieved vector schemas.

    Args:
        retrieved_vector_schema (List[RetrievedAsset]): A list of retrieved vector schemas.
        full_vector_schema (Resource): A resource instance of the full vector schema.

    Returns:
        Resource: A reconstructed table schema.

    Raises:
        NotImplementedError: If the record type is not implemented.

    """
    tables = {}
    for record in retrieved_vector_schema:
        table_name = record.content.table_name
        table_description = record.content.table_description
        record_type = record.content.record_type
        if record_type == VectorSchemaRecordType.Table:
            matched_record = get_vector_schema_record(
                full_vector_schema, record_type, table_name, None, None
            )
            if table_name in tables:
                pass
            else:
                table = {}
                table["name"] = table_name
                table["description"] = table_description
                table["tags"] = []
                table["primary_key"] = matched_record.values["primary_key"]
                table["foreign_key"] = matched_record.values["foreign_key"]
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
    reconstructed_tables = IngestionData(
        data=data,
        type=ResourceType.Schema,
        module_name="",
        module_type="",
        formatter="",
    )
    return reconstructed_tables


def extract_schema_from_dataframe(
    df_dict: Dict[str, pd.DataFrame],
    meta_dict: Dict[str, Dict[str, Dict[str, Optional[str]]]],
    categorical_values_pct_threshold: float = 0.5,
    categorical_values_cnt_threshold: float = 40,
):
    """
    Generate a schema dictionary from a dictionary of dataframes and their metadata.

    Args:
        df_dict (Dict[str, pd.DataFrame]): A dictionary of dataframes, where the keys are the table names.
        meta_dict (Dict[str, Dict[str, Dict[str, Optional[str]]]]): A dictionary of metadata for each column in each table.
        categorical_values_pct_threshold (float, optional): The threshold for the percentage of unique values in a column to consider it categorical. Defaults to 0.5.
        categorical_values_cnt_threshold (float, optional): The threshold for the count of unique values in a column to consider it categorical. Defaults to 40.

    Returns:
        Dict[str, Any]: A dictionary containing the schema of the database.

    """
    schema = {
        "metadata": {
            "database_name": "my_database_name",
            "query_language": "SQL",
            "data_source": "snowflake",
            "version": "v1.01",
            "updated_at": "2025/05/09",
        }
    }

    tables = []
    for table_name, df in df_dict.items():
        meta_data = meta_dict[table_name]
        columns = []
        table = {
            "name": table_name,
            "description": "",
            "tags": [],
            "primary_key": "",
            "foreign_key": [],
            "columns": columns,
        }

        for column_name in df.columns:
            column_dtype = meta_data[column_name].get("type", None)
            column_description = meta_data[column_name].get("description", None)
            sample_values = meta_data[column_name].get("sample_values", None)
            column = {
                "name": column_name,
                "description": column_description,
                "type": column_dtype,
                "sample_values": sample_values,
            }
            values = df[column_name].unique().tolist()
            if (
                (
                    len(values) / len(df[column_name])
                    < categorical_values_pct_threshold
                )
                and (len(values) < categorical_values_cnt_threshold)
                and (column["type"] in ["string", "text"])
            ):
                print(
                    f"{column['name']}: {len(values)}, {len(df[column_name])}"
                )
                values_new = []
                for value in values:
                    values_new.append({"value": value})
                column["values"] = values_new
            columns.append(column)

        tables.append(table)
    schema["tables"] = tables
    return schema


def convert_table_schema_to_sql_str(
    table_schema: Dict[str, Union[str, Dict]],
) -> str:
    """
    Generate a SQL table creation command from a table schema.

    Args:
        table_schema (Dict[str, Union[str, Dict]]): The schema of the table.

    Returns:
        str: The SQL table creation command.

    """
    sql_table_creation_command = (
        "\n"
        "-- {table_description}\n"
        "\n"
        "CREATE TABLE {table_name} (\n"
        "{column_values}\n"
        ");\n"
    )
    sql_column_creation_command = """
    /*
    description: {description}{col_value_descriptions}{sample_values}
    */
    {column_name} {type},\n"""

    sql_column_value_description_specification = "    {value} {description}"

    sql_columns: List[str] = []
    for column in table_schema["columns"]:
        column_value_description_list: List[str] = []
        for column_value in column.get("values", []):
            column_value_description_list.append(
                sql_column_value_description_specification.format(
                    **{
                        "value": column_value["value"],
                        "description": "/ "
                        + str(column_value.get("description", ""))
                        if "description" in column_value
                        and column_value["description"] != ""
                        else "",
                    }
                )
            )
        assembled_column_value_descriptions = ""
        if column_value_description_list:
            value_string = "\n  ".join(column_value_description_list)
            assembled_column_value_descriptions = (
                f"""\n    values:\n  {value_string}"""
            )
        sample_values = column.get("sample_values", "")
        if sample_values != "":
            sample_values = "\n    sample values: " + sample_values
        sql_columns.append(
            sql_column_creation_command.format(
                **{
                    "description": column["description"],
                    "col_value_descriptions": assembled_column_value_descriptions,
                    "column_name": column["name"],
                    "type": column["type"],
                    "sample_values": sample_values,
                }
            )
        )

    table_creation_syntax = sql_table_creation_command.format(
        **{
            "table_name": table_schema["table_name"],
            "table_description": table_schema["description"],
            "column_values": "".join(sql_columns),
        }
    )

    return table_creation_syntax


def prepare_bird_database(
    database_name: str, base_path: str
) -> Tuple[
    Dict[str, pd.DataFrame], Dict[str, Dict[str, Dict[str, Optional[str]]]]
]:
    """
    Given a database name and a base path, this function will prepare the database by loading all tables
    from the database into a dictionary of pandas DataFrames. It will also load the metadata for each
    table into a dictionary of dictionaries.

    Args:
        database_name (str): The name of the database.
        base_path (str): The base path to the database.

    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Dict[str, Optional[str]]]]]: A tuple containing
            the dictionary of pandas DataFrames and the dictionary of metadata.
    """
    base_directory = Path(base_path)
    db_folder = base_directory / database_name
    db_file = db_folder / f"{database_name}.sqlite"

    with sqlite3.connect(db_file) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    table_names = [row[0] for row in tables]
    table_names = [
        table_name
        for table_name in table_names
        if table_name not in ["sqlite_sequence", "sqlite_stat1"]
    ]

    df_dict = {}
    meta_dict = {}
    for table_name in table_names:
        df = pd.read_sql_query(f"SELECT * FROM '{table_name}'", conn)
        df_dict[table_name] = df

        table_meta_file = (
            db_folder / "database_description" / f"{table_name}.csv"
        )
        if table_meta_file.exists():
            try:
                column_meta_data = pd.read_csv(table_meta_file)
            except Exception:
                column_meta_data = pd.read_csv(
                    table_meta_file, encoding="cp1252"
                )
            column_meta_data.fillna("", inplace=True)
            column_dict = {}
            for row in column_meta_data.itertuples():
                column_name = row.original_column_name.strip()
                column_type = row.data_format
                column_description = row.column_description
                unique_values = list(df[column_name].unique())
                if len(unique_values) <= 5:
                    sample_values = map(str, unique_values)
                else:
                    sample_values = map(
                        str,
                        np.random.choice(
                            list(df[column_name].unique()),
                            size=5,
                            replace=False,
                        ),
                    )
                sample_values = ", ".join(sample_values)
                column_dict[column_name] = {
                    "type": column_type,
                    "description": column_description,
                    "sample_values": sample_values,
                }

            meta_dict[table_name] = column_dict
    return df_dict, meta_dict
