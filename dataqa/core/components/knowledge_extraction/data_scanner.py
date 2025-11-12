import argparse
import asyncio
import io
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml
from pydantic import create_model

from dataqa.core.components.code_executor.in_memory_code_executor import (
    InMemoryCodeExecutor,
)
from dataqa.core.components.knowledge_extraction.infer_metadata import (
    MetaInference,
    meta_inference_prompt,
)
from dataqa.core.data_models.asset_models import (
    CategoricalValue,
    ColumnSchema,
    DatabaseSchema,
    TableSchema,
)
from dataqa.core.llm.openai import AzureOpenAI
from dataqa.core.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    DEFAULT_THREAD,
    THREAD_ID,
    TOKEN,
)
from dataqa.scripts.azure_token import get_az_token_using_cert

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
    )
)

logger.addHandler(console_handler)

DATA_SCANNER_SQL = {
    "table_list": {
        "sqlite": """SELECT name FROM sqlite_master WHERE type='table'""",
        "duckdb": """SHOW TABLES""",
        "databricks": """""",
        "redshift": """""",
        "snowflake": """""",
        "sqlserver": """SELECT TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE';""",
    },
    "column_list": {
        "sqlite": """PRAGMA table_info('{table_name}');""",
        "duckdb": """PRAGMA table_info('{table_name}');""",
        "databricks": """""",
        "redshift": """""",
        "snowflake": """""",
        "sqlserver": """SELECT
    COLUMN_NAME,
    DATA_TYPE
FROM
    INFORMATION_SCHEMA.COLUMNS
WHERE
    TABLE_SCHEMA = {database_name} AND TABLE_NAME = {table_name};""",
    },
}
TESTED_DATABASES = ["sqlite", "duckdb"]
HIGH_CARDINALITY_THRESHOLD = 50


def get_llm_and_run_config():
    """
    Retrieves the Azure OpenAI LLM and the run configuration.

    Returns:
        Tuple[AzureOpenAI, Dict[str, Any]]:
            The first element is the Azure OpenAI LLM.

            The second element is the run configuration.

    Raises:
        EnvironmentError:
            If the Azure token is not set.
    """
    token = get_az_token_using_cert()[0]
    os.environ["AZURE_OPENAI_API_TOKEN"] = token
    config = {
        "configurable": {
            THREAD_ID: DEFAULT_THREAD,
            API_KEY: os.environ.get("AZURE_OPENAI_API_KEY", ""),
            BASE_URL: os.environ.get("OPENAI_API_BASE", ""),
            TOKEN: token,
        }
    }
    llm_config = {
        "model": "gpt-4o-2024-08-06",
        "api_version": "2024-08-01-preview",
        "api_type": "azure_ad",
        "temperature": 0,
        "num_response": 1,
        "azure_model_params": {"model_name": "gpt-4o"},
    }
    llm = AzureOpenAI(**llm_config)
    return llm, config


class DataScanner:
    def __init__(self, config: Union[Dict, str]):
        if isinstance(config, str):
            self.config = yaml.safe_load(open(config, "r"))
        else:
            self.config = config

        self.executor = InMemoryCodeExecutor(
            config=self.config["executor_config"]
        )
        self.database_type = self.config["database_type"]
        self.database_schema = None
        self.inferred_schema = None

    def get_column_information_for_table(self, table_name: str) -> str:
        """
        Return a markdown formatted string of the column information for a given table.

        Args:
            table_name (str): The name of the table.

        Returns:
            str: A markdown formatted string of the column information.

        Raises:
            ValueError: If the database schema has not been extracted yet.
            ValueError: If the table is not found in the database schema.
        """
        if self.database_schema is None:
            raise ValueError("Database schema is not extracted yet.")
        for table in self.database_schema.tables:
            if table_name == table.table_name:
                column_list = []
                for column in table.columns:
                    example_values = [f"[{v}]" for v in column.example_values]
                    column_list.append(
                        [
                            column.name,
                            column.type,
                            ", ".join(example_values),
                            column.distinct_count,
                            column.null_count,
                        ]
                    )
                df_column_info = pd.DataFrame(
                    column_list,
                    columns=[
                        "column_name",
                        "column_type",
                        "example_values",
                        "distinct_count",
                        "null_count",
                    ],
                )
                return df_column_info.to_markdown()
        raise ValueError(f"Table {table_name} not found in database schema.")

    async def prepare_infer_metadata_input(self, table_name: str) -> Dict:
        """
        Prepares the input data for the `infer_metadata` method.

        Args:
            table_name (str): The name of the table.

        Returns:
            Dict: A dictionary containing the following keys:
                - table_name (str): The name of the table.
                - table_shape (str): The shape of the table.
                - sample_rows (str): The sample rows of the table in markdown format.
                - column_info (str): The column information of the table in markdown format.

        Raises:
            ValueError: If the database schema has not been extracted yet.
            ValueError: If the table is not found in the database schema.

        """
        if self.database_schema is None:
            raise ValueError("Database schema is not extracted yet.")
        for table in self.database_schema.tables:
            if table_name == table.table_name:
                table_shape = f"({table.row_count}, {len(table.columns)})"
                num_rows = self.config.get("parameters", {}).get(
                    "metadata_inference_sample_row_count", 50
                )
                sample_rows = await self.get_n_rows(
                    table_name=table_name, n=num_rows
                )
                logger.info(
                    f"Extracted {num_rows} sample rows for table {table_name}"
                )
                sample_rows_markdown = sample_rows.to_markdown(index=False)
                output = {
                    "table_name": table_name,
                    "table_shape": table_shape,
                    "sample_rows": sample_rows_markdown,
                    "column_info": self.get_column_information_for_table(
                        table_name
                    ),
                }
                return output
        raise ValueError(f"Table {table_name} not found in database schema.")

    async def execute_sql(self, sql: str) -> Optional[pd.DataFrame]:
        """
        Executes the given SQL query and returns the result as a pandas DataFrame.

        Args:
            sql (str): The SQL query to execute.

        Returns:
            Optional[pd.DataFrame]: The result of the query as a pandas DataFrame. If the query execution fails,
                None is returned.

        """
        SqlInput = create_model("SqlInput", code=(str, ...))
        executor_input = SqlInput(code=sql)
        result = await self.executor.run(executor_input)
        if result.error and len(result.dataframe) == 0:
            logger.error(
                f"Error execution SQL. \nSQL: {sql}\nError: {result.error}"
            )
            return None
        result_df: str = result.dataframe[0]
        result_df = pd.read_json(io.StringIO(result_df))
        return result_df

    async def get_list_of_tables(self) -> Optional[pd.DataFrame]:
        """
        Get a list of tables in the database.

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame with the list of tables.
                If the query execution fails, None is returned.

        Raises:
            ValueError: If the database type is not supported.

        """
        if self.database_type not in TESTED_DATABASES:
            raise ValueError(f"Unsupported database type: {self.database_type}")
        list_table_sql = DATA_SCANNER_SQL["table_list"][self.database_type]

        result = await self.execute_sql(list_table_sql)
        return result

    async def get_columns_info_of_table(
        self, table_name: str
    ) -> Optional[pd.DataFrame]:
        """
        Get the information of the columns in a table.

        Args:
            table_name (str): The name of the table.

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame with the information of the columns in the table.
                If the query execution fails, None is returned.

        Raises:
            ValueError: If the database type is not supported.

        """
        if self.database_type not in TESTED_DATABASES:
            raise ValueError(f"Unsupported database type: {self.database_type}")
        list_column_sql = DATA_SCANNER_SQL["column_list"][
            self.database_type
        ].format(table_name=table_name)
        result = await self.execute_sql(list_column_sql)
        return result

    async def get_example_values(
        self, table_name: str, column_name: str
    ) -> List[Any]:
        """
        Get example values of a column in a table.

        Args:
            table_name (str): The name of the table.
            column_name (str): The name of the column.

        Returns:
            List[Any]: A list of example values.

        """
        num_values = self.config.get("parameters", {}).get(
            "column_example_value_count", 5
        )

        get_example_sql = f'SELECT DISTINCT "{column_name}" AS unique_values FROM "{table_name}" LIMIT 50'
        values = await self.execute_sql(get_example_sql)
        if values is None:
            return []

        if len(values) > num_values:
            example_values = list(values["unique_values"].sample(num_values))
        else:
            example_values = list(values["unique_values"])
        return example_values

    async def get_distinct_value(
        self, table_name: str, column_name: str
    ) -> List:
        """
        Get distinct values of a column in a table.

        Args:
            table_name (str): The name of the table.
            column_name (str): The name of the column.

        Returns:
            List: A list of distinct values.

        """
        get_example_sql = f'SELECT DISTINCT "{column_name}" as unique_values FROM "{table_name}"'
        values = await self.execute_sql(get_example_sql)
        if values is None:
            return []
        return values["unique_values"].tolist()

    async def get_null_count(
        self, table_name: str, column_name: str
    ) -> Optional[int]:
        """
        Get the count of null values in a column of a table.

        Args:
            table_name (str): The name of the table.
            column_name (str): The name of the column.

        Returns:
            Optional[int]: The count of null values in the column.
                Returns None if the count cannot be determined.

        """
        get_example_sql = f'SELECT COUNT(*) as null_count FROM "{table_name}" WHERE "{column_name}" IS NULL'
        values = await self.execute_sql(get_example_sql)
        if values is None:
            return None
        return values["null_count"][0]

    async def get_row_count(self, table_name: str) -> Optional[int]:
        """
        Get the count of rows in a table.

        Args:
            table_name (str): The name of the table.

        Returns:
            Optional[int]: The count of rows in the table.
                Returns None if the count cannot be determined.

        """
        get_example_sql = f'SELECT COUNT(*) as row_count FROM "{table_name}"'
        values = await self.execute_sql(get_example_sql)
        if values is None:
            return None
        return values["row_count"][0]

    async def get_n_rows(
        self, table_name: str, n: int
    ) -> Optional[pd.DataFrame]:
        """
        Get the first `n` rows of a table.

        Args:
            table_name (str): The name of the table.
            n (int): The number of rows to return.

        Returns:
            Optional[pd.DataFrame]: The first `n` rows of the table.
                Returns None if the rows cannot be determined.

        """
        get_example_sql = f'SELECT * FROM "{table_name}" LIMIT {n}'
        values = await self.execute_sql(get_example_sql)
        if values is None:
            return None
        return values

    async def extract_schema(self) -> Optional[DatabaseSchema]:
        """
        Extracts the schema of the database by scanning the tables and columns.

        Returns:
            Optional[DatabaseSchema]: The extracted schema of the database.
                Returns None if the schema extraction fails.

        """
        logger.info(
            "Extracting schema by scanning database through executor..."
        )
        table_list = []
        df_table_names = await self.get_list_of_tables()
        if df_table_names is None:
            logger.error("Failed to get table list")
            return None
        for table_name in df_table_names["name"]:
            logger.info(f"Scanning table: {table_name}")
            if table_name in [
                "sqlite_sequence",
                "sqlite_stat1",
                "sqlite_stat2",
            ]:
                continue
            row_count = await self.get_row_count(table_name=table_name)
            df_columns: pd.DataFrame = await self.get_columns_info_of_table(
                table_name=table_name
            )
            if df_columns is None:
                logger.error(
                    f"Failed to get column list of table: {table_name}"
                )
                continue
            column_list = []
            for column_name, column_type, is_pk in zip(
                df_columns["name"], df_columns["type"], df_columns["pk"]
            ):
                logger.info(
                    f"Scanning column: {column_name} of table: {table_name}"
                )
                if not isinstance(is_pk, bool):
                    if isinstance(is_pk, int):
                        if is_pk == 1:
                            is_pk = True
                        elif is_pk == 0:
                            is_pk = False
                        else:
                            logger.warning(
                                f"Unrecognized is_primary_key value: {is_pk}"
                            )
                            is_pk = None
                    else:
                        logger.warning(
                            f"Unrecognized is_primary_key type: {type(is_pk)}"
                        )
                        is_pk = None

                example_values = await self.get_example_values(
                    table_name=table_name, column_name=column_name
                )
                distinct_values = await self.get_distinct_value(
                    table_name=table_name, column_name=column_name
                )
                distinct_count = (
                    len(distinct_values) if distinct_values else None
                )
                null_count = await self.get_null_count(
                    table_name=table_name, column_name=column_name
                )

                is_categorical = column_name in self.config.get(
                    "categorical_columns", {}
                ).get(table_name, [])
                categorical_values = (
                    [
                        CategoricalValue(value=str(value))
                        for value in distinct_values
                    ]
                    if is_categorical
                    else []
                )

                column_schema = ColumnSchema(
                    name=column_name,
                    type=column_type,
                    example_values=example_values,
                    distinct_count=distinct_count,
                    null_count=null_count,
                    is_primary_key=is_pk,
                    values=categorical_values,
                )
                column_list.append(column_schema)
            table_schema = TableSchema(
                table_name=table_name, columns=column_list, row_count=row_count
            )
            table_list.append(table_schema)
        database_schema = DatabaseSchema(tables=table_list)
        self.database_schema = database_schema
        output_file = f"{self.config.get('output_path')}/schema_extracted.yaml"
        with open(output_file, "w") as f:
            yaml.safe_dump(
                self.database_schema.model_dump(),
                f,
                sort_keys=False,
                default_flow_style=False,
            )
        logger.info(
            f"Schema extraction completed. Output file saved: {output_file}"
        )
        return database_schema

    async def infer_metadata(self) -> None:
        """
        Infers metadata for each table in the database schema.

        This method iterates over each table in the database schema and
        prepares the input for the `MetaInference` class. The input is
        generated using the `prepare_infer_metadata_input` method. The
        `MetaInference` class is then used to infer the metadata for each
        table.

        Returns:
            None
        """
        logger.info("Inferring metadata using LLM...")
        llm, config = get_llm_and_run_config()
        metadata_inference = MetaInference(
            llm=llm, prompt=meta_inference_prompt
        )
        inferred_schema = {}
        for table in self.database_schema.tables:
            table_name = table.table_name
            logger.info(f"Inferring metadata for table: {table_name}")
            input = await self.prepare_infer_metadata_input(
                table_name=table_name
            )
            input["config"] = config
            inferred_metadata = await metadata_inference(**input)
            inferred_schema[table_name] = inferred_metadata["metadata"]
        self.inferred_schema = inferred_schema

        self.fill_in_metadata()
        output_file = f"{self.config.get('output_path')}/schema_inferred.yaml"
        with open(output_file, "w") as f:
            yaml.safe_dump(
                self.database_schema.model_dump(),
                f,
                sort_keys=False,
                default_flow_style=False,
            )
        logger.info(
            f"Schema inference completed. Output file saved: {output_file}"
        )

    def fill_in_metadata(self) -> None:
        """
        Fill in the description of the tables and columns in the database schema using the descriptions
        inferred by the `MetaInference` class.

        This method iterates over each table in the database schema and checks if its description is
        `None`. If it is, it is replaced by the description of the table inferred by the `MetaInference`
        class.

        For each table, the method also iterates over each column and checks if its description is
        `None`. If it is, it is replaced by the description of the column inferred by the
        `MetaInference` class.

        Returns:
            None
        """
        logger.info("Filling in extracted schema with inferred metadata...")
        inferred_description = {}
        for table_name, metadata in self.inferred_schema.items():
            # inferred_description[table_name] = dict()
            for data in metadata.metadata:
                search_key = f"{table_name}.{data.level}.{data.name}"
                inferred_description[search_key] = data.description

        for table in self.database_schema.tables:
            table_name = table.table_name
            search_key = f"{table_name}.table.{table_name}"
            if search_key in inferred_description.keys():
                if table.description is None:
                    table.description = inferred_description[search_key]
            for column in table.columns:
                search_key = f"{table_name}.column.{column.name}"
                if search_key in inferred_description.keys():
                    if column.description is None:
                        column.description = inferred_description[search_key]


def run_data_scanner(config):
    data_scanner = DataScanner(config)
    _ = asyncio.run(data_scanner.extract_schema())
    asyncio.run(data_scanner.infer_metadata())
    print(
        yaml.dump(
            data_scanner.database_schema.model_dump(),
            sort_keys=False,
            default_flow_style=False,
        )
    )


def compare_schemas(schema_file_1, schema_file_2):
    schema1 = yaml.safe_load(open(schema_file_1, "r"))
    schema2 = yaml.safe_load(open(schema_file_2, "r"))
    schema1 = DatabaseSchema(**schema1)
    schema2 = DatabaseSchema(**schema2)
    schema1_output = []
    schema2_output = []
    for table in schema1.tables:
        table_name = table.table_name
        table_description = table.description
        sort_key = f"{table_name}"
        schema1_output.append(
            [sort_key, "table", table_name, table_description]
        )
        for column in table.columns:
            column_name = column.name
            column_description = column.description
            sort_key = f"{table_name}.{column_name}"
            schema1_output.append(
                [sort_key, "column", column_name, column_description]
            )
    for table in schema2.tables:
        table_name = table.table_name
        table_description = table.description
        sort_key = f"{table_name}"
        schema2_output.append(
            [sort_key, "table", table_name, table_description]
        )
        for column in table.columns:
            column_name = column.name
            column_description = column.description
            sort_key = f"{table_name}.{column_name}"
            schema2_output.append(
                [sort_key, "column", column_name, column_description]
            )
    schema1_output.sort(key=lambda x: x[0])
    schema2_output.sort(key=lambda x: x[0])
    if len(schema1_output) != len(schema2_output):
        print("Schema 1 does not match schema 2")
    else:
        combined_output = []
        for i in range(len(schema1_output)):
            combined_output.append(schema1_output[i] + schema2_output[i])
        df_combined = pd.DataFrame(
            combined_output,
            columns=[
                "sort_key_1",
                "type_1",
                "name_1",
                "description_1",
                "sort_key_2",
                "type_2",
                "name_2",
                "description_2",
            ],
        )
        df_combined.to_csv(
            "./benchmark/input/bird_dev/formula_1/schema_comparison_1.csv",
            index=False,
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Data Scanner")
    argparser.add_argument(
        "--config", type=str, help="Path to the config file."
    )
    args = argparser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))

    run_data_scanner(config)
