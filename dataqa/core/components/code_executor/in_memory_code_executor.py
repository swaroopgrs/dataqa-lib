import logging
import sqlite3
from typing import Dict, List, Union, Optional

import duckdb
import pandas as pd
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession

from dataqa.core.components.base_component import (
    OutputVariable,
    Variable,
)
from dataqa.core.components.code_executor.base_code_executor import (
    CodeExecutor,
    CodeExecutorConfig,
    CodeExecutorOutput,
    DatabaseType,
)
from dataqa.core.utils.component_utils import build_base_model_from_parameters
from dataqa.core.agent.cwd_agent.error_message import ColumnNamingError

logger = logging.getLogger(__name__)


class DataFile(BaseModel):
    path: str
    table_name: str
    date_columns: List[str] = Field(default_factory=list)


class InMemoryCodeExecutorConfig(CodeExecutorConfig):
    data_files: List[DataFile] = Field(
        description="List of dictionaries containing 'path' to the CSV file and 'table_name' for the DuckDB table"
    )
    input: List[Variable] = Field(
        description="the schema of input parameters", default=[]
    )
    output: List[OutputVariable] = Field(
        description="the schema of output parameters", default=[]
    )
    backend: DatabaseType = Field(
        default=DatabaseType.duckdb,
        description="The backend to use for execution, 'duckdb', 'sqlite' or 'pyspark'",
    )
    execution_parameters: Optional[Dict[str, Union[str, int, float, bool]]] = (
        Field(
            description="Parameters for the code executor", default_factory=dict
        )
    )


class InMemoryCodeExecutor(CodeExecutor):
    component_type = "InMemoryCodeExecutor"
    config_base_model = InMemoryCodeExecutorConfig
    input_base_model = "dynamically built"
    output_base_model = CodeExecutorOutput
    config: InMemoryCodeExecutorConfig

    def __init__(
        self, config: Union[InMemoryCodeExecutorConfig, Dict], **kwargs
    ):
        super().__init__(config=config, **kwargs)
        self.input_base_model = build_base_model_from_parameters(
            base_model_name=f"{self.config.name}_input",
            parameters=self.config.input,
        )
        self.backend = self.config.backend
        if self.backend == DatabaseType.duckdb:
            db_file_path = self.config.data_files[0].path
            if db_file_path.endswith(".duckdb"):
                self.connection = duckdb.connect(db_file_path, read_only=True)
            else:
                self.connection = duckdb.connect(database=":memory:")
        elif self.backend == DatabaseType.pyspark:
            self.spark = SparkSession.builder.appName(
                "InMemoryCodeExecutor"
            ).getOrCreate()
        elif self.backend == DatabaseType.sqlite:
            pass
        else:
            raise ValueError(
                "Unsupported backend specified. Use 'duckdb', 'sqlite' or 'pyspark'."
            )
        logger.info(f"{self.component_type}: Using backend {self.backend}")
        self.load_data_into_backend()

    def display(self):
        logger.info(f"Component Name: {self.config.name}")
        logger.info(f"Component Type: {self.component_type}")
        logger.info(f"Input BaseModel: {self.input_base_model.model_fields}")
        logger.info(f"Output BaseModel: {self.output_base_model.model_fields}")

    def load_dataframe(self, path: str, date_columns: List[str]):
        if path.endswith("csv"):
            df = pd.read_csv(path, low_memory=False)
        elif path.endswith("xlsx"):
            df = pd.read_excel(path)
        else:
            raise NotImplementedError
        for date_column in date_columns:
            df[date_column] = pd.to_datetime(df[date_column])
        return df

    def load_data_into_backend(self):
        for data_file in self.config.data_files:
            path = data_file.path
            table_name = data_file.table_name
            date_columns = data_file.date_columns
            if self.backend == DatabaseType.duckdb:
                db_file_path = self.config.data_files[0].path
                if db_file_path.endswith(".duckdb"):
                    return
                dataframe = self.load_dataframe(path, date_columns)
                self.connection.register("data", dataframe)
                self.connection.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM data"
                )
                self.connection.unregister("data")
            elif self.backend == DatabaseType.pyspark:
                dataframe = self.load_dataframe(path, date_columns)
                spark_df = self.spark.createDataFrame(dataframe)
                spark_df.createOrReplaceTempView(table_name)
            elif self.backend == DatabaseType.sqlite:
                self.connection = sqlite3.connect(path)

    async def run(self, input_data, config={}) -> CodeExecutorOutput:
        try:
            if self.backend == DatabaseType.duckdb:
                result_df = self.connection.execute(input_data.code).fetchdf()
            elif self.backend == DatabaseType.pyspark:
                result_df = self.spark.sql(input_data.code).toPandas()
            elif self.backend == DatabaseType.sqlite:
                result_df = pd.read_sql_query(input_data.code, self.connection)
            else:
                raise ValueError(
                    "SQL execution failed. Unsupported backend specified. Use 'duckdb', 'sqlite' or 'pyspark'."
                )
            column_names = result_df.columns.tolist()
            if len(column_names) != len(set(column_names)):
                return CodeExecutorOutput(
                    code=input_data.code,
                    error=ColumnNamingError.message_to_agent(),
                )
            response = CodeExecutorOutput(
                code=input_data.code,
                dataframe=[result_df.to_json(index=False)],
            )
        except Exception as e:
            response = CodeExecutorOutput(code=input_data.code, error=repr(e))
        return response
