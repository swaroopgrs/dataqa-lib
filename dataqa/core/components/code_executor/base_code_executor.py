from abc import ABC, abstractmethod
from typing import Any, List
from enum import Enum

from pydantic import BaseModel, Field

from dataqa.core.components.base_component import (
    Component,
    ComponentConfig,
)


class DatabaseType(Enum):
    duckdb = "duckdb"
    pyspark = "pyspark"
    sqlite = "sqlite"
    snowflake = "snowflake"
    redshift = "redshift"
    sqlserver = "sqlserver"
    databricks = "databricks"


class CodeExecutorOutput(BaseModel):
    code: str = ""
    dataframe: List[str] = Field(default_factory=list)
    image_byte_str: List[str] = Field(default_factory=list)
    html: str = ""
    markdown: str = ""
    running_log: str = ""
    error: str = ""


CodeExecutorConfig = ComponentConfig


class CodeExecutor(Component, ABC):
    config: CodeExecutorConfig
    component_type = "CodeExecutor"

    def __init__(self, config: CodeExecutorConfig):
        super().__init__(config)

    @abstractmethod
    def run(self, input_data: Any) -> CodeExecutorOutput:
        pass
