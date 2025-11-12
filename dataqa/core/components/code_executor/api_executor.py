import logging
from typing import Any, Dict, Union

import pandas as pd
from pydantic import BaseModel, Field

from dataqa.core.components.code_executor.base_code_executor import (
    CodeExecutor,
    CodeExecutorConfig,
    CodeExecutorOutput,
)

logger = logging.getLogger(__name__)


class ApiCodeExecutorConfig(CodeExecutorConfig):
    """Configuration for the API Code Executor"""

    backend: str = Field(
        default="redshift",
        description="The backend to use for execution, either 'redshift' or 'snowflake'",
    )
    connect_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Connection Arguments to connect with the API Executor",
    )
    timeout: int = Field(description="Timeout in seconds", default=[])


class APICodeExecutorInput(BaseModel):
    code: str


class ApiCodeExecutor(CodeExecutor):
    """
    A code executor that connects to databases via SQLAlchemy and executes SQL queries.
    Supports any database type that SQLAlchemy supports.
    """

    component_type = "ApiCodeExecutor"
    config_base_model = ApiCodeExecutorConfig
    input_base_model = APICodeExecutorInput
    output_base_model = CodeExecutorOutput
    config: ApiCodeExecutorConfig

    def __init__(self, config: Union[ApiCodeExecutorConfig, Dict], **kwargs):
        super().__init__(config=config, **kwargs)
        self.engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the SQLAlchemy engine"""
        try:
            if self.config.backend.lower() == "snowflake":
                import snowflake.connector

                self.engine = snowflake.connector.connect(
                    **self.config.connect_args
                )
            elif self.config.backend.lower() == "redshift":
                import redshift_connector

                self.engine = redshift_connector.connect(
                    **self.config.connect_args
                )
            else:
                raise ValueError(
                    f"Unsupported backend specified: {self.config.backend}. Use 'snowflake' or 'redshift'."
                )
        except Exception as e:
            logger.error(f"Failed to initialize SQLAlchemy engine: {str(e)}")
            raise

    def display(self):
        logger.info(f"Component Name: {self.config.name}")
        logger.info(f"Component Type: {self.component_type}")
        logger.info(f"Connection String: {self.config.connect_args}")
        logger.info(f"Input BaseModel: {self.input_base_model.model_fields}")
        logger.info(f"Output BaseModel: {self.output_base_model.model_fields}")

    async def run(
        self, input_data: APICodeExecutorInput, config={}
    ) -> CodeExecutorOutput:
        """Execute SQL code using SQLAlchemy"""
        try:
            # Extract SQL code from input
            sql_code = input_data.code

            # Execute the sql
            cursor = self.engine.cursor()
            # print(f"Running...{sql_code}")
            cursor.execute(sql_code)

            # Fetch the results and column names
            rows = cursor.fetchall()
            headers = [desc[0] for desc in cursor.description]

            # Convert to DataFrame
            result_df = pd.DataFrame(rows, columns=headers)

            if isinstance(result_df, pd.DataFrame):
                response = CodeExecutorOutput(
                    code=sql_code,
                    dataframe=[
                        result_df.to_json(orient="records", date_format="iso")
                    ],
                    running_log=f"Query executed successfully. Returned {len(result_df)} rows.",
                )

                return response
            else:
                response = CodeExecutorOutput(
                    code=sql_code,
                    dataframe=[pd.DataFrame().to_json()],
                    running_log=f"Error running the query: {result_df}",
                )

        except Exception as e:
            # Handle errors
            error_message = f"Error executing SQL: {str(e)}"
            logger.error(error_message)
            return CodeExecutorOutput(code=input_data.code, error=error_message)
