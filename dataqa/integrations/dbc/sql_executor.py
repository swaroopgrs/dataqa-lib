from typing import Callable, Dict

import pandas as pd

from dataqa.core.components.base_component import ComponentConfig
from dataqa.core.components.code_executor.base_code_executor import (
    CodeExecutor,
    CodeExecutorOutput,
)


class DBCSQLExecutor(CodeExecutor):
    """
    An adapter for the DBC-provided SQL callable.
    """

    config_base_model = ComponentConfig
    component_type = "DBCSQLExecutor"
    input_base_model = "dynamically built"
    output_base_model = CodeExecutorOutput

    def __init__(self, sql_callable: Callable, config: Dict):
        super().__init__(config={"name": "dbc_sql_executor", **config})
        self.sql_callable = sql_callable

    async def run(self, input_data, config={}) -> CodeExecutorOutput:
        """
        Overrides the local execution logic to use the DBC callable.
        'input_data' is expected to have a 'code' attribute (the SQL string).
        """
        try:
            # The callable expects config_id and the sql query.
            response = await self.sql_callable(sql_query=input_data.code)
            result_df = response.get("data", "")
            if isinstance(result_df, pd.DataFrame):
                result_df = result_df.to_json(orient="records")
            else:
                result_df = ""
            error = response.get("error", "")
            # The component interface expects the dataframe to be a list of JSON strings.
            return CodeExecutorOutput(
                code=input_data.code, dataframe=[result_df], error=error
            )
        except Exception as e:
            return CodeExecutorOutput(code=input_data.code, error=repr(e))
