# dataqa/integrations/dbc/sql_executor.py
from typing import Callable, Dict
import pandas as pd
import uuid

from dataqa.core.components.base_component import ComponentConfig
from dataqa.core.components.code_executor.base_code_executor import CodeExecutor, CodeExecutorOutput

class DBCSQLExecutor(CodeExecutor):
    """
    An adapter for the DBC-provided SQL callable.
    """
    config_base_model = ComponentConfig
    component_type = "DBCSQLExecutor"
    input_base_model = "dynamically built"
    output_base_model = CodeExecutorOutput
    
    def __init__(self, sql_callable: Callable, config_id: uuid.UUID, config: Dict):
        super().__init__(config={"name": "dbc_sql_executor", **config})
        self.sql_callable = sql_callable
        self.config_id = config_id

    async def run(self, input_data, config={}) -> CodeExecutorOutput:
        """
        Overrides the local execution logic to use the DBC callable.
        'input_data' is expected to have a 'code' attribute (the SQL string).
        """
        try:
            # The callable expects config_id and the sql query.
            result_df = await self.sql_callable(
                config_id=self.config_id,
                sql_query=input_data.code
            )
            
            # The component interface expects the dataframe to be a list of JSON strings.
            return CodeExecutorOutput(
                code=input_data.code,
                dataframe=[result_df.to_json(orient="records")],
            )
        except Exception as e:
            return CodeExecutorOutput(code=input_data.code, error=repr(e))