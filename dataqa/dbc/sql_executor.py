from typing import Callable, Dict
import pandas as pd

from dataqa.components.code_executor.base_code_executor import CodeExecutorOutput
from dataqa.components.code_executor.in_memory_code_executor import InMemoryCodeExecutor, InMemoryCodeExecutorConfig

class DBCSQLExecutor(InMemoryCodeExecutor):
    """
    An adapter for the DBC-provided SQL callable. It inherits from InMemoryCodeExecutor
    to match the expected interface but overrides the execution logic.
    """
    def __init__(self, sql_callable: Callable, config: Dict):
        # We still call super().__init__ to set up the component's name and type,
        # but we don't need the backend or data loading.
        dummy_config = InMemoryCodeExecutorConfig(name="dbc_sql_executor", **config)
        super(InMemoryCodeExecutor, self).__init__(dummy_config)
        self.sql_callable = sql_callable

    async def run(self, input_data, config={}) -> CodeExecutorOutput:
        """
        Overrides the local execution logic to use the DBC callable.
        """
        try:
            # Assumes the sql_callable is async and returns a pandas DataFrame
            result_df = await self.sql_callable(sql=input_data.code)
            
            # The InMemoryCodeExecutor expects the dataframe to be serialized to JSON in the output
            return CodeExecutorOutput(
                code=input_data.code,
                dataframe=[result_df.to_json(index=False)],
            )
        except Exception as e:
            return CodeExecutorOutput(code=input_data.code, error=repr(e))