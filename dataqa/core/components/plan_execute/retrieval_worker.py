import logging
import time
from operator import add
from typing import Annotated, Dict, List, Union, Optional

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel, Field, create_model
import sqlglot
import pandas as pd

from dataqa.core.agent.cwd_agent.prompt import TASK_REJECTED
from dataqa.core.agent.cwd_agent.error_message import InternalDataframeError
from dataqa.core.components.base_component import Component, ComponentConfig
from dataqa.core.components.code_executor.base_code_executor import (
    CodeExecutor,
    DatabaseType,
    CodeExecutorConfig,
)
from dataqa.core.components.plan_execute.schema import (
    Plan,
    TaskResponse,
    WorkerResponse,
    worker_response_reducer,
)
from dataqa.core.agent.cwd_agent.error_message import SqlExecutionError
from dataqa.core.llm.base_llm import LLMOutput
from dataqa.core.llm.openai import AzureOpenAI
from dataqa.core.memory import Memory
from dataqa.core.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    CONFIGURABLE,
    QUESTION_ID,
    THREAD_ID,
    TOKEN,
)
from dataqa.core.utils.prompt_utils import build_prompt, prompt_type

logger = logging.getLogger(__name__)


class SQLGeneratorOutput(BaseModel):
    sql: str = Field(description="the generated SQL query")
    reasoning: str = Field(
        description="the reasoning procedure for generating SQL"
    )
    output: str = Field(
        description="the name of the output dataframe obtained by executing the generated SQL"
    )


class SQLValidatorOutput(BaseModel):
    unsafe: bool = Field(
        description="True if SQL statement has safety concerns. Otherwise False."
    )
    safety_concern: str = Field(
        description="If SQL is unsafe, explain in English what is the safety concern."
    )


class SQLExecutorOutput(BaseModel):
    sql: str
    dataframe: str = ""
    error: str = ""
    time: float


class RetrievalWorkerState(BaseModel):
    task: str
    sql_generator_output: SQLGeneratorOutput = None
    sql_validator_output: SQLValidatorOutput = None
    sql_executor_output: SQLExecutorOutput = None
    llm_output: Annotated[List[LLMOutput], add] = Field(
        default_factory=list,
        description="the list of llm calls triggered by planner and replanner",
    )
    rule: str = ""
    schema: str = ""
    example: str = ""


class SQLGenerator:
    def __init__(self, llm: AzureOpenAI, prompt: prompt_type, memory: Memory):
        self.llm = llm
        self.prompt = build_prompt(prompt)
        self.memory = memory

    async def __call__(
        self, state: RetrievalWorkerState, config: RunnableConfig
    ):
        thread_id = config.get(CONFIGURABLE, {}).get(THREAD_ID, "")
        question_id = config.get(CONFIGURABLE, {}).get(QUESTION_ID, "")
        logger.info(
            f"Conversation ID: {thread_id}; question ID: {question_id}. Starting to generate SQL."
        )
        messages = self.prompt.invoke(
            dict(
                query=state.task,
                use_case_schema=state.schema,
                use_case_sql_instruction=state.rule,
                use_case_sql_example=state.example,
                dataframe_summary=self.memory.summarize_dataframe(
                    config=config
                ),
            )
        )

        api_key = config.get(CONFIGURABLE, {}).get(API_KEY, "")
        token = config.get(CONFIGURABLE, {}).get(TOKEN, "")
        base_url = config.get(CONFIGURABLE, {}).get(BASE_URL, "")

        start_time = time.monotonic()
        response = await self.llm.ainvoke(
            messages=messages,
            api_key=api_key,
            token=token,
            base_url=base_url,
            with_structured_output=SQLGeneratorOutput,
        )
        run_time = time.monotonic() - start_time
        logger.info(
            f"Conversation ID: {thread_id}; question ID: {question_id}. SQL generation finished in {round(run_time, 2)} seconds."
        )

        return {
            "sql_generator_output": response.generation,
            "llm_output": [response],
        }


class SQLValidator:
    def __init__(self, llm: AzureOpenAI, prompt: prompt_type):
        self.llm = llm
        self.prompt = build_prompt(prompt)

    async def __call__(
        self, state: RetrievalWorkerState, config: RunnableConfig
    ):
        sql = state.sql_generator_output.sql
        thread_id = config.get(CONFIGURABLE, {}).get(THREAD_ID, "")
        question_id = config.get(CONFIGURABLE, {}).get(QUESTION_ID, "")
        logger.info(
            f"Conversation ID: {thread_id}; question ID: {question_id}. Starting to validate SQL."
        )

        if not sql or sql == TASK_REJECTED:
            error_msg = (
                "SQL validation skipped, as no SQL statement can be found."
            )
            if state.sql_generator_output.reasoning:
                error_msg += f" Reason: {state.sql_generator_output.reasoning}"
            response = SQLValidatorOutput(
                unsafe=False, safety_concern=error_msg
            )
            logger.error(
                f"Conversation ID: {thread_id}; question ID: {question_id}. {error_msg}"
            )
            return {"sql_validator_output": response}

        messages = self.prompt.invoke(dict(sql_statement=sql))
        api_key = config.get(CONFIGURABLE, {}).get(API_KEY, "")
        token = config.get(CONFIGURABLE, {}).get(TOKEN, "")
        base_url = config.get(CONFIGURABLE, {}).get(BASE_URL, "")

        start_time = time.monotonic()
        response = await self.llm.ainvoke(
            messages=messages,
            api_key=api_key,
            token=token,
            base_url=base_url,
            with_structured_output=SQLValidatorOutput,
        )
        run_time = time.monotonic() - start_time
        logger.info(
            f"Conversation ID: {thread_id}; question ID: {question_id}. SQL validation finished in {round(run_time, 2)} seconds."
        )

        return {
            "sql_validator_output": response.generation,
            "llm_output": [response],
        }


def clean_table_name(sql: str, table_name: str, replace_with: str) -> str:
    # Parse the SQL into an AST
    """
    Cleans the table name in the SQL by replacing the specified table name with
    another name.

    Args:
        sql (str): The SQL statement.
        table_name (str): The name of the table to be replaced.
        replace_with (str): The name to replace the table with.

    Returns:
        str: The modified SQL statement.
    """
    expression = sqlglot.parse_one(sql)

    # Find all table nodes
    for table in expression.find_all(sqlglot.exp.Table):
        if table.name == table_name:
            table.set("this", replace_with)

    # Return the modified SQL
    return expression.sql()


def dataframe_to_values_subquery(df: pd.DataFrame) -> str:
    """
    Generate a SQL string representing the DataFrame as a subquery table using VALUES.

    Args:
        df (pd.DataFrame): The DataFrame to convert.

    Returns:
        str: SQL string for use as a subquery table.
    """
    # Get column names
    columns = ", ".join(df.columns)
    # Build VALUES rows
    values = []
    for row in df.itertuples(index=False):
        row_str = ", ".join(
            f"'{v}'" if isinstance(v, str) else str(v) for v in row
        )
        values.append(f"({row_str})")
    values_clause = ",\n    ".join(values)
    return f"(VALUES\n    {values_clause}\n) AS t({columns})"


def dataframe_to_union_all_subquery(df: pd.DataFrame) -> str:
    """
    Generate a SQL string representing the DataFrame as a subquery table using UNION ALL.

    Args:
        df (pd.DataFrame): The DataFrame to convert.

    Returns:
        str: SQL string for use as a subquery table.
    """
    columns = df.columns.tolist()
    select_rows = []
    for _, row in df.iterrows():
        values = []
        for i, v in enumerate(row):
            column_name = columns[i]
            if pd.isnull(v):
                values.append(f"NULL AS {column_name}")
            elif isinstance(v, str):
                v_escaped = v.replace("'", "''")
                values.append(
                    f"'{v_escaped}' AS {column_name}"
                )  # Escape single quotes
            else:
                values.append(f"{str(v)} AS {column_name}")
        select_row = f"SELECT {', '.join(values)}"
        select_rows.append(select_row)
    union_all_sql = "\nUNION ALL\n".join(select_rows)
    return f"(\n{union_all_sql}\n)"


def clean_sql(
    sql: str,
    memory: Memory,
    config: RunnableConfig,
    executor_config: CodeExecutorConfig,
) -> Optional[str]:
    """
    Clean the generated SQL by replacing internal dataframes with their original SQL or subquery of dataframe values.
    Args:
        sql (str): The generated SQL.
        memory (Memory): The memory object that holds the dataframes.
        config (RunnableConfig): The runnable configuration.
        executor_config (CodeExecutorConfig): The type of database.
    Returns:
        Optional[str]: The cleaned SQL.
    """
    for internal_df_name in memory.list_dataframes(config):
        df, from_sql = memory.get_dataframe(
            internal_df_name, config, with_sql=True
        )
        if internal_df_name in sql:
            if executor_config.backend in [
                DatabaseType.duckdb,
                DatabaseType.snowflake,
            ]:
                value_str = dataframe_to_values_subquery(df)
                if len(value_str) < executor_config.execution_parameters.get(
                    "character_limit_of_subquery", 5000
                ):
                    sql = clean_table_name(sql, internal_df_name, value_str)
                    return sql
            elif executor_config.backend in [
                DatabaseType.sqlite,
                DatabaseType.databricks,
                DatabaseType.redshift,
            ]:
                union_all_str = dataframe_to_union_all_subquery(df)
                if len(
                    union_all_str
                ) < executor_config.execution_parameters.get(
                    "character_limit_of_subquery", 5000
                ):
                    sql = clean_table_name(sql, internal_df_name, union_all_str)
                    return sql
            if from_sql is None:
                logger.warning(
                    f"Internal dataframe {internal_df_name} found in generated SQL\n{sql}\nNo associated from_sql found for the dataframe."
                )
                return None
            else:
                logger.info(
                    f"Internal dataframe {internal_df_name} found in generated SQL\n{sql}\nReplaced with from_sql\n{from_sql}"
                )
                from_sql = from_sql.rstrip(";")
                from_sql = f"({from_sql})"
                sql = clean_table_name(sql, internal_df_name, from_sql)
                logger.info(f"SQL after replacement\n{sql}")
    return sql


class SQLExecutor:
    def __init__(self, executor: CodeExecutor, memory: Memory):
        self.executor = executor
        self.memory = memory

    async def __call__(
        self, state: RetrievalWorkerState, config: RunnableConfig
    ):
        sql = state.sql_generator_output.sql
        df_name = state.sql_generator_output.output

        thread_id = config.get(CONFIGURABLE, {}).get(THREAD_ID, "")
        question_id = config.get(CONFIGURABLE, {}).get(QUESTION_ID, "")
        logger.info(
            f"Conversation ID: {thread_id}; question ID: {question_id}. Starting to execute SQL."
        )

        if not sql:
            error_msg = (
                "SQL Execution skipped, as no SQL statement can be found."
            )
            if state.sql_generator_output.reasoning:
                error_msg += f" Reason: {state.sql_generator_output.reasoning}"
            response = SQLExecutorOutput(sql=sql, error=error_msg, time=0)
            logger.error(
                f"Conversation ID: {thread_id}; question ID: {question_id}. {error_msg}"
            )
            return {"sql_executor_output": response}

        if sql == TASK_REJECTED or df_name == TASK_REJECTED:
            error_msg = f"SQL Execution skipped, as SQL Generation task is rejected due to the following reason: {state.sql_generator_output.reasoning}"
            response = SQLExecutorOutput(sql=sql, error=error_msg, time=0)
            logger.error(
                f"Conversation ID: {thread_id}; question ID: {question_id}. {error_msg}"
            )
            return {"sql_executor_output": response}

        if state.sql_validator_output.unsafe:
            error_msg = "SQL Execution skipped, as the generated SQL statement has safety concerns."
            logger.error(
                f"Conversation ID: {thread_id}; question ID: {question_id}. {error_msg}"
            )
            error_msg += state.sql_validator_output.safety_concern
            response = SQLExecutorOutput(sql=sql, error=error_msg, time=0)
            return {"sql_executor_output": response}

        run_time = 0
        try:
            executor_config = self.executor.config
            sql_clean = clean_sql(sql, self.memory, config, executor_config)
            if sql_clean is None:
                error_msg = InternalDataframeError.message_to_agent()
                response = SQLExecutorOutput(sql=sql, error=error_msg, time=0)
                logger.error(
                    f"Conversation ID: {thread_id}; question ID: {question_id}. SQL Execution failed with error:\n{error_msg}"
                )
                return {"sql_executor_output": response}
            SqlInput = create_model("SqlInput", code=(str, ...))
            sql = sql_clean
            executor_input = SqlInput(code=sql)
            logger.debug(f"SQL executor input:\n{executor_input}")

            start_time = time.monotonic()
            executor_output = await self.executor.run(
                executor_input, config=config
            )
            run_time = time.monotonic() - start_time
            logger.info(
                f"Conversation ID: {thread_id}; question ID: {question_id}. SQL execution finished in {round(run_time, 2)} seconds."
            )

            if executor_output.error:
                sql_execution_error = SqlExecutionError(
                    cause=executor_output.error
                )
                raise Exception(sql_execution_error.message_to_agent())

            # The output dataframe is a JSON string, need to deserialize
            import json

            import pandas as pd

            result_df = pd.DataFrame(json.loads(executor_output.dataframe[0]))
            self.memory.put_dataframe(
                name=df_name, df=result_df, config=config, sql=sql
            )
            response = SQLExecutorOutput(
                sql=sql, dataframe=df_name, time=run_time
            )
        except Exception as e:
            response = SQLExecutorOutput(sql=sql, error=repr(e), time=run_time)
            logger.error(
                f"Conversation ID: {thread_id}; question ID: {question_id}. SQL Execution failed with error:\n{e}"
            )
        return {"sql_executor_output": response}


class RetrievalWorkerConfig(ComponentConfig):
    sql_generation_prompt: prompt_type
    sql_validation_prompt: prompt_type
    worker_state_required: bool = True


class RetrievalWorkerInput(BaseModel):
    plan: List[Plan]
    rule: str = ""
    schema: str = ""
    example: str = ""


class RetrievalWorkerOutput(BaseModel):
    worker_response: Annotated[WorkerResponse, worker_response_reducer] = (
        WorkerResponse()
    )
    retrieval_worker_state: Annotated[List[RetrievalWorkerState], add] = Field(
        default_factory=list
    )


class RetrievalWorker(Component):
    component_type = "RetrievalWorker"
    config_base_model = RetrievalWorkerConfig
    input_base_model = RetrievalWorkerInput
    output_base_model = RetrievalWorkerOutput
    config: RetrievalWorkerConfig

    def __init__(
        self,
        memory: Memory,
        llm: AzureOpenAI,
        sql_executor: CodeExecutor,
        config: Union[RetrievalWorkerConfig, Dict] = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.llm = llm
        self.memory = memory
        self.sql_executor = sql_executor
        self.workflow = self.build_workflow()
        logger.info(
            f"Component {self.config.name} of type {self.component_type} initialized."
        )

    def build_workflow(self) -> CompiledGraph:
        workflow = StateGraph(RetrievalWorkerState)

        workflow.add_node(
            "sql_generator",
            SQLGenerator(
                llm=self.llm,
                prompt=self.config.sql_generation_prompt,
                memory=self.memory,
            ),
        )

        workflow.add_node(
            "sql_validator",
            SQLValidator(
                llm=self.llm, prompt=self.config.sql_validation_prompt
            ),
        )

        workflow.add_node(
            "sql_executor",
            SQLExecutor(executor=self.sql_executor, memory=self.memory),
        )

        workflow.add_edge(START, "sql_generator")
        workflow.add_edge("sql_generator", "sql_validator")
        workflow.add_edge("sql_validator", "sql_executor")
        workflow.add_edge("sql_executor", END)

        return workflow.compile()

    def display(self):
        logger.info(f"Component Name: {self.config.name}")
        logger.info(f"Component Type: {self.component_type}")
        logger.info(f"Input BaseModel: {self.input_base_model.__fields__}")
        logger.info(f"Output BaseModel: {self.output_base_model.__fields__}")

    async def run(
        self, input_data: RetrievalWorkerInput, config: RunnableConfig
    ):
        assert isinstance(input_data, RetrievalWorkerInput)
        task = input_data.plan[-1].tasks[0].task_description
        worker = input_data.plan[-1].tasks[0].worker
        response = await self.workflow.ainvoke(
            RetrievalWorkerState(
                task=task,
                rule=input_data.rule,
                example=input_data.example,
                schema=input_data.schema,
            ),
            config=config,
        )
        response = RetrievalWorkerState(**response)

        output = response.sql_executor_output
        message = (
            f"To complete the task {task}, the following SQL has been generated\n"
            "```sql\n"
            f"{output.sql}\n"
            "```\n"
        )

        if output.dataframe:
            message += f"After running this SQL query, the output is saved in dataframe {output.dataframe}."
        elif response.sql_validator_output.unsafe:
            message += f"This SQL query cannot be executed due the following safety concern:\n{response.sql_validator_output.safety_concern}"
        elif output.error:
            message += f"While running this SQL query, the following runtime error was thrown:\n{output.error}"
        return RetrievalWorkerOutput(
            worker_response=WorkerResponse(
                task_response=[
                    TaskResponse(
                        worker=worker,
                        task_description=task,
                        response=message,
                    )
                ]
            ),
            retrieval_worker_state=[response]
            if self.config.worker_state_required
            else [],
        )
