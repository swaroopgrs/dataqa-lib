import logging
from operator import add
from typing import Annotated, Dict, List, Union

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel, Field

from dataqa.agent.cwd_agent.prompt import TASK_REJECTED
from dataqa.components.base_component import Component, ComponentConfig
from dataqa.components.code_executor.in_memory_code_executor import (
    InMemoryCodeExecutor,
    InMemoryCodeExecutorConfig,
)
from dataqa.components.plan_execute.schema import (
    Plan,
    TaskResponse,
    WorkerResponse,
    worker_response_reducer,
)
from dataqa.llm.base_llm import LLMOutput
from dataqa.llm.openai import AzureOpenAI
from dataqa.memory import Memory
from dataqa.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    CONFIGURABLE,
)
from dataqa.utils.prompt_utils import build_prompt, prompt_type

logger = logging.getLogger(__name__)


class SQLGeneratorOutput(BaseModel):
    sql: str = Field(description="the generated SQL query")
    reasoning: str = Field(
        description="the reasoning procedure for generating SQL"
    )
    output: str = Field(
        description="the name of the output dataframe obtained by executing the generated SQL"
    )


class SQLExecutorOutput(BaseModel):
    sql: str
    dataframe: str = ""
    error: str = ""


class RetrievalWorkerState(BaseModel):
    task: str
    sql_generator_output: SQLGeneratorOutput = None
    sql_executor_output: SQLExecutorOutput = None
    llm_output: Annotated[List[LLMOutput], None] = Field(
        default_factory=list,
        description="the list of llm calls triggered by planner and replanner",
    )
    rule: str = ""
    schema: str = ""
    example: str = ""


class SQLGenerator:
    def __init__(self, llm: AzureOpenAI, prompt: prompt_type):
        self.llm = llm
        self.prompt = build_prompt(prompt)

    async def __call__(
        self, state: RetrievalWorkerState, config: RunnableConfig
    ):
        messages = self.prompt.invoke(
            dict(
                query=state.task,
                use_case_schema=state.schema,
                use_case_sql_instruction=state.rule,
                use_case_sql_example=state.example,
            )
        )
        api_key = config.get(CONFIGURABLE, {}).get(API_KEY, "")
        base_url = config.get(CONFIGURABLE, {}).get(BASE_URL, "")
        response = await self.llm.ainvoke(
            messages=messages,
            api_key=api_key,
            base_url=base_url,
            with_structured_output=SQLGeneratorOutput,
        )
        return {
            "sql_generator_output": response.generation,
            "llm_output": [response],
        }


class SQLExecutor(InMemoryCodeExecutor):
    def __init__(
        self,
        config: Union[InMemoryCodeExecutorConfig, Dict],
        memory: Memory,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.memory = memory

    async def __call__(
        self, state: RetrievalWorkerState, config: RunnableConfig
    ):
        sql = state.sql_generator_output.sql
        df_name = state.sql_generator_output.output
        if sql == TASK_REJECTED or df_name == TASK_REJECTED:
            error_msg = f"SQL Execution skipped, as SQL Generation task is rejected due to the following reason: {state.sql_generator_output.reasoning}"
            response = SQLExecutorOutput(sql=sql, error=error_msg)
            return {"sql_executor_output": response}
        try:
            if self.config.backend == "duckdb":
                result_df = self.connection.execute(sql).fetchdf()
            elif self.config.backend == "pyspark":
                result_df = self.spark.sql(sql).toPandas()
            self.memory.put_dataframe(name=df_name, df=result_df, config=config)
            response = SQLExecutorOutput(sql=sql, dataframe=df_name)
        except Exception as e:
            response = SQLExecutorOutput(sql=sql, error=repr(e))
        return {"sql_executor_output": response}


class RetrievalWorkerConfig(ComponentConfig):
    sql_prompt: prompt_type
    sql_execution_config: InMemoryCodeExecutorConfig
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
    retrieval_worker_state: Annotated[List[RetrievalWorkerState], None] = Field(
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
        config: Union[RetrievalWorkerConfig, Dict] = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.llm = llm
        self.memory = memory
        self.workflow = self.build_workflow(memory=self.memory, llm=self.llm)

    def build_workflow(self, memory: Memory, llm: AzureOpenAI) -> CompiledGraph:
        workflow = StateGraph(RetrievalWorkerState)

        workflow.add_node(
            "sql_generator",
            SQLGenerator(llm=llm, prompt=self.config.sql_prompt),
        )
        workflow.add_node(
            "sql_executor",
            SQLExecutor(config=self.config.sql_execution_config, memory=memory),
        )

        workflow.add_edge(START, "sql_generator")
        workflow.add_edge("sql_generator", "sql_executor")
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