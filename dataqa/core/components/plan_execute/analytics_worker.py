import logging
from operator import add
from typing import Annotated, Dict, List, Union

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from dataqa.core.components.base_component import Component, ComponentConfig
from dataqa.core.components.plan_execute.schema import (
    Plan,
    TaskResponse,
    WorkerResponse,
    worker_response_reducer,
)
from dataqa.core.llm.openai import AzureOpenAI
from dataqa.core.memory import Memory
from dataqa.core.tools import (
    DEFAULT_ANALYTICS_TOOLS,
    get_analytics_tools_and_descriptions,
)
from dataqa.core.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    CONFIGURABLE,
)
from dataqa.core.utils.prompt_utils import (
    build_prompt,
    messages_to_serializable,
    prompt_type,
)

logger = logging.getLogger(__name__)


class AnalyticsWorkerState(BaseModel):
    messages: Annotated[List, add] = Field(default_factory=list)


class AnalyticsWorkerConfig(ComponentConfig):
    prompt: prompt_type
    tools: List[str] = Field(
        description="Tool names. Default to None for using all analytics tools",
        default=None,
    )
    worker_state_required: bool = True


class AnalyticsWorkerInput(BaseModel):
    plan: List[Plan]
    worker_response: WorkerResponse
    rule: str = ""


class AnalyticsWorkerOutput(BaseModel):
    worker_response: Annotated[WorkerResponse, worker_response_reducer] = (
        WorkerResponse()
    )
    analytics_worker_state: Annotated[List[AnalyticsWorkerState], add] = Field(
        default_factor=list
    )


class AnalyticsWorker(Component):
    component_type = "AnalyticsWorker"
    config_base_model = AnalyticsWorkerConfig
    input_base_model = AnalyticsWorkerInput
    output_base_model = AnalyticsWorkerOutput
    config: AnalyticsWorkerConfig

    def __init__(
        self,
        memory: Memory,
        llm: AzureOpenAI,
        config: Union[AnalyticsWorkerConfig, Dict] = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.llm = llm
        self.memory = memory
        self.prompt = build_prompt(self.config.prompt)
        tool_names = self.config.tools
        if not tool_names:
            tool_names = DEFAULT_ANALYTICS_TOOLS
        self.tools = get_analytics_tools_and_descriptions(
            memory=memory, tool_names=tool_names
        )[0]
        self.workflow = self.build_workflow(memory=self.memory, llm=self.llm)

    def build_workflow(
        self, memory: Memory, llm: AzureOpenAI, **kwargs
    ) -> CompiledGraph:
        return create_react_agent(
            model=llm._get_model(**kwargs), tools=self.tools
        )

    def display(self):
        logger.info(f"Component Name: {self.config.name}")
        logger.info(f"Component Type: {self.component_type}")
        logger.info(f"Input BaseModel: {self.input_base_model.__fields__}")
        logger.info(f"Output BaseModel: {self.output_base_model.__fields__}")

    async def run(
        self, input_data: AnalyticsWorkerInput, config: RunnableConfig
    ):
        task = input_data.plan[-1].tasks[0].task_description
        worker = input_data.plan[-1].tasks[0].worker

        rule = input_data.rule
        if rule:
            rule = f"\n\n``Use Case Instruction``:\n{rule.strip()}"

        messages = self.prompt.invoke(
            dict(
                dataframe_summary=self.memory.summarize_dataframe(
                    config=config
                ),
                plan=input_data.plan[-1].summarize(),
                task=task,
                past_steps=input_data.worker_response.summarize(),
                use_case_analytics_worker_instruction=rule,
            )
        )

        api_key = config.get(CONFIGURABLE, {}).get(API_KEY, "")
        base_url = config.get(CONFIGURABLE, {}).get(BASE_URL, "")
        self.workflow = self.build_workflow(
            memory=self.memory, llm=self.llm, api_key=api_key, base_url=base_url
        )

        response = await self.workflow.ainvoke(
            {"messages": messages.to_messages()}
        )

        return AnalyticsWorkerOutput(
            worker_response=WorkerResponse(
                task_response=[
                    TaskResponse(
                        worker=worker,
                        task_description=task,
                        response=response["messages"][-1].content,
                    )
                ]
            ),
            analytics_worker_state=[
                AnalyticsWorkerState(
                    messages=messages_to_serializable(response["messages"])
                )
            ]
            if self.config.worker_state_required
            else [],
        )
