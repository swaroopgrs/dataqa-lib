import logging
from operator import add
from typing import Annotated, Dict, List, Union

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from dataqa.components.base_component import Component, ComponentConfig
from dataqa.components.plan_execute.schema import (
    Plan,
    TaskResponse,
    WorkerResponse,
    worker_response_reducer,
)
from dataqa.llm.openai import AzureOpenAI
from dataqa.memory import Memory
from dataqa.tools import DEFAULT_PLOT_TOOLS, get_plot_tools_and_descriptions
from dataqa.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    CONFIGURABLE,
)
from dataqa.utils.prompt_utils import (
    build_prompt,
    messages_to_serializable,
    prompt_type,
)

logger = logging.getLogger(__name__)


class PlotWorkerState(BaseModel):
    messages: Annotated[List, add] = Field(default_factory=list)


class PlotWorkerConfig(ComponentConfig):
    prompt: prompt_type
    tools: List[str] = Field(
        description="Tool names. Default to None for using all plot tools",
        default=None,
    )
    worker_state_required: bool = True


class PlotWorkerInput(BaseModel):
    plan: List[Plan]
    worker_response: WorkerResponse
    rule: str = ""


class PlotWorkerOutput(BaseModel):
    worker_response: Annotated[WorkerResponse, worker_response_reducer] = (
        WorkerResponse()
    )
    plot_worker_state: Annotated[List[PlotWorkerState], add] = Field(
        default_factory=list
    )


class PlotWorker(Component):
    component_type = "PlotWorker"
    config_base_model = PlotWorkerConfig
    input_base_model = PlotWorkerInput
    output_base_model = PlotWorkerOutput
    config: PlotWorkerConfig

    def __init__(
        self,
        memory: Memory,
        llm: AzureOpenAI,
        config: Union[PlotWorkerConfig, Dict] = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.llm = llm
        self.memory = memory
        self.prompt = build_prompt(self.config.prompt)
        tool_names = self.config.tools
        if not tool_names:
            tool_names = DEFAULT_PLOT_TOOLS
        self.tools = get_plot_tools_and_descriptions(
            memory=memory, tool_names=tool_names
        )[0]
        self.workflow = self.build_workflow(memory=self.memory, llm=self.llm)

    def build_workflow(
        self, memory: Memory, llm: AzureOpenAI, **kwargs
    ) -> CompiledGraph:
        return create_react_agent(
            model=llm.get_model(**kwargs), tools=self.tools
        )

    async def run(self, input_data: PlotWorkerInput, config: RunnableConfig):
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
                use_case_plot_worker_instruction=rule,
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
        
        return PlotWorkerOutput(
            worker_response=WorkerResponse(
                task_response=[
                    TaskResponse(
                        worker=worker,
                        task_description=task,
                        response=response["messages"][-1].content,
                    )
                ]
            ),
            plot_worker_state=[
                PlotWorkerState(
                    messages=messages_to_serializable(response["messages"])
                )
            ]
            if self.config.worker_state_required
            else [],
        )