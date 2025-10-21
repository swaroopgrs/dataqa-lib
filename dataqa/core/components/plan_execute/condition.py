from typing import Coroutine, Dict, List, Literal, Optional, Union  # noqa: F401

from langchain_core.runnables.config import RunnableConfig
from langgraph.constants import END
from pydantic import BaseModel

from dataqa.core.components.base_component import Component, ComponentConfig
from dataqa.core.components.base_utils import get_field
from dataqa.core.components.plan_execute.schema import (
    Plan,
    Response,
    WorkerName,
)

PlanConditionalEdgeConfig = ComponentConfig


class PlanConditionalEdgeInput(BaseModel):
    final_response: Union[Response, None]
    plan: List[Plan]


class PlanConditionalEdge(Component):
    component_type = "PlanConditionalEdge"
    is_conditional_edge = True
    config_base_model = PlanConditionalEdgeConfig
    input_base_model = PlanConditionalEdgeInput
    output_base_model = str

    def __init__(self, config: Union[ComponentConfig, Dict] = None, **kwargs):
        super().__init__(config=config, **kwargs)

    def get_function(self) -> Coroutine:
        valid_args = [
            WorkerName.RetrievalWorker.value,
            WorkerName.AnalyticsWorker.value,
            WorkerName.PlotWorker.value,
            END,
        ]

        literal_type_str = (
            f"Literal[{', '.join([repr(s) for s in valid_args])}]"
        )

        async def func(state, config) -> eval(literal_type_str):
            return await self(state, config)

        return func

    async def run(
        self, input_data: PlanConditionalEdgeInput, config: RunnableConfig
    ):
        if input_data.final_response is not None:
            return END
        if len(input_data.plan) == 0:
            raise ValueError(
                f"No plan or final response provided to {self.component_type}"
            )
        return input_data.plan[-1].tasks[0].worker.value

    async def __call__(self, state, config: Optional[RunnableConfig] = {}):
        # build input data from state
        input_data = {
            field: get_field(state, mapped_field)
            for field, mapped_field in self.input_mapping.items()
        }

        input_data = self.input_base_model(**input_data)

        # run
        response = await self.run(input_data=input_data, config=config)

        # validate output and update state
        assert isinstance(response, self.output_base_model)

        return response
