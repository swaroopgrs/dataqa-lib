from abc import ABC, abstractmethod
from typing import Coroutine, Dict, List, Optional, Union

from langchain_core.runnables.config import RunnableConfig
from langgraph.constants import END, START
from pydantic import BaseModel, Field

from dataqa.components.base_component import Component, ComponentConfig
from dataqa.components.base_utils import get_field
from dataqa.pipelines.constants import PIPELINE_END, PIPELINE_START


class Condition(BaseModel):
    output: str = Field(description="the name of target node")


class BaseConditionalEdgeConfig(ComponentConfig):
    condition: List[Condition] = Field(
        description="the config of every condition"
    )
    default_output: Optional[str] = Field(
        description="the output if failed to meet any conditions",
        default="__end__",
    )


class BaseConditionalEdge(Component, ABC):
    is_conditional_edge = True
    config_base_model = BaseConditionalEdgeConfig
    output_base_model = str
    config: BaseConditionalEdgeConfig

    def __init__(self, config: Union[ComponentConfig, Dict] = None, **kwargs):
        super().__init__(config=config, **kwargs)
        for condition in self.config.condition:
            if condition.output == PIPELINE_START:
                condition.output = START
            if condition.output == PIPELINE_END:
                condition.output = END

    @abstractmethod
    def check_condition(self, condition, input_data, **kwargs) -> bool:
        raise NotImplementedError

    def get_function(self) -> Coroutine:
        """
        Return a function pointer as the callable of the conditional edge.
        Add annotated types.
        """
        valid_args = [condition.output for condition in self.config.condition]
        valid_args.append(self.config.default_output)
        valid_args = list(set(valid_args))
        literal_type_str = (
            f"Literal[{', '.join([repr(s) for s in valid_args])}]"
        )

        async def func(state, config) -> eval(literal_type_str):
            return await self(state, config)

        return func

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

