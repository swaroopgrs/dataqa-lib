import logging
from typing import Dict, List, Tuple, Union

from pydantic import BaseModel, Field

from dataqa.components.base_component import (
    Component,
    ComponentConfig,
    RunnableConfig,
    Variable,
)
from dataqa.utils.component_utils import build_base_model_from_parameters
from dataqa.utils.prompt_utils import build_prompt, prompt_type

logger = logging.getLogger(__name__)


class BasePromptConfig(ComponentConfig):
    prompt: prompt_type
    role: str = Field(
        default="system",
        description="the role of this generated prompt as a message",
    )
    input: List[Variable] = Field(
        description="the schema of input parameters", default=[]
    )


class BasePromptOutput(BaseModel):
    messages: List[Tuple[str, str]] = Field(
        description="the generated prompt messages"
    )


class BasePrompt(Component):
    component_type = "BasePrompt"
    config_base_model = BasePromptConfig
    input_base_model = "build dynamically from config.input"
    output_base_model = BasePromptOutput
    config: BasePromptConfig

    def __init__(self, config: Union[BasePromptConfig, Dict] = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.prompt = build_prompt(self.config.prompt)
        self.input_base_model = build_base_model_from_parameters(
            base_model_name=f"{self.config.name}_input",
            parameters=self.config.input,
        )

    def validate_llm_input(self):
        for field in self.prompt.input_schema.__annotations__:
            assert field in self.input_base_model.__annotations__, (
                f"The prompt of {self.config.name} requires `{field}` as input, but it is not defined the input BaseModel"
            )

    def display(self):
        logger.info(f"Component Name: {self.config.name}")
        logger.info(f"Component Type: {self.component_type}")
        logger.info(f"Input BaseModel: {self.input_base_model.__fields__}")
        logger.info(f"Output BaseModel: {self.output_base_model.__fields__}")

    async def run(self, input_data, config: RunnableConfig = {}):
        logger.info(
            f"Run {self.config.name} with input: {input_data.model_dump_json(indent=4)}"
        )

        assert isinstance(input_data, self.input_base_model)

        messages = self.prompt.invoke(input_data.model_dump()).to_messages()

        return self.output_base_model(
            messages=[(message.type, message.content) for message in messages]
        )
