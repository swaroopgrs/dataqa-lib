import logging
from typing import Dict, List, Literal, Tuple, Union

from pydantic import BaseModel, Field

from dataqa.core.components.base_component import (
    Component,
    ComponentConfig,
    OutputVariable,
    RunnableConfig,
)
from dataqa.core.llm.base_llm import BaseLLM
from dataqa.core.utils.component_utils import (
    build_base_model_from_parameters,
    extract,
)
from dataqa.core.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    CONFIGURABLE,
    TOKEN,
)

logger = logging.getLogger(__name__)


class BaseLLMComponentConfig(ComponentConfig):
    output: List[OutputVariable] = Field(
        description="the schema of output parameters", default=[]
    )
    output_parser: Literal["basemodel", "xml"] = Field(
        default="basemodel",
        description="""
            How to parse the llm generation to output_base_model.
            - Default to 'basemodel': use `llm.with_structured_output(output_base_model)`
            - If use `xml`, manually parse every field of `output_base_model` as text between <field> </field>
        """,
    )


class BaseLLMComponentInput(BaseModel):
    messages: List[Tuple[str, str]] = Field(description="the input messages")


class BaseLLMComponent(Component):
    component_type = "BaseLLMComponent"
    config_base_model = BaseLLMComponentConfig
    input_base_model = BaseLLMComponentInput
    output_base_model = "build dynamically from config.output"
    config: BaseLLMComponentConfig

    def __init__(
        self,
        llm: BaseLLM,
        config: Union[ComponentConfig, Dict] = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.llm = llm
        self.output_base_model = build_base_model_from_parameters(
            base_model_name=f"{self.config.name}_output",
            parameters=self.config.output,
        )
        if self.config.output_parser == "basemodel":
            self.llm.config.with_structured_output = self.output_base_model

    def display(self):
        logger.info(f"Component Name: {self.config.name}")
        logger.info(f"Component Type: {self.component_type}")
        logger.info(f"Input BaseModel: {self.input_base_model.__fields__}")
        logger.info(f"Output BaseModel: {self.output_base_model.__fields__}")

    async def run(self, input_data, config: RunnableConfig = {}):
        assert isinstance(input_data, self.input_base_model)

        api_key = config.get(CONFIGURABLE, {}).get(API_KEY, "")
        token = config.get(CONFIGURABLE, {}).get(TOKEN, "")
        base_url = config.get(CONFIGURABLE, {}).get(BASE_URL, "")

        if self.config.output_parser == "basemodel":
            with_structured_output = self.output_base_model
        else:
            with_structured_output = None

        response = await self.llm.ainvoke(
            messages=input_data.messages,  # TODO validation
            api_key=api_key,
            token=token,
            base_url=base_url,
            from_component=self.config.name,
            with_structured_output=with_structured_output,
        )

        if self.config.output_parser == "xml":
            assert isinstance(response.generation, str)
            response.generation = self.output_base_model(  # TODO validation
                **{
                    field: extract(
                        response.generation, f"<{field}>", f"</{field}>"
                    )
                    for field in self.output_base_model.__fields__
                }
            )

        assert isinstance(response.generation, self.output_base_model)

        return response.generation  # TODO return raw llm response to a list
