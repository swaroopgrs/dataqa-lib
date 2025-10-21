import logging
from typing import Dict, List, Literal, Union

from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field

from dataqa.components.base_component import (
    Component,
    ComponentConfig,
    OutputVariable,
    RunnableConfig,
    Variable,
)
from dataqa.llm.base_llm import BaseLLM
from dataqa.utils.component_utils import (
    build_base_model_from_parameters,
    extract,
)
from dataqa.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    CONFIGURABLE,
)
from dataqa.utils.prompt_utils import build_prompt, prompt_type

logger = logging.getLogger(__name__)


class BasePromptLLMChainConfig(ComponentConfig):
    prompt: prompt_type
    input: List[Variable] = Field(
        description="the schema of input parameters", default=[]
    )
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


class BasePromptLLMChain(Component):
    component_type = "BasePromptLLMChain"
    config_base_model = BasePromptLLMChainConfig
    input_base_model = "build dynamically from config.input"
    output_base_model = "build dynamically from config.output"
    prompt: (
        ChatPromptTemplate  # TODO should prompt be a str or a list of messages
    )
    config: BasePromptLLMChainConfig

    def __init__(
        self,
        llm: BaseLLM,
        config: Union[BasePromptLLMChainConfig, Dict] = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.llm = llm
        self.prompt = build_prompt(self.config.prompt)
        self.input_base_model = build_base_model_from_parameters(
            base_model_name=f"{self.config.name}_input",
            parameters=self.config.input,
        )
        self.output_base_model = build_base_model_from_parameters(
            base_model_name=f"{self.config.name}_output",
            parameters=self.config.output,
        )
        self.llm.config.with_structured_output = (
            self.output_base_model
        )  # add structured output
        self.validate_llm_input()

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

        messages = self.prompt.invoke(input_data.model_dump())

        api_key = config.get(CONFIGURABLE, {}).get(API_KEY, "")
        base_url = config.get(CONFIGURABLE, {}).get(BASE_URL, "")

        if self.config.output_parser == "basemodel":
            with_structured_output = self.output_base_model
        else:
            with_structured_output = None

        response = await self.llm.ainvoke(
            messages=messages,  # TODO validation
            api_key=api_key,
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

        # logger.info(
        #     f"{self.config.name} gets response {response.generation.model_dump_json(indent=4)}"
        # )

        return response.generation  # TODO return raw llm response to a list
