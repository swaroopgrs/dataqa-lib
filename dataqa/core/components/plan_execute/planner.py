import logging
from operator import add
from typing import Annotated, Dict, List, Union

from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from dataqa.core.components.base_component import Component, ComponentConfig
from dataqa.core.components.plan_execute.schema import (
    Feasibility,
    Plan,
    Response,
)
from dataqa.core.llm.base_llm import LLMOutput
from dataqa.core.llm.openai import BaseLLM
from dataqa.core.memory import Memory
from dataqa.core.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    CONFIGURABLE,
    PROMPT_BACK,
    TOKEN,
)
from dataqa.core.utils.prompt_utils import build_prompt, prompt_type

logger = logging.getLogger(__name__)


class PlannerConfig(ComponentConfig):
    action_prompt: prompt_type
    plan_prompt: prompt_type
    num_retries: int = Field(
        description="the number of retires to counter output format errors",
        default=5,
    )
    llm_output_required: bool = True


class PlannerInput(BaseModel):
    query: str  # TODO if planner input should be dynamic
    history: List[str]
    rule: str = ""
    schema: str = ""


class PlannerOutput(BaseModel):
    final_response: Union[Response, None] = None
    plan: Annotated[List[Plan], add] = Field(default_factory=list)
    llm_output: Annotated[List[LLMOutput], add] = Field(default_factory=list)


class Planner(Component):
    """
    Planner Component

    Input:
        query: str
    Output:
        plan: Plan
    """

    component_type = "Planner"
    config_base_model = PlannerConfig
    input_base_model = PlannerInput
    output_base_model = PlannerOutput
    config: PlannerConfig

    def __init__(
        self,
        memory: Memory,
        llm: BaseLLM,
        config: Union[PlannerConfig, Dict] = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.memory = memory
        self.action_prompt = build_prompt(self.config.action_prompt)
        self.plan_prompt = build_prompt(self.config.plan_prompt)
        self.llm = llm
        logger.info(
            f"Component {self.config.name} of type {self.component_type} initialized."
        )
        # self.validate_llm_input()

    @classmethod
    def memory_required(cls):
        return True

    def display(self):
        logger.info(f"Component Name: {self.config.name}")
        logger.info(f"Component Type: {self.component_type}")
        logger.info(f"Input BaseModel: {self.input_base_model.__fields__}")
        logger.info(f"Output BaseModel: {self.output_base_model.__fields__}")

    def validate_llm_input(self):
        for field in self.action_prompt.input_schema.__annotations__:
            assert field in self.input_base_model.__annotations__, (
                f"The prompt of {self.config.name} requires `{field}` as input, but it is not defined the input BaseModel"
            )

    async def run(self, input_data: PlannerInput, config: RunnableConfig):
        assert isinstance(input_data, PlannerInput)

        api_key = config.get(CONFIGURABLE, {}).get(API_KEY, "")
        base_url = config.get(CONFIGURABLE, {}).get(BASE_URL, "")
        token = config.get(CONFIGURABLE, {}).get(TOKEN, "")
        prompt_back = config.get(CONFIGURABLE, {}).get(PROMPT_BACK, True)

        rule = input_data.rule
        if rule:
            rule = f"\n\n**USE CASE INSTRUCTIONS**:\n{rule.strip()}"

        responses = []

        if prompt_back:
            messages = self.action_prompt.invoke(
                dict(
                    query=input_data.query,
                    history="\n".join(input_data.history),
                    dataframe_summary=self.memory.summarize_dataframe(
                        config=config
                    ),
                    use_case_planner_instruction=rule,
                    use_case_schema=input_data.schema,
                )
            )

            for _ in range(self.config.num_retries):
                response = await self.llm.ainvoke(
                    messages=messages,
                    api_key=api_key,
                    token=token,
                    base_url=base_url,
                    from_component=self.config.name,
                    with_structured_output=Feasibility,
                )
                if self.config.llm_output_required:
                    responses.append(response)
                if isinstance(response.generation, Feasibility):
                    break

            if not isinstance(response.generation, Feasibility):
                raise Exception(
                    f"Planner failed to generate Feasibility response. Raw LLM output: {response.generation}"
                )

            # if not solvable, return prompt
            if not response.generation.solvable:
                return PlannerOutput(
                    final_response=Response(
                        response=response.generation.prompt_back,
                        output_df_name=[],
                        output_img_name=[],
                    ),
                    llm_output=responses,
                )
        else:
            logger.info("Skip action prompt in planner")

        # solvable, generate the plan

        messages = self.plan_prompt.invoke(
            dict(
                query=input_data.query,
                history="\n".join(input_data.history),
                dataframe_summary=self.memory.summarize_dataframe(
                    config=config
                ),
                use_case_planner_instruction=rule,
                use_case_schema=input_data.schema,
            )
        )

        for _ in range(self.config.num_retries):
            response = await self.llm.ainvoke(
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                token=token,
                from_component=self.config.name,
                with_structured_output=Plan,
            )
            if self.config.llm_output_required:
                responses.append(response)
            if isinstance(response.generation, Plan):
                break

        if not isinstance(response.generation, Plan):
            raise Exception(
                f"Planner failed to generate Plan response. Raw LLM output: {response.generation}"
            )

        return PlannerOutput(plan=[response.generation], llm_output=responses)
