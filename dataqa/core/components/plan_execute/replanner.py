import logging
from typing import List

from langchain_core.runnables.config import RunnableConfig
from pydantic import Field

from dataqa.core.components.plan_execute.planner import (
    Planner,
    PlannerConfig,
    PlannerInput,
    PlannerOutput,
)
from dataqa.core.components.plan_execute.schema import (
    EndCheck,
    Plan,
    Response,
    WorkerResponse,
)
from dataqa.core.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    CONFIGURABLE,
    TOKEN,
)

logger = logging.getLogger(__name__)


class ReplannerConfig(PlannerConfig):
    max_tasks: int = Field(
        description="maximum number of tasks to be executed.", default=10
    )


class ReplannerInput(PlannerInput):
    plan: List[Plan]
    history: List[str]
    worker_response: WorkerResponse
    rule: str = ""
    schema: str = ""


class Replanner(Planner):
    """
    Replanner Component

    Input:
        query: str
        plan: Plan
        past_steps: List[WorkerResponse]
        memory_summary: str
    Output: (plan or final_response)
        plan: Plan
        final_response: str
    """

    component_type = "Replanner"
    config_base_model = ReplannerConfig
    input_base_model = ReplannerInput
    output_base_model = PlannerOutput
    config: ReplannerConfig

    def display(self):
        logger.info(f"Component Name: {self.config.name}")
        logger.info(f"Component Type: {self.component_type}")
        logger.info(f"Input BaseModel: {self.input_base_model.__fields__}")
        logger.info(f"Output BaseModel: {self.output_base_model.__fields__}")

    async def run(self, input_data: ReplannerInput, config: RunnableConfig):
        assert isinstance(input_data, ReplannerInput)

        api_key = config.get(CONFIGURABLE, {}).get(API_KEY, "")
        token = config.get(CONFIGURABLE, {}).get(TOKEN, "")
        base_url = config.get(CONFIGURABLE, {}).get(BASE_URL, "")

        rule = input_data.rule
        if rule:
            rule = f"\n\n**USE CASE INSTRUCTIONS**:\n{rule.strip()}"

        messages = self.action_prompt.invoke(
            dict(
                query=input_data.query,
                history="\n".join(input_data.history),
                plan=input_data.plan[-1].summarize(),
                past_steps=input_data.worker_response.summarize(),
                dataframe_summary=self.memory.summarize_dataframe(
                    config=config
                ),
                use_case_schema=input_data.schema,
                use_case_replanner_instruction=rule,
            )
        )

        responses = []
        for _ in range(self.config.num_retries):
            response = await self.llm.ainvoke(
                messages=messages,
                api_key=api_key,
                token=token,
                base_url=base_url,
                from_component=self.config.name,
                with_structured_output=EndCheck,
            )
            if self.config.llm_output_required:
                responses.append(response)
            if isinstance(response.generation, EndCheck):
                break

        if not isinstance(response.generation, EndCheck):
            raise Exception(
                f"Replanner failed to generate an EndCheck response. Raw LLM output: {response.generation}"
            )

        # should return
        if not response.generation.should_continue:
            return PlannerOutput(
                final_response=Response(
                    response=response.generation.output_message,
                    output_df_name=response.generation.output_df_name,
                    output_img_name=response.generation.output_img_name,
                ),
                llm_output=responses,
            )

        messages = self.plan_prompt.invoke(
            dict(
                query=input_data.query,
                history="\n".join(input_data.history),
                plan=input_data.plan[-1].summarize(),
                past_steps=input_data.worker_response.summarize(),
                dataframe_summary=self.memory.summarize_dataframe(
                    config=config
                ),
                use_case_schema=input_data.schema,
                use_case_replanner_instruction=rule,
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
                f"Replanner failed to generate an Plan response. Raw LLM output: {response.generation}"
            )

        return PlannerOutput(plan=[response.generation], llm_output=responses)
