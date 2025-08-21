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
    Action,
    Plan,
    ReplannerAct,
    Response,
    WorkerResponse,
)
from dataqa.core.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    CONFIGURABLE,
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
    Replanner component

    Input:
       query: str
       plan: Plan
       past_steps: List[WorkerResponse]
       memory_summary: str
    Output: (Plan or final_response)
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

        rule = input_data.rule
        if rule:
            rule = f"\n\n``Use Case Instruction``:\n{rule.strip()}"

        messages = self.prompt.invoke(
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
        api_key = config.get(CONFIGURABLE, {}).get(API_KEY, "")
        base_url = config.get(CONFIGURABLE, {}).get(BASE_URL, "")

        responses = []
        for _ in range(self.config.num_retries):
            response = await self.llm.ainvoke(
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                from_component=self.config.name,
                with_structured_output=ReplannerAct,
            )
            responses.append(response)
            if isinstance(response.generation, ReplannerAct):
                break

        if not isinstance(response.generation, ReplannerAct):
            raise Exception(
                f"Replanner failed to generate an Act. Raw LLM output: {response.generation}"
            )

        llm_output = responses if self.config.llm_output_required else []

        if response.generation.action == Action.Return:
            return PlannerOutput(
                final_response=response.generation.response,
                llm_output=llm_output,
            )
        else:
            # continue with a new plan
            # check if reach the max_tasks
            if (
                len(input_data.worker_response.task_response)
                >= self.config.max_tasks
            ):
                return PlannerOutput(
                    final_response=Response(
                        response="Reach the maximum number of steps. No final response generated.",
                        output_df_name=[],
                        output_img_name=[],
                    ),
                    llm_output=llm_output,
                )

            return PlannerOutput(
                plan=[response.generation.plan], llm_output=llm_output
            )