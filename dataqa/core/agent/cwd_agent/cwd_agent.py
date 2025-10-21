import asyncio
import logging
import time
import traceback
from enum import Enum
from operator import add
from typing import Annotated, Coroutine, Dict, Generator, List, Tuple, Union

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import START, StateGraph
from pydantic import BaseModel, Field

from dataqa.code.utils.prompt_utils import build_prompt
from dataqa.core.agent.base import Agent
from dataqa.core.agent.cwd_agent.config import (
    CwdAgentDefinitionConfig,
)
from dataqa.core.agent.cwd_agent.prompt import (
    instantiate_analytics_worker_prompt_by_use_case_jinja,
    instantiate_planner_prompt_by_use_case_jinja,
    instantiate_plot_worker_prompt_by_use_case_jinja,
    instantiate_replanner_prompt_by_use_case_jinja,
    instantiate_sql_generator_prompt_by_use_cas_jinja,
    instantiate_sql_validation_prompt_jinja,
    instantiate_summarization_prompt_by_use_case_jinja,
)
from dataqa.core.components.code_executor.base_code_executor import CodeExecutor
from dataqa.core.components.plan_execute.analytics_worker import (
    AnalyticsWorker,
    AnalyticsWorkerConfig,
    AnalyticsWorkerState,
)
from dataqa.core.components.plan_execute.condition import (
    PlanConditionalEdge,
    PlanConditionalEdgeConfig,
)
from dataqa.core.components.plan_execute.planner import Planner, PlannerConfig
from dataqa.core.components.plan_execute.plot_worker import (
    PlotWorker,
    PlotWorkerConfig,
    PlotWorkerState,
)
from dataqa.core.components.plan_execute.replanner import (
    Replanner,
    ReplannerConfig,
)
from dataqa.core.components.plan_execute.retrieval_worker import (
    RetrievalWorker,
    RetrievalWorkerConfig,
    RetrievalWorkerState,
)
from dataqa.core.components.plan_execute.schema import (
    PlanExecuteState,
    Response,
    worker_response_reducer,
)
from dataqa.core.components.resource_manager.resource_manager import (
    ResourceManager,
)
from dataqa.core.llm.base_llm import BaseLLM
from dataqa.core.memory import Memory
from dataqa.core.tools import (
    get_analytics_tools_and_descriptions,
    get_plot_tools_and_descriptions,
)
from dataqa.core.utils.agent_util import AgentResponseParser
from dataqa.core.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    CONFIGURABLE,
    DEBUG,
    QUESTION_ID,
    THREAD_ID,
    TOKEN,
)
from dataqa.core.utils.utils import cls_from_str

logger = logging.getLogger(__name__)


class StatusMessage(Enum):
    retriever = "Retrieving data schema and business rules..."
    planner = "Generating a plan..."
    replanner = "Evaluating the progress and updating the plan..."
    sql_generator = "Generating SQL query..."
    sql_executor = "Executing SQL query..."
    analytics_worker = "Performing data analysis..."
    plot_worker = "Visualizing data..."


class Summary(BaseModel):
    "The summary of agent working trajectory."

    summary: str


class CWDState(PlanExecuteState):
    summary: str = ""
    retrieval_worker_state: Annotated[List[RetrievalWorkerState], add] = Field(
        default_factory=list
    )
    analytics_worker_state: Annotated[List[AnalyticsWorkerState], add] = Field(
        default_factory=list
    )
    plot_worker_state: Annotated[List[PlotWorkerState], add] = Field(
        default_factory=list
    )
    planner_rule: str = ""
    planner_schema: str = ""
    planner_example: str = ""
    replanner_rule: str = ""
    replanner_schema: str = ""
    replanner_example: str = ""
    retrieval_worker_rule: str = ""
    retrieval_worker_schema: str = ""
    retrieval_worker_example: str = ""
    analytics_worker_rule: str = ""
    analytics_worker_schema: str = ""
    analytics_worker_example: str = ""
    plot_worker_rule: str = ""
    plot_worker_schema: str = ""
    plot_worker_example: str = ""
    schema_fallback_level: str = ""
    error: str = ""
    total_time: float = 0

    def update_field(self, field, value):
        if not hasattr(self, field):
            raise ValueError(f"{field} is not a valid field for CWDState")
        if field in [
            "plan",
            "log",
            "retrieval_worker_state",
            "analytics_worker_state",
            "plot_worker_state",
            "llm_output",
        ]:
            value = getattr(self, field) + value
        if field == "worker_response":
            value = worker_response_reducer(getattr(self, field), value)
        setattr(self, field, value)


class CWDAgent(Agent):
    """
    CWD Agent
    """

    components = [
        "default",
        "planner",
        "replanner",
        "retrieval_worker",
        "analytics_worker",
        "plot_worker",
    ]

    def __init__(
        self,
        memory: Memory,
        config: CwdAgentDefinitionConfig,
        llms: Dict[str, BaseLLM],
        resource_manager: ResourceManager,
        sql_executor: CodeExecutor,
    ):
        self.config = config
        self.llms = llms
        self.resource_manager = resource_manager
        self.sql_executor = sql_executor

        # 1. Load tools and descriptions
        (
            self.analytics_tools,
            self.analytics_worker_short_tool_description,
            self.analytics_worker_long_tool_description,
        ) = get_analytics_tools_and_descriptions(memory)

        (
            self.plot_tools,
            self.plot_worker_short_tool_description,
            self.plot_worker_long_tool_description,
        ) = get_plot_tools_and_descriptions(memory)

        # 2. Instantiate prompt templates
        self.processed_prompts = self._instantiate_prompt_template(
            analytics_worker_short_tool_description=self.analytics_worker_short_tool_description,
            analytics_worker_long_tool_description=self.analytics_worker_long_tool_description,
            plot_worker_short_tool_description=self.plot_worker_short_tool_description,
            plot_worker_long_tool_description=self.plot_worker_long_tool_description,
        )

        # 3. Instantiate the retriever
        self.retriever = cls_from_str(config.retriever_config.type)(
            config=config.retriever_config.config,
            resource_manager=self.resource_manager,
        )
        self.retriever.set_input_mapping(dict(query="query"))
        retriever_output = {}
        for field in self.retriever.output_base_model.__fields__:
            retriever_output[field] = field
        self.retriever.output_mapping = retriever_output

        # 3.1 Instantiate fallback retriever
        if config.fallback_retriever_config is not None:
            self.fallback_retriever = cls_from_str(
                config.fallback_retriever_config.type
            )(
                config=config.fallback_retriever_config.config,
                resource_manager=self.resource_manager,
            )
            self.fallback_retriever.set_input_mapping(dict(query="query"))
            retriever_output = {}
            for field in self.fallback_retriever.output_base_model.__fields__:
                retriever_output[field] = field
            self.fallback_retriever.output_mapping = retriever_output
        else:
            self.fallback_retriever = None

        # 4. Finalize initialization by calling the parent constructor
        super().__init__(memory=memory, llm=self.llms["default"])

    def _instantiate_prompt_template(
        self,
        analytics_worker_short_tool_description: str,
        analytics_worker_long_tool_description: str,
        plot_worker_short_tool_description: str,
        plot_worker_long_tool_description: str,
    ):
        # ... (This method remains unchanged)
        planner_prompt = instantiate_planner_prompt_by_use_case_jinja(
            use_case_name=self.config.use_case_name,
            use_case_description=self.config.use_case_description,
            analytics_worker_tool_description=analytics_worker_short_tool_description,
            plot_worker_tool_description=plot_worker_short_tool_description,
        )
        replanner_prompt = instantiate_replanner_prompt_by_use_case_jinja(
            use_case_name=self.config.use_case_name,
            use_case_description=self.config.use_case_description,
            analytics_worker_tool_description=analytics_worker_short_tool_description,
            plot_worker_tool_description=plot_worker_short_tool_description,
        )
        sql_generation_prompt = (
            instantiate_sql_generator_prompt_by_use_case_jinja(
                dialect=self.config.dialect.value,
                functions=self.config.dialect.functions,
            )
        )
        sql_validation_prompt = instantiate_sql_validation_prompt_jinja()
        analytics_prompt = instantiate_analytics_worker_prompt_by_use_case_jinja(
            use_case_name=self.config.use_case_name,
            use_case_description=self.config.use_case_description,
            analytics_worker_tool_description=analytics_worker_long_tool_description,
        )
        plot_prompt = instantiate_plot_worker_prompt_by_use_case_jinja(
            use_case_name=self.config.use_case_name,
            use_case_description=self.config.use_case_description,
            plot_worker_tool_description=plot_worker_long_tool_description,
        )
        summary_prompt = instantiate_summarization_prompt_by_use_case_jinja(
            use_case_name=self.config.use_case_name,
            use_case_description=self.config.use_case_description,
            analytics_worker_tool_description=analytics_worker_short_tool_description,
            plot_worker_tool_description=plot_worker_short_tool_description,
        )
        return dict(
            planner_prompt=planner_prompt,
            replanner_prompt=replanner_prompt,
            sql_generation_prompt=sql_generation_prompt,
            sql_validation_prompt=sql_validation_prompt,
            analytics_prompt=analytics_prompt,
            plot_prompt=plot_prompt,
            summary_prompt=summary_prompt,
        )

    def build_planner(self, memory: Memory, llm: BaseLLM) -> Planner:
        config = PlannerConfig(
            name="planner", **self.processed_prompts["planner_prompt"]
        )
        planner = Planner(memory=memory, llm=llm, config=config)
        planner.set_input_mapping(
            dict(
                query="query",
                rule="planner_rule",
                schema="planner_schema",
                history="history",
            )
        )
        planner.output_mapping = dict(
            plan="plan",
            final_response="final_response",
            llm_output="llm_output",
        )
        return planner

    def build_replanner(self, memory: Memory, llm: BaseLLM) -> Replanner:
        config = ReplannerConfig(
            name="replanner", **self.processed_prompts["replanner_prompt"]
        )
        replanner = Replanner(memory=memory, llm=llm, config=config)
        replanner.set_input_mapping(
            dict(
                query="query",
                history="history",
                plan="plan",
                worker_response="worker_response",
                rule="replanner_rule",
                schema="replanner_schema",
            )
        )
        replanner.output_mapping = dict(
            plan="plan",
            final_response="final_response",
            llm_output="llm_output",
        )
        return replanner

    def build_retrieval_worker(
        self, memory: Memory, llm: BaseLLM
    ) -> RetrievalWorker:
        config = RetrievalWorkerConfig(
            name="retrieval_worker",
            sql_generation_prompt=self.processed_prompts[
                "sql_generation_prompt"
            ],
            sql_validation_prompt=self.processed_prompts[
                "sql_validation_prompt"
            ],
        )
        worker = RetrievalWorker(
            memory=memory,
            llm=llm,
            config=config,
            sql_executor=self.sql_executor,
            fallback_schema_retriever=self.fallback_retriever,
        )
        worker.set_input_mapping(
            dict(
                plan="plan",
                rule="retrieval_worker_rule",
                schema="retrieval_worker_schema",
                example="retrieval_worker_example",
                schema_fallback_level="schema_fallback_level",
            )
        )
        worker.output_mapping = dict(
            worker_response="worker_response",
            retrieval_worker_state="retrieval_worker_state",
        )
        return worker

    def build_analytics_worker(
        self, memory: Memory, llm: BaseLLM
    ) -> AnalyticsWorker:
        config = AnalyticsWorkerConfig(
            name="analytics_worker",
            prompt=self.processed_prompts["analytics_prompt"],
            max_recursion=self.config.max_react_recursion,
        )
        worker = AnalyticsWorker(memory=memory, llm=llm, config=config)
        worker.set_input_mapping(
            dict(
                plan="plan",
                worker_response="worker_response",
                rule="analytics_worker_rule",
            )
        )
        worker.output_mapping = dict(
            worker_response="worker_response",
            analytics_worker_state="analytics_worker_state",
        )
        return worker

    def build_plot_worker(self, memory: Memory, llm: BaseLLM) -> PlotWorker:
        config = PlotWorkerConfig(
            name="plot_worker",
            prompt=self.processed_prompts["plot_prompt"],
            max_recursion=self.config.max_react_recursion,
        )
        worker = PlotWorker(memory=memory, llm=llm, config=config)
        worker.set_input_mapping(
            dict(
                plan="plan",
                worker_response="worker_response",
                rule="plot_worker_rule",
            )
        )
        worker.output_mapping = dict(
            worker_response="worker_response",
            plot_worker_state="plot_worker_state",
        )
        return worker

    def build_plan_condition_function(self) -> Coroutine:
        config = PlanConditionalEdgeConfig(name="plan_condition")
        plan_condition = PlanConditionalEdge(config=config)
        plan_condition.set_input_mapping(
            dict(final_response="final_response", plan="plan")
        )
        return plan_condition.get_function()

    def build_workflow(self, memory: Memory, llm: BaseLLM):
        # use component-specific LLMs if available
        self.planner = self.build_planner(memory, self.llms.get("planner", llm))
        self.replanner = self.build_replanner(
            memory, self.llms.get("replanner", llm)
        )
        self.retrieval_worker = self.build_retrieval_worker(
            memory, self.llms.get("retrieval_worker", llm)
        )
        self.analytics_worker = self.build_analytics_worker(
            memory, self.llms.get("analytics_worker", llm)
        )
        self.plot_worker = self.build_plot_worker(
            memory, self.llms.get("plot_worker", llm)
        )
        self.plan_condition_function = self.build_plan_condition_function()

        workflow = StateGraph(CWDState)

        workflow.add_node("retriever", self.retriever)
        workflow.add_node("planner", self.planner)
        workflow.add_node("replanner", self.replanner)
        workflow.add_node("retrieval_worker", self.retrieval_worker)
        workflow.add_node("analytics_worker", self.analytics_worker)
        workflow.add_node("plot_worker", self.plot_worker)

        workflow.add_edge(START, "retriever")
        workflow.add_edge("retriever", "planner")
        workflow.add_edge("retrieval_worker", "replanner")
        workflow.add_edge("analytics_worker", "replanner")
        workflow.add_edge("plot_worker", "replanner")
        workflow.add_conditional_edges("planner", self.plan_condition_function)
        workflow.add_conditional_edges(
            "replanner", self.plan_condition_function
        )

        return workflow.compile()

    async def __call__(
        self,
        state: CWDState,
        config: RunnableConfig,
        streaming: bool = False,
        summarize: bool = False,
    ) -> Generator[
        Union[
            Tuple[CWDState, List[Dict]],  # final CWDState, a list of events
            Tuple[str, str],  # node_name, streaming message
        ],
        None,
        None,
    ]:
        """
        Run CWDAgent, return a generator of streaming message and CWDState

        - If streaming = True, first return a list of (node_name, streaming_message), then return (final CWDState, the list of events)

        - If streaming = False, return (final CWDState, the list of events) directly.
        """
        all_events = []

        async def generate_summary() -> str:
            llm = self.llms["default"]
            prompt = build_prompt(self.processed_prompts["summary_prompt"])

            trajectory = []
            for i, response in enumerate(state.worker_response.task_response):
                trajectory.append(
                    f"Step {i + 1}\nTask: {response.task_description}\nResponse: {response.response}"
                )
            messages = prompt.invoke(
                input=dict(
                    query=state.query,
                    trajectory="\n".join(trajectory),
                )
            )

            api_key = config.get(CONFIGURABLE, {}).get(API_KEY, "")
            base_url = config.get(CONFIGURABLE, {}).get(BASE_URL, "")
            token = config.get(CONFIGURABLE, {}).get(TOKEN, "")

            output = await llm.ainvoke(
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                token=token,
                from_comoponent="summarization",
                with_structured_output=Summary,
            )
            if isinstance(output.generation, Summary):
                return output.generation.summary
            return ""

        async def stream():
            try:
                if config[CONFIGURABLE].get(DEBUG, False):
                    agent_response_parser = AgentResponseParser(
                        [], self.memory, config
                    )
                async for event in self.workflow.astream(
                    state,
                    config=config,
                    stream_mode="updates",
                    subgraphs=True,
                ):
                    all_events.append(event)
                    for node_name, node_response in event[1].items():
                        for field, value in node_response.items():
                            if hasattr(state, field):
                                state.update_field(field, value)
                        if hasattr(StatusMessage, node_name):
                            yield (
                                node_name,
                                getattr(StatusMessage, node_name).value,
                            )
                    if config[CONFIGURABLE].get(DEBUG, False):
                        formatted_event = (
                            agent_response_parser.process_event_step(
                                event, len(all_events), "text"
                            )
                        )
                        logger.info(formatted_event)
                if summarize and state.worker_response.task_response:
                    state.summary = await generate_summary()
            except asyncio.CancelledError:
                logger.exception("Agent streaming is cancelled.")
                raise
            finally:
                logger.info("Finish agent streaming.")

        thread_id = config.get(CONFIGURABLE, {}).get(THREAD_ID, "")
        question_id = config.get(CONFIGURABLE, {}).get(QUESTION_ID, "")
        try:
            timeout = self.config.timeout
            start_time = time.monotonic()
            logger.info(
                f"Conversation ID: {thread_id}, question ID: {question_id} - Start to run agent graph."
            )
            async with asyncio.timeout(timeout):
                async for name, message in stream():
                    logger.info(f"{name}: {message}")
                    if streaming:
                        yield name, message

            state.total_time = time.monotonic() - start_time
            logger.info(
                f"Conversation ID: {thread_id}, question ID: {question_id} - Finished running agent graph in {round(state.total_time, 2)} seconds."
            )
            yield state, all_events
        except asyncio.TimeoutError as e:
            # TODO: better handle intermediate results during timeout
            state.final_response = Response(
                response="Reach time limit for running CWD Agent, No final response generated.",
                output_df_name=[],
                output_img_name=[],
            )
            state.error = repr(e)
            yield state, all_events
        except Exception as e:
            call_stack = traceback.format_exc()
            state.final_response = Response(
                response="Failed to generate final response.",
                output_df_name=[],
                output_img_name=[],
            )
            state.error = f"{repr(e)}\n{call_stack}"
            logger.error(
                f"Conversation ID: {thread_id}, question ID: {question_id}, Error: agent failed with error:\n{state.error}"
            )
            yield state, all_events
