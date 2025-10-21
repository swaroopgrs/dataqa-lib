import asyncio
import time
from operator import add
from typing import Annotated, Coroutine, Dict, List, Tuple

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import START, StateGraph
from pydantic import Field

from dataqa.core.agent.base import Agent
from dataqa.core.agent.cwd_agent.config import (
    CwdAgentDefinitionConfig,
)
from dataqa.core.agent.cwd_agent.prompt import (
    instantiate_analytics_worker_prompt_by_use_case,
    instantiate_planner_prompt_by_use_case,
    instantiate_plot_worker_prompt_by_use_case,
    instantiate_replanner_prompt_by_use_case,
    instantiate_sql_generator_prompt_by_use_case,
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
from dataqa.core.utils.langgraph_utils import CONFIGURABLE, DEBUG
from dataqa.core.utils.utils import cls_from_str


class CWDState(PlanExecuteState):
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
        sql_executor: CodeExecutor,  # CORRECT: Use the abstract base class
    ):
        self.config = config
        self.llms = llms
        self.resource_manager = resource_manager
        self.sql_executor = sql_executor

        # Agent no longer builds its own dependencies. It receives them fully formed.

        # 1. Load tools and descriptions (standard setup)
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

        # 2. Instantiate prompt templates (standard setup)
        self.processed_prompts = self._instantiate_prompt_template(
            analytics_worker_short_tool_description=self.analytics_worker_short_tool_description,
            analytics_worker_long_tool_description=self.analytics_worker_long_tool_description,
            plot_worker_short_tool_description=self.plot_worker_short_tool_description,
            plot_worker_long_tool_description=self.plot_worker_long_tool_description,
        )

        # 3. Instantiate the retriever (standard setup)
        self.retriever = cls_from_str(config.retriever_config.type)(
            config=config.retriever_config.config,
            resource_manager=self.resource_manager,
        )
        self.retriever.set_input_mapping(dict(query="query"))
        retriever_output = {}
        for field in self.retriever.output_base_model.__fields__:
            retriever_output[field] = field
        self.retriever.output_mapping = retriever_output

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
        planner_prompt = instantiate_planner_prompt_by_use_case(
            use_case_name=self.config.use_case_name,
            use_case_description=self.config.use_case_description,
            analytics_worker_tool_description=analytics_worker_short_tool_description,
            plot_worker_tool_description=plot_worker_short_tool_description,
        )
        replanner_prompt = instantiate_replanner_prompt_by_use_case(
            use_case_name=self.config.use_case_name,
            use_case_description=self.config.use_case_description,
            analytics_worker_tool_description=analytics_worker_short_tool_description,
            plot_worker_tool_description=plot_worker_short_tool_description,
        )
        sql_generator_prompt = instantiate_sql_generator_prompt_by_use_case()
        analytics_prompt = instantiate_analytics_worker_prompt_by_use_case(
            use_case_name=self.config.use_case_name,
            use_case_description=self.config.use_case_description,
            analytics_worker_tool_description=analytics_worker_long_tool_description,
        )
        plot_prompt = instantiate_plot_worker_prompt_by_use_case(
            use_case_name=self.config.use_case_name,
            use_case_description=self.config.use_case_description,
            plot_worker_tool_description=plot_worker_long_tool_description,
        )
        return dict(
            planner_prompt=planner_prompt,
            replanner_prompt=replanner_prompt,
            sql_generator_prompt=sql_generator_prompt,
            analytics_prompt=analytics_prompt,
            plot_prompt=plot_prompt,
        )

    def build_planner(self, memory: Memory, llm: BaseLLM) -> Planner:
        config = PlannerConfig(
            name="planner", prompt=self.processed_prompts["planner_prompt"]
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
            name="replanner", prompt=self.processed_prompts["replanner_prompt"]
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
            sql_prompt=self.processed_prompts["sql_generator_prompt"],
            sql_execution_config=self.config.workers.retrieval_worker.sql_execution_config,
        )
        worker = RetrievalWorker(
            memory=memory,
            llm=llm,
            config=config,
            sql_executor=self.sql_executor,
        )
        worker.set_input_mapping(
            dict(
                plan="plan",
                rule="retrieval_worker_rule",
                schema="retrieval_worker_schema",
                example="retrieval_worker_example",
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
            name="plot_worker", prompt=self.processed_prompts["plot_prompt"]
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
        self, state: CWDState, config: RunnableConfig
    ) -> Tuple[CWDState, List[Dict]]:
        async def stream():
            all_events = []
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
                for _, v in event[1].items():
                    for k1, v1 in v.items():
                        if hasattr(state, k1):
                            state.update_field(k1, v1)
                if config[CONFIGURABLE].get(DEBUG, False):
                    formatted_event = agent_response_parser.process_event_step(
                        event, len(all_events), "text"
                    )
                    print(formatted_event)

            return state, all_events

        timeout = self.config.timeout
        start_time = time.monotonic()
        state, all_events = await asyncio.wait_for(stream(), timeout=timeout)
        state.total_time = time.monotonic() - start_time
        return state, all_events
