import os

import yaml


class Pipeline:
    pipeline_name: str
    workflow: CompiledStateGraph
    state_base_model: Type[BaseModel]

    def __init__(
        self,
        pipeline_name: str,
        workflow: CompiledStateGraph,
        state_base_model: Type[BaseModel],
    ):
        self.pipeline_name = pipeline_name
        self.workflow = workflow
        self.state_base_model = state

    async def retrieve_rewritten_query(
        self,
        conversation_id,
    ):
        if self.workflow.checkpointer is None:
            return "None"
        previous_state = await self.workflow.checkpointer.aget(conversation_id)
        if previous_state is None:
            return "None"
        try:
            if "return_output" in previous_state["channel_values"]:
                return previous_state["channel_values"][
                    "return_output"
                ].rewritten_query
        except Exception:
            return "None"

    async def update_state(self, state, conversation_id):
        if self.workflow.checkpointer is not None:
            await self.workflow.aupdate_state(
                conversation_id, state.model_dump()
            )

    def prepare_output(self, state: PipelineOutput) -> CWDResponse:
        if state.rewritten_query:
            pass

    async def run(self, query: str, conversation_id=None):
        previous_rewritten_query = await self.retrieve_rewritten_query(
            conversation_id
        )
        state = self.state_base_model(
            input=PipelineInput(
                query=query,
                previous_rewritten_query=previous_rewritten_query,
            )
        )
        await self.update_state(state, conversation_id)
        async for event in self.workflow.astream(
            state, config, stream_mode="updates"
        ):
            for event_name, event_output in event.items():
                for k, v in event_output.items():
                    setattr(state, k, v)
                    if k == "error":
                        raise Exception(v.error_message)
        response = self.prepare_output(state.return_outptut)
        return response


base_dir = os.environ.get("BASE_DIR")
config_path = os.path.join(base_dir, "examples/payments/config/config.yaml")
pipeline_config = open(config_path).read().format(BASE_DIR=base_dir)
pipeline_config = yaml.safe_load(pipeline_config)


pipeline_schema = PipelineConfig(**pipeline_config)
workflow, state_base_model = build_from_config(pipeline_schema)

pipeline = Pipeline(
    pipeline_name=config["pipeline_name"],
    workflow=workflow,
    state_base_model=state_base_model,
)
response = await pipeline.run(query, previous_rewritten_query)
