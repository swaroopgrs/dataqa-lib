# dataqa/integrations/dbc/client.py
from typing import Callable

from dataqa.core.agent.cwd_agent.cwd_agent import CWDState
from dataqa.integrations.dbc.factory import DBC_CWDAgentFactory
from dataqa.integrations.dbc.models import (
    DBCRequest,
    DBCResponse,
    FileType,
    StepResponse,
    UsecaseConfig,
)


class DBCClient:
    """
    Client for DBC service integration.
    """

    def __init__(
        self,
        usecase_config: UsecaseConfig,
        request: DBCRequest,
        llm_callable: Callable,
        asset_callable: Callable,
        sql_callable: Callable,
        storage_callable: Callable,
    ):
        self.usecase_config = usecase_config
        self.request = request
        self.llm_callable = llm_callable
        self.asset_callable = asset_callable
        self.sql_callable = sql_callable
        self.storage_callable = storage_callable

    async def process_query(self) -> DBCResponse:
        """
        Main entry point to process a query from the DBC service.
        """
        from dataqa.core.data_models.asset_models import (
            IngestionData as CoreIngestionData,
        )
        from dataqa.core.memory import Memory
        from dataqa.core.utils.langgraph_utils import CONFIGURABLE, THREAD_ID

        memory = Memory()
        # Use a unique ID for the runnable config to avoid state collision
        runnable_config = {
            CONFIGURABLE: {
                THREAD_ID: self.request.conversation_id
                + self.request.question_id
            }
        }

        # 1. Prepare conversation history for the agent
        history = [
            turn.output_text for turn in self.request.conversation_history
        ]

        # 2. Fetch all necessary assets for the use case
        dbc_ingestion_data = self.asset_callable(
            config_id=self.usecase_config.config_id,
            file_types={FileType.RULES, FileType.SCHEMA, FileType.EXAMPLES},
        )

        # 3. Translate DBC model to core library model
        core_ingestion_data = CoreIngestionData.model_validate(
            dbc_ingestion_data.model_dump()
        )

        # 4. Create the agent instance using the DBC factory
        agent = DBC_CWDAgentFactory.create_agent(
            usecase_config=self.usecase_config,
            ingestion_data=core_ingestion_data,
            memory=memory,
            llm_callable=self.llm_callable,
            sql_callable=self.sql_callable,
        )

        # 5. Run the agent
        initial_state = CWDState(query=self.request.user_query, history=history)
        final_state, events = await agent(initial_state, runnable_config)

        # 6. Process and persist final outputs
        from dataqa.core.utils.agent_util import AgentResponseParser

        parser = AgentResponseParser(events, memory, runnable_config)

        final_response_obj = final_state.final_response
        df_s3_paths, img_s3_paths = [], []
        text_response = (
            "An error occurred, and no final response was generated."
        )

        if final_response_obj:
            text_response = final_response_obj.response

            for name in final_response_obj.output_df_name:
                df = memory.get_dataframe(name, runnable_config)
                if df is not None:
                    df_bytes = df.to_parquet(index=False)
                    s3_path = self.storage_callable(
                        data=df_bytes, path_suffix=f"dataframes/{name}.parquet"
                    )
                    df_s3_paths.append(s3_path)

            for name in final_response_obj.output_img_name:
                img_bytes, _ = memory.get_image_data(
                    name, runnable_config
                )  # get_image_data returns (bytes, df)
                if img_bytes:
                    s3_path = self.storage_callable(
                        data=img_bytes, path_suffix=f"images/{name}.png"
                    )
                    img_s3_paths.append(s3_path)

        steps = [
            StepResponse(name=f"Step {i + 1}", content=s)
            for i, s in enumerate(parser.formatted_events)
        ]

        return DBCResponse(
            text=text_response,
            output_df_names=df_s3_paths,
            output_image_names=img_s3_paths,
            steps=steps,
        )
