import uuid
from collections.abc import AsyncIterable
from logging import getLogger
from typing import Any, Callable, Dict, List, Set, Union

import pandas as pd
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from dataqa.core.agent.cwd_agent.cwd_agent import CWDState
from dataqa.core.utils.dataframe_utils import df_to_markdown
from dataqa.integrations.dbc.factory import DBC_CWDAgentFactory
from dataqa.integrations.dbc.models import (
    DBCRequest,
    DBCResponse,
    FileType,
    IngestionData,
    StatusResponse,
    StepResponse,
    UsecaseConfig,
)

logger = getLogger(__name__)


class DBCClient:
    """
    Client for DBC service integration.
    """

    def __init__(
        self,
        usecase_config: UsecaseConfig,
        request: DBCRequest,
        llm_callable: Callable[[str, List[BaseMessage], Any], BaseMessage],
        s3_retrieval: Callable[[uuid.UUID, Set[FileType]], IngestionData],
        sql_callable: Callable[[str], Dict[str, Union[pd.DataFrame, str]]],
        storage_callable: Callable[[str, Union[pd.DataFrame, bytes]], None],
        os_retrieval: Callable[
            [str, uuid.UUID, Set[FileType], int],
            Union[IngestionData, List[str]],
        ],
        retrieve_previous_data: Callable[
            [str, uuid.UUID],  # name, question_id
            Union[None, pd.DataFrame],
        ],
    ):
        self.usecase_config = usecase_config
        self.request = request
        self.llm_callable = llm_callable
        self.s3_retrieval = s3_retrieval
        self.sql_callable = sql_callable
        self.storage_callable = storage_callable
        self.retrieve_previous_data = retrieve_previous_data
        self.os_retrieval = os_retrieval

    def construct_history(self, num_turns: int = 10) -> List[str]:
        """
        Construct conversation history from previous turns.

        Args:
            num_turns: Number of previous turns to include (default: 10)

        Returns:
            List of formatted history strings
        """
        history = []
        # Take the last num_turns from conversation history
        recent_turns = (
            self.request.conversation_history[-num_turns:]
            if self.request.conversation_history
            else []
        )

        for turn in recent_turns:
            # Format: user query + assistant response with dataframe names
            user_part = f"USER: {turn.query}"
            assistant_part = f"ASSISTANT: {turn.output_text}"

            # Add dataframe names if they exist
            if turn.output_dataframes:
                df_names = ", ".join(turn.output_dataframes)
                assistant_part += f" [DataFrames: {df_names}]"

            history_entry = f"{user_part}\n{assistant_part}\n---\n"
            history.append(history_entry)

        return history

    async def initialize_memory_from_history(
        self, num_turns: int = 1, config: RunnableConfig = {}
    ):
        """
        Initialize memory by loading dataframes from previous conversation turns.

        Args:
            num_turns: Number of previous turns to load dataframes from (default: 1)

        Returns:
            Memory object with loaded dataframes from conversation history
        """
        from dataqa.core.memory import Memory

        memory = Memory()

        # Get recent turns based on num_turns parameter
        recent_turns = (
            self.request.conversation_history[-num_turns:]
            if self.request.conversation_history
            else []
        )

        for turn in recent_turns:
            if turn.output_dataframes:
                for df_name in turn.output_dataframes:
                    try:
                        # Retrieve dataframe using name and question_id
                        df = await self.retrieve_previous_data(
                            df_name, turn.question_id
                        )
                        if df is not None:
                            memory.put_dataframe(df_name, df, config)
                            logger.info(
                                f"Loaded dataframe {df_name} from question {turn.question_id}"
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to load dataframe {df_name} from question {turn.question_id}: {repr(e)}"
                        )

        return memory

    async def process_query(
        self,
        streaming: bool = True,
        summarize: bool = True,
        num_history_turns: int = 10,
        num_memory_turns: int = 1,
    ) -> AsyncIterable[Union[StatusResponse, DBCResponse]]:
        """
        Main entry point to process a query from the DBC service.

        Args:
            streaming: Whether to stream intermediate responses
            summarize: Whether to include summary in final response
            num_history_turns: Number of previous turns to include in history
            num_memory_turns: Number of previous turns to load dataframes from
        """
        from dataqa.core.data_models.asset_models import (
            IngestionData as CoreIngestionData,
        )
        from dataqa.core.utils.langgraph_utils import (
            CONFIGURABLE,
            QUESTION_ID,
            THREAD_ID,
        )

        # Use a unique ID for the runnable config to avoid state collision
        runnable_config = {
            CONFIGURABLE: {
                THREAD_ID: self.request.conversation_id,
                QUESTION_ID: self.request.question_id,
            }
        }

        # Load memory with previous dataframes
        memory = await self.initialize_memory_from_history(
            num_memory_turns, runnable_config
        )

        # 1. Prepare conversation history for the agent
        history = self.construct_history(num_history_turns)

        # 2. Fetch all necessary assets for the use case
        dbc_ingestion_data = await self.s3_retrieval(
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

        final_state = None
        async for chunk in agent(
            state=initial_state,
            config=runnable_config,
            streaming=streaming,
            summarize=summarize,
        ):
            if isinstance(chunk[0], CWDState):
                final_state, _ = chunk
            elif streaming:
                yield StatusResponse(name=chunk[0], message=chunk[1])

        # 6. Process and persist final outputs
        final_response_obj = final_state.final_response if final_state else None
        df_s3_names, img_s3_names = [], []
        text_response = (
            "An error occurred, and no final response was generated."
        )

        if final_response_obj:
            text_response = final_response_obj.response

            for name in final_response_obj.output_df_name:
                df = memory.get_dataframe(name, runnable_config)
                if df is not None:
                    try:
                        await self.storage_callable(name=name, data=df)
                        df_s3_names.append(name)
                        logger.info(
                            f"Store dataframe {name} to s3. Question ID {self.request.question_id} Conversation ID {self.request.conversation_id}."
                        )
                        text_response += f"\n\nPlease check the table {name} below:\n{df_to_markdown(df=df, fold=True)}"
                    except Exception as e:
                        logger.error(
                            f"Failed to save dataframe {name} to s3 due to {repr(e)}. Question ID {self.request.question_id} Conversation ID {self.request.conversation_id}."
                        )

            for name in final_response_obj.output_img_name:
                img_bytes = memory.get_image(name, runnable_config)
                if img_bytes:
                    try:
                        await self.storage_callable(name=name, data=img_bytes)
                        img_s3_names.append(name)
                        logger.info(
                            f"Store image {name} to s3. Question ID {self.request.question_id} Conversation ID {self.request.conversation_id}."
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to save image {name} to s3 due to {repr(e)}. Question ID {self.request.question_id} Conversation ID {self.request.conversation_id}."
                        )

            if summarize and final_state and final_state.summary:
                text_response += f"\n\n---\n\nHere is a brief summary of how your question was being handled:\n\n{final_state.summary}"

        steps = []
        if (
            final_state
            and final_state.worker_response
            and final_state.worker_response.task_response
        ):
            steps = [
                StepResponse(name=f"Step {i + 1}", content=s.response)
                for i, s in enumerate(final_state.worker_response.task_response)
            ]

        yield DBCResponse(
            text=text_response,
            output_df_names=df_s3_names,
            output_image_names=img_s3_names,
            steps=steps,
        )
