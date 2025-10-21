import os
from typing import List

import pandas as pd

from dataqa.core.agent.cwd_agent.cwd_agent import CWDAgent, CWDState
from dataqa.core.client import CoreRequest, CoreResponse, CoreStep, DataQAClient
from dataqa.core.memory import Memory
from dataqa.core.utils.agent_util import AgentResponseParser
from dataqa.core.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    CONFIGURABLE,
    THREAD_ID,
)
from dataqa.integrations.local.factory import LocalAgentFactory


class LocalClient(DataQAClient):
    """
    The default client for local development and usage of the dataqa library.
    It operates on a local project configuration file and its associated assets.
    """

    def __init__(self, config_path: str):
        """
        Initializes the client with a path to a CWD Agent configuration file.
        This file is the single entry point for a local project setup.

        Args:
            config_path: The path to the CWD Agent's main YAML configuration file.
        """
        self.config_path = config_path
        self._agent: CWDAgent = None

    def _get_or_create_agent(self, memory: Memory) -> CWDAgent:
        # Agent is created on-demand, which is efficient.
        if self._agent is None:
            self._agent = LocalAgentFactory.create_from_config(
                self.config_path, memory
            )
        return self._agent

    async def process_query(self, request: CoreRequest) -> CoreResponse:
        """
        Processes a query using the agent configured in the local project.
        """
        memory = Memory()
        runnable_config = {
            CONFIGURABLE: {
                THREAD_ID: request.conversation_id,
                # For local mode, we assume credentials are in env vars
                API_KEY: os.environ.get("AZURE_OPENAI_API_KEY", ""),
                BASE_URL: os.environ.get("OPENAI_API_BASE", ""),
            }
        }

        agent = self._get_or_create_agent(memory)

        history_texts = [turn.output_text for turn in request.history]
        initial_state = CWDState(
            query=request.user_query, history=history_texts
        )

        final_state, events = await agent(initial_state, runnable_config)

        # Process the final state into a CoreResponse
        final_response_obj = final_state.final_response
        output_dfs: List[pd.DataFrame] = []
        output_imgs: List[bytes] = []
        text_response = (
            "An error occurred, and no final response was generated."
        )

        if final_response_obj:
            text_response = final_response_obj.response
            for name in final_response_obj.output_df_name:
                df = memory.get_dataframe(name, runnable_config)
                if df is not None:
                    output_dfs.append(df)

            for name in final_response_obj.output_img_name:
                img_bytes, _ = memory.get_image_data(name, runnable_config)
                if img_bytes:
                    output_imgs.append(img_bytes)

        parser = AgentResponseParser(events, memory, runnable_config)
        steps = [
            CoreStep(name=f"Step {i + 1}", content=s)
            for i, s in enumerate(parser.formatted_events)
        ]

        return CoreResponse(
            text=text_response,
            output_dataframes=output_dfs,
            output_images=output_imgs,
            steps=steps,
        )
