import os
from typing import Generator, List, Union

import pandas as pd

from dataqa.core.agent.cwd_agent.cwd_agent import CWDAgent, CWDState
from dataqa.core.client import (
    CoreRequest,
    CoreResponse,
    CoreStatus,
    CoreStep,
    DataQAClient,
)
from dataqa.core.memory import Memory
from dataqa.core.utils.agent_util import AgentResponseParser
from dataqa.core.utils.langgraph_utils import (
    API_KEY,
    BASE_URL,
    CONFIGURABLE,
    PROMPT_BACK,
    QUESTION_ID,
    THREAD_ID,
    TOKEN,
)
from dataqa.integrations.local.factory import LocalAgentFactory
from dataqa.scripts.azure_token import get_az_token_using_cert


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

    def get_streaming_message(self) -> CoreStatus:
        pass

    async def process_query(
        self,
        request: CoreRequest,
        streaming: bool = False,
        summarize: bool = False,
        prompt_back: bool = True,
    ) -> Generator[Union[CoreStatus, CoreResponse], None, None]:
        """
        Processes a query using the agent configured in the local project.

        Run in the streaming mode if `streaming` = True.
        """
        memory = Memory()

        if os.environ.get("CERT_PATH"):
            # print(f"Initializing LLM using CERT_PATH: {os.environ.get('CERT_PATH')}")
            token = ""
            api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
            if api_key == "":
                print("Running Standard LLM Azure API Subscription........")
                api_key = get_az_token_using_cert()[0]
            else:
                print("Running Multi-Tenant LLM Azure API Subscription........")
                token = get_az_token_using_cert()[0]

            runnable_config = {
                CONFIGURABLE: {
                    THREAD_ID: request.conversation_id,
                    QUESTION_ID: request.question_id,
                    # For local mode, we assume credentials are in env vars
                    API_KEY: api_key,
                    BASE_URL: os.environ.get("OPENAI_API_BASE", ""),
                    TOKEN: token,
                    PROMPT_BACK: prompt_back,
                }
            }
        else:
            runnable_config = {
                CONFIGURABLE: {
                    THREAD_ID: request.conversation_id,
                    QUESTION_ID: request.question_id,
                    # For local mode, we assume credentials are in env vars
                    API_KEY: os.environ.get("AZURE_OPENAI_API_KEY", ""),
                    BASE_URL: os.environ.get("OPENAI_API_BASE", ""),
                    TOKEN: os.environ.get("AZURE_OPENAI_API_TOKEN", ""),
                    PROMPT_BACK: prompt_back,
                }
            }

        agent = self._get_or_create_agent(memory)

        history_texts = [turn.output_text for turn in request.history]
        initial_state = CWDState(
            query=request.user_query, history=history_texts
        )

        async for chunk in agent(
            state=initial_state,
            config=runnable_config,
            streaming=streaming,
            summarize=summarize,
        ):
            if isinstance(chunk[0], CWDState):
                final_state, events = chunk
            elif streaming:
                yield CoreStatus(name=chunk[0], message=chunk[1])

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
                img_bytes = memory.get_image(name, runnable_config)
                if img_bytes:
                    output_imgs.append(img_bytes)

        parser = AgentResponseParser(events, memory, runnable_config)
        # print("\n" + "=" * 20 + " DEBUG INFO " + "=" * 20)
        # print("### List of dataframes in memory")
        # for df_name, val in memory.get_dataframes(runnable_config).items():
        #     print(f"df_name: {df_name}\nSQL: {val[1]}\n")
        # parser.pretty_print_output()
        # from dataqa.core.agent.cwd_agent.error_message import agent_error_log
        # print(f"### Total number of agent errors: {len(agent_error_log)}\nList of error logs:\n{agent_error_log}\n")
        steps = [
            CoreStep(name=f"Step {i + 1}", content=s)
            for i, s in enumerate(parser.formatted_events)
        ]

        yield CoreResponse(
            text=text_response,
            output_dataframes=output_dfs,
            output_images=output_imgs,
            steps=steps,
        )
