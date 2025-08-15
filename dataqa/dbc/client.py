import asyncio
import uuid
from typing import Callable, Dict
import pandas as pd
import yaml

from dataqa.agent.cwd_agent.cwd_agent import CWDAgent, CWDState, CwdAgentDefinitionConfig
from dataqa.dbc.llm import DBCLLMAdapter
from dataqa.dbc.models import DBCRequest, DBCResponse, StepResponse
from dataqa.dbc.sql_executor import DBCSQLExecutor
from dataqa.dbc.storage import DBCDataSource
from dataqa.memory import Memory
from dataqa.utils.agent_util import AgentResponseParser
from dataqa.utils.langgraph_utils import CONFIGURABLE, THREAD_ID
from dataqa.components.resource_manager.resource_manager import ResourceManager


class DBCClient:
    """
    Main client interface for DBC service integration.
    """
    def __init__(
        self,
        llm_callable: Callable,
        s3_callable: Callable,
        sql_callable: Callable,
        agent_config: Dict,
        asset_s3_prefix: str,
        data_s3_prefix: str,
    ):
        self.llm_callable = llm_callable
        self.s3_callable = s3_callable
        self.sql_callable = sql_callable
        self.agent_config_dict = agent_config
        self.asset_s3_prefix = asset_s3_prefix
        self.data_s3_prefix = data_s3_prefix

    async def process_query(self, request: DBCRequest) -> DBCResponse:
        """
        Main entry point to process a query from the DBC service.
        """
        memory = Memory()
        config = {CONFIGURABLE: {THREAD_ID: request.conversation_id}}

        # Load historical data into memory
        for turn in request.conversation_history:
            for i, df_path in enumerate(turn.output_df_names):
                try:
                    df_bytes = self.s3_callable(df_path, mode='r')
                    df = pd.read_parquet(df_bytes)
                    logical_name = f"history_df_{turn.query[:10]}_{i}"
                    memory.put_dataframe(logical_name, df, config)
                except Exception as e:
                    print(f"Warning: Could not load dataframe from {df_path}. Error: {e}")
        
        # Instantiate and configure the agent with DBC adapters
        agent = self._create_agent_with_dbc_services(memory)

        # Run the agent
        initial_state = CWDState(query=request.user_query)
        final_state, events = await agent(initial_state, config=config)
        
        # Process and persist final outputs
        final_response_obj = final_state.get("final_response")
        df_s3_paths, img_s3_paths = [], []
        text_response = "An error occurred, and no final response was generated."

        if final_response_obj:
            text_response = final_response_obj.response
            
            for name in final_response_obj.output_df_name:
                df = memory.get_dataframe(name, config)
                if df is not None:
                    s3_path = f"{self.data_s3_prefix.rstrip('/')}/dataframes/{name}-{uuid.uuid4()}.parquet"
                    df_bytes = df.to_parquet(index=False)
                    self.s3_callable(s3_path, mode='w', content=df_bytes)
                    df_s3_paths.append(s3_path)

            for name in final_response_obj.output_img_name:
                image_bytes = memory.get_image(name, config)
                if image_bytes:
                    s3_path = f"{self.data_s3_prefix.rstrip('/')}/images/{name}-{uuid.uuid4()}.png"
                    self.s3_callable(s3_path, mode='w', content=image_bytes)
                    img_s3_paths.append(s3_path)

        # Format the final DBCResponse
        parser = AgentResponseParser(events, memory, config)
        steps = [StepResponse(name=f"Step {i+1}", content=s) for i, s in enumerate(parser.formatted_events)]

        return DBCResponse(
            text=text_response,
            output_df_names=df_s3_paths,
            output_image_names=img_s3_paths,
            steps=steps
        )
        
    def _create_agent_with_dbc_services(self, memory: Memory) -> CWDAgent:
        """
        Factory method to assemble the CWDAgent with DBC-specific components.
        """
        agent_def_config = CwdAgentDefinitionConfig(**self.agent_config_dict)

        # 1. Create LLM Adapters
        dbc_llm_adapter = DBCLLMAdapter(self.llm_callable)
        llm_adapters = {name: dbc_llm_adapter for name in CWDAgent.components}

        # 2. Create Data Source and Resource Manager
        data_source = DBCDataSource(self.s3_callable, self.asset_s3_prefix)
        resource_manager = ResourceManager(data_source=data_source)
        
        # 3. Create SQL Executor Adapter
        sql_executor = DBCSQLExecutor(self.sql_callable, config={})
        
        # 4. Instantiate CWDAgent, injecting all DBC components
        agent = CWDAgent(
            memory=memory, 
            config=agent_def_config, 
            llms=llm_adapters,
            resource_manager=resource_manager,
            sql_executor=sql_executor
        )
        
        return agent