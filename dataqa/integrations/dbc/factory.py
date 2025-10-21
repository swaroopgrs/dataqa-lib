# dataqa/integrations/dbc/factory.py
from typing import Callable
import yaml
from pathlib import Path

from dataqa.core.agent.cwd_agent.builder import CWDAgentBuilder
from dataqa.core.agent.cwd_agent.cwd_agent import CWDAgent, CwdAgentDefinitionConfig
from dataqa.core.components.resource_manager.resource_manager import ResourceManager
from dataqa.core.data_models.asset_models import IngestionData as CoreIngestionData
from dataqa.core.memory import Memory
from dataqa.integrations.dbc.llm import DBCLLMAdapter
from dataqa.integrations.dbc.sql_executor import DBCSQLExecutor
from dataqa.integrations.dbc.models import UsecaseConfig

class DBC_CWDAgentFactory:
    """
    Factory to create a CWDAgent instance for the DBC environment.
    """
    @staticmethod
    def create_agent(
        usecase_config: UsecaseConfig,
        ingestion_data: CoreIngestionData,
        memory: Memory,
        llm_callable: Callable,
        sql_callable: Callable,
    ) -> CWDAgent:
        """
        Builds the CWDAgent using DBC-provided callables and service adapters.
        """
        # 1. Load the base structural template for the agent.
        base_config_path = Path(__file__).parent / "agent_template.yml"
        if not base_config_path.exists():
            raise FileNotFoundError(f"Base agent structure template not found at {base_config_path}")
        
        base_config_dict = yaml.safe_load(open(base_config_path))
        
        # 2. Inject use case name and description into the config.
        base_config_dict["use_case_name"] = usecase_config.usecase_name
        base_config_dict["use_case_description"] = usecase_config.usecase_description
        agent_config = CwdAgentDefinitionConfig(**base_config_dict)

        # 3. Build adapters for DBC services.
        dbc_llm_adapter = DBCLLMAdapter(llm_callable)
        llms = {name: dbc_llm_adapter for name in CWDAgent.components}
        
        resource_manager = ResourceManager(ingestion_data=ingestion_data)
        
        sql_executor = DBCSQLExecutor(sql_callable, config_id=usecase_config.config_id, config={})
        
        # 4. Use the generic builder to assemble the agent.
        builder = CWDAgentBuilder(config=agent_config)
        agent = (builder
                 .with_memory(memory)
                 .with_llms(llms)
                 .with_resource_manager(resource_manager)
                 .with_sql_executor(sql_executor)
                 .build())
        
        return agent