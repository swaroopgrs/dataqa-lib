# dataqa/integrations/local/factory.py
import os
from pathlib import Path
from typing import Dict
import yaml

from dataqa.core.agent.cwd_agent.builder import CWDAgentBuilder
from dataqa.core.agent.cwd_agent.cwd_agent import CWDAgent, CwdAgentDefinitionConfig
from dataqa.core.agent.cwd_agent.config import CwdAgentLLMReferences
from dataqa.core.components.code_executor.in_memory_code_executor import InMemoryCodeExecutor
from dataqa.core.components.resource_manager.resource_manager import ResourceManager
from dataqa.core.llm.base_llm import BaseLLM
from dataqa.core.memory import Memory
from dataqa.core.services.storage import LocalFileDataSource
from dataqa.core.utils.utils import cls_from_str

class LocalAgentFactory:
    """
    Factory to build a CWDAgent and its dependencies for a local execution environment.
    It reads all configuration from a single, self-contained agent configuration file.
    """
    @staticmethod
    def create_from_config(config_path: str, memory: Memory) -> CWDAgent:
        resolved_path = Path(config_path).resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Agent configuration file not found at {config_path}")

        config_dir = resolved_path.parent

        # Load and process the main agent.yml configuration
        with open(resolved_path, "r") as f:
            config_str_template = f.read()
            config_str = os.path.expandvars(config_str_template)
            # Resolve the <CONFIG_DIR> placeholder to make paths absolute
            config_str = config_str.replace("<CONFIG_DIR>", str(config_dir))
            raw_config = yaml.safe_load(config_str)

        # The raw_config now contains everything needed.
        agent_config = CwdAgentDefinitionConfig(**raw_config)

        # 1. Build LLMs
        llms: Dict[str, BaseLLM] = {}
        llm_configs_map = {}
        for name, llm_config in agent_config.llm_configs.items():
            llm_cls = cls_from_str(llm_config.type)
            llm_spec_config = llm_cls.config_base_model(**llm_config.config)
            llm_configs_map[name] = llm_cls(config=llm_spec_config)
        
        for component in CWDAgent.components:
            llm_name = agent_config.llm.get_component_llm_name(component)
            llms[component] = llm_configs_map[llm_name]

        # 2. Build ResourceManager
        asset_dir_str = agent_config.resource_manager_config.config.get("asset_directory")
        if not asset_dir_str:
            raise ValueError("`asset_directory` must be defined in `resource_manager_config`")
        local_data_source = LocalFileDataSource(asset_directory=asset_dir_str)
        resource_manager = ResourceManager(data_source=local_data_source)

        # 3. Build SQL Executor
        # The data_files paths have already been resolved by replacing <CONFIG_DIR>
        sql_exec_config = agent_config.workers.retrieval_worker.sql_execution_config
        sql_executor = InMemoryCodeExecutor(config=sql_exec_config)
        
        # 4. Use the generic builder to assemble the agent
        builder = CWDAgentBuilder(config=agent_config)
        agent = (builder
                 .with_memory(memory)
                 .with_llms(llms)
                 .with_resource_manager(resource_manager)
                 .with_sql_executor(sql_executor)
                 .build())
        
        return agent