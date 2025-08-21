from typing import Dict

from dataqa.core.agent.cwd_agent.cwd_agent import CWDAgent, CwdAgentDefinitionConfig
from dataqa.core.components.code_executor.base_code_executor import CodeExecutor
from dataqa.core.components.resource_manager.resource_manager import ResourceManager
from dataqa.core.llm.base_llm import BaseLLM
from dataqa.core.memory import Memory

class CWDAgentBuilder:
    """
    A generic builder for the CWDAgent.
    It takes fully constructed dependencies and injects them into the agent.
    It has no knowledge of "local" or "dbc" modes.
    """
    def __init__(self, config: CwdAgentDefinitionConfig):
        self.config = config
        self._memory: Memory = None
        self._llms: Dict[str, BaseLLM] = None
        self._resource_manager: ResourceManager = None
        self._sql_executor: CodeExecutor = None

    def with_memory(self, memory: Memory) -> "CWDAgentBuilder":
        self._memory = memory
        return self

    def with_llms(self, llms: Dict[str, BaseLLM]) -> "CWDAgentBuilder":
        self._llms = llms
        return self

    def with_resource_manager(self, resource_manager: ResourceManager) -> "CWDAgentBuilder":
        self._resource_manager = resource_manager
        return self

    def with_sql_executor(self, sql_executor: CodeExecutor) -> "CWDAgentBuilder":
        self._sql_executor = sql_executor
        return self

    def build(self) -> CWDAgent:
        """Constructs the CWDAgent with the provided components."""
        if not all([self._memory, self._llms, self._resource_manager, self._sql_executor]):
            raise ValueError("All dependencies (memory, llms, resource_manager, sql_executor) must be provided.")

        return CWDAgent(
            memory=self._memory,
            config=self.config,
            llms=self._llms,
            resource_manager=self._resource_manager,
            sql_executor=self._sql_executor,
        )