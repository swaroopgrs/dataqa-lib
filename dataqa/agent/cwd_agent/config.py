from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class PromptMessageConfig(BaseModel):
    role: str = Field(default="system", description="Role of the message")
    content: str = Field(
        description="Content of the message. Can use {placeholders} and <schema>."
    )


CwdAgentPromptValue = Union[str, List[PromptMessageConfig]]


class CwdAgentPromptsConfig(BaseModel):
    planner_prompt: CwdAgentPromptValue
    replanner_prompt: CwdAgentPromptValue
    sql_generator_prompt: CwdAgentPromptValue
    analytics_prompt: CwdAgentPromptValue
    plot_prompt: CwdAgentPromptValue


class InMemorySqlExecutorConfig(BaseModel):
    data_files: Any = Field(
        description="List of data files to load into the in-memory SQL database."
    )
    backend: str = Field(
        default="duckdb",
    )


class RetrievalWorkerConfig(BaseModel):
    sql_execution_config: InMemorySqlExecutorConfig


class AnalyticsWorkerConfig(BaseModel):
    pass


class PlotWorkerConfig(BaseModel):
    pass


class CwdAgentWorkersModulesConfig(BaseModel):
    retrieval_worker: RetrievalWorkerConfig
    analytics_worker: Optional[AnalyticsWorkerConfig] = Field(
        default_factory=AnalyticsWorkerConfig
    )
    plot_worker: Optional[PlotWorkerConfig] = Field(
        default_factory=PlotWorkerConfig
    )


class LLMSelectionConfig(BaseModel):
    type: str = Field(
        description="Fully qualified class name for the LLM (e.g., 'dataqa.llm.openai.AzureOpenAI')."
    )
    config: Dict[str, Any] = Field(
        description="Configuration dictionary for the chosen LLM type (e.g., model, api_key, base_url)."
    )


class ResourceManagerConfig(BaseModel):
    type: str
    config: Dict[str, Any]


class RetrieverSelectionConfig(BaseModel):
    type: str
    config: Dict[str, Any]


class CwdAgentPromptTemplateConfig(BaseModel):
    use_case_name: str
    use_case_description: str
    use_case_schema: str  # For now we consider SQL-based use case only. schema may be empty for API-based use cases.
    use_case_sql_example: str  # we require at least one SQL example TODO build an example BaseModel
    use_case_planner_instruction: str = ""
    use_case_replanner_instruction: str = ""
    use_case_sql_instruction: str = ""
    use_case_analytics_worker_instruction: str = ""
    use_case_plot_worker_instruction: str = ""


class CwdAgentLLMReferences(BaseModel):
    """References to LLM configurations defined in llm_configs."""

    default: str
    planner: Optional[str] = None
    replanner: Optional[str] = None
    retrieval_worker: Optional[str] = None
    analytics_worker: Optional[str] = None
    plot_worker: Optional[str] = None

    def get_component_llm_name(self, component_name: str) -> str:
        """
        Get the LLM name for a specific component.
        Falls back to default if the component-specific name is not provided.
        """
        if (
            hasattr(self, component_name)
            and getattr(self, component_name) is not None
        ):
            return getattr(self, component_name)
        return self.default


class CwdAgentRetrieverReferences(BaseModel):
    """References to LLM configurations defined in llm_configs."""

    default: str
    planner: Optional[str] = None
    replanner: Optional[str] = None
    retrieval_worker: Optional[str] = None
    analytics_worker: Optional[str] = None
    plot_worker: Optional[str] = None

    def get_component_retriever_name(self, component_name: str) -> str:
        """
        Get the retriever name for a specific component.
        Falls back to default if the component-specific name is not provided.
        """
        if (
            hasattr(self, component_name)
            and getattr(self, component_name) is not None
        ):
            return getattr(self, component_name)
        return self.default


class CwdAgentDefinitionConfig(BaseModel):
    agent_name: Optional[str] = Field(
        default="CwdAgent",
        description="An optional name for this agent configuration.",
    )
    use_case_name: str
    use_case_description: str
    llm_configs: Dict[str, LLMSelectionConfig]
    llm: CwdAgentLLMReferences
    resource_manager_config: ResourceManagerConfig
    retriever_config: RetrieverSelectionConfig
    workers: CwdAgentWorkersModulesConfig
    max_tasks: int = Field(
        description="Maximum number of tasks that can be executed before termination.",
        default=10,
    )
    timeout: int = Field(
        description="timeout in seconds for running agent on inputs",
        default=300,
    )

    class Config:
        extra = "forbid"

