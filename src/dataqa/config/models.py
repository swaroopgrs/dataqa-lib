"""
Pydantic configuration models for DataQA agents.

This module defines the configuration schema for all DataQA components including
agents, LLMs, knowledge bases, and executors. All configurations support YAML
loading and environment variable substitution.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class KnowledgeProvider(str, Enum):
    """Supported knowledge base providers."""
    FAISS = "faiss"
    OPENSEARCH = "opensearch"
    MEMORY = "memory"


class ExecutorProvider(str, Enum):
    """Supported executor providers."""
    INMEMORY = "inmemory"
    API = "api"
    DOCKER = "docker"


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    
    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="LLM provider to use"
    )
    model: str = Field(
        default="gpt-4",
        description="Model name/identifier"
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for the LLM provider"
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Custom API base URL"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens in response"
    )
    timeout: float = Field(
        default=30.0,
        gt=0,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for failed requests"
    )
    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific parameters"
    )

    @field_validator('api_key', mode='before')
    @classmethod
    def resolve_api_key(cls, v: Any) -> Optional[SecretStr]:
        """Resolve API key from environment variables if needed."""
        if v is None:
            return None
        if isinstance(v, str) and v.startswith('${') and v.endswith('}'):
            env_var = v[2:-1]
            env_value = os.getenv(env_var)
            if env_value:
                return SecretStr(env_value)
            return None
        return SecretStr(str(v)) if v else None


class KnowledgeConfig(BaseModel):
    """Configuration for knowledge base providers."""
    
    provider: KnowledgeProvider = Field(
        default=KnowledgeProvider.FAISS,
        description="Knowledge base provider to use"
    )
    index_path: Optional[Path] = Field(
        default=None,
        description="Path to knowledge base index"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    chunk_size: int = Field(
        default=512,
        gt=0,
        description="Text chunk size for document processing"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Overlap between text chunks"
    )
    top_k: int = Field(
        default=5,
        gt=0,
        description="Number of top results to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for retrieval"
    )
    
    # OpenSearch specific settings
    opensearch_host: Optional[str] = Field(
        default=None,
        description="OpenSearch host URL"
    )
    opensearch_port: Optional[int] = Field(
        default=9200,
        gt=0,
        le=65535,
        description="OpenSearch port"
    )
    opensearch_username: Optional[str] = Field(
        default=None,
        description="OpenSearch username"
    )
    opensearch_password: Optional[SecretStr] = Field(
        default=None,
        description="OpenSearch password"
    )
    opensearch_index: str = Field(
        default="dataqa-knowledge",
        description="OpenSearch index name"
    )
    
    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific parameters"
    )

    @field_validator('opensearch_password', mode='before')
    @classmethod
    def resolve_opensearch_password(cls, v: Any) -> Optional[SecretStr]:
        """Resolve OpenSearch password from environment variables if needed."""
        if v is None:
            return None
        if isinstance(v, str) and v.startswith('${') and v.endswith('}'):
            env_var = v[2:-1]
            env_value = os.getenv(env_var)
            if env_value:
                return SecretStr(env_value)
            return None
        return SecretStr(str(v)) if v else None


class ExecutorConfig(BaseModel):
    """Configuration for code execution environments."""
    
    provider: ExecutorProvider = Field(
        default=ExecutorProvider.INMEMORY,
        description="Executor provider to use"
    )
    
    # Database connection settings
    database_url: Optional[SecretStr] = Field(
        default=None,
        description="Database connection URL"
    )
    database_type: str = Field(
        default="duckdb",
        description="Database type (duckdb, postgresql, mysql, etc.)"
    )
    
    # Execution limits
    max_execution_time: float = Field(
        default=30.0,
        gt=0,
        description="Maximum execution time in seconds"
    )
    max_memory_mb: int = Field(
        default=512,
        gt=0,
        description="Maximum memory usage in MB"
    )
    max_rows: int = Field(
        default=10000,
        gt=0,
        description="Maximum number of rows to return"
    )
    
    # Security settings
    allow_file_access: bool = Field(
        default=False,
        description="Allow file system access in code execution"
    )
    allowed_imports: List[str] = Field(
        default_factory=lambda: [
            "pandas", "numpy", "matplotlib", "seaborn", 
            "datetime", "math", "statistics"
        ],
        description="List of allowed Python imports"
    )
    blocked_functions: List[str] = Field(
        default_factory=lambda: [
            "exec", "eval", "open", "__import__", "compile"
        ],
        description="List of blocked Python functions"
    )
    
    # API executor settings
    api_url: Optional[str] = Field(
        default=None,
        description="API URL for remote execution"
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for remote execution"
    )
    
    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific parameters"
    )

    @field_validator('database_url', 'api_key', mode='before')
    @classmethod
    def resolve_secrets(cls, v: Any) -> Optional[SecretStr]:
        """Resolve secrets from environment variables if needed."""
        if v is None:
            return None
        if isinstance(v, str) and v.startswith('${') and v.endswith('}'):
            env_var = v[2:-1]
            env_value = os.getenv(env_var)
            if env_value:
                return SecretStr(env_value)
            return None
        return SecretStr(str(v)) if v else None


class WorkflowConfig(BaseModel):
    """Configuration for agent workflow behavior."""
    
    strategy: str = Field(
        default="react",
        description="Agent reasoning strategy (react, workflow, plan_execute)"
    )
    max_iterations: int = Field(
        default=10,
        gt=0,
        description="Maximum number of workflow iterations"
    )
    require_approval: bool = Field(
        default=True,
        description="Require human approval for code execution"
    )
    auto_approve_safe: bool = Field(
        default=False,
        description="Auto-approve operations deemed safe"
    )
    conversation_memory: bool = Field(
        default=True,
        description="Enable conversation memory and context"
    )
    max_context_length: int = Field(
        default=4000,
        gt=0,
        description="Maximum context length for LLM prompts"
    )
    enable_visualization: bool = Field(
        default=True,
        description="Enable automatic visualization generation"
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug logging and verbose output"
    )


class AgentConfig(BaseModel):
    """Main configuration for DataQA agents."""
    
    name: str = Field(
        description="Agent name/identifier"
    )
    description: Optional[str] = Field(
        default=None,
        description="Agent description"
    )
    version: str = Field(
        default="1.0.0",
        description="Agent configuration version"
    )
    
    # Component configurations
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration"
    )
    knowledge: KnowledgeConfig = Field(
        default_factory=KnowledgeConfig,
        description="Knowledge base configuration"
    )
    executor: ExecutorConfig = Field(
        default_factory=ExecutorConfig,
        description="Executor configuration"
    )
    workflow: WorkflowConfig = Field(
        default_factory=WorkflowConfig,
        description="Workflow configuration"
    )
    
    # Global settings
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    data_dir: Path = Field(
        default=Path("./data"),
        description="Data directory path"
    )
    cache_dir: Path = Field(
        default=Path("./cache"),
        description="Cache directory path"
    )
    
    extra_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional configuration parameters"
    )

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

    @model_validator(mode='after')
    def validate_config_consistency(self) -> 'AgentConfig':
        """Validate configuration consistency across components."""
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate knowledge base path
        if (self.knowledge.provider == KnowledgeProvider.FAISS and 
            self.knowledge.index_path is None):
            self.knowledge.index_path = self.data_dir / "knowledge" / "faiss_index"
        
        return self