"""
DBC Service Integration Models

Request/response models and configuration classes for DBC service integration.
"""

from typing import List, Callable
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """
    Represents a single conversation turn in the history.
    
    Used to reconstruct context for the CWDAgent when processing queries.
    """
    query: str = Field(..., description="The user query from this conversation turn")
    output_text: str = Field(..., description="The text response generated")
    output_df_names: List[str] = Field(
        default_factory=list,
        description="List of S3 paths to dataframes generated in this turn"
    )


class DBCRequest(BaseModel):
    """
    Standardized request format from DBC service to DataQA library.
    
    This model defines the interface that the DBC service uses to send
    queries to the DataQA library through the DBCClient.
    """
    user_query: str = Field(..., description="The natural language query from the user")
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    question_id: str = Field(..., description="Unique identifier for this specific question")
    conversation_history: List[ConversationTurn] = Field(
        default_factory=list,
        description="Previous conversation turns for context"
    )


class StepResponse(BaseModel):
    """
    Represents an intermediate processing step in the query execution.
    
    Used to provide visibility into the agent's reasoning and execution process.
    """
    name: str = Field(..., description="Name or identifier of the processing step")
    content: str = Field(default="", description="Content or details of the step")


class DBCResponse(BaseModel):
    """
    Standardized response format from DataQA library to DBC service.
    
    This model defines the structured response that the DBC service expects
    to receive from the DataQA library after processing a query.
    """
    text: str = Field(..., description="The main text response to the user query")
    output_df_names: List[str] = Field(
        default_factory=list,
        description="List of S3 paths to dataframes generated during processing"
    )
    output_image_names: List[str] = Field(
        default_factory=list,
        description="List of S3 paths to images/plots generated during processing"
    )
    steps: List[StepResponse] = Field(
        default_factory=list,
        description="List of intermediate processing steps for transparency"
    )


class DBCClientConfig(BaseModel):
    """
    Configuration parameters for DBC client initialization.
    
    Contains service-specific configuration and callable functions that the DBC service provides
    when creating a DBCClient instance.
    """
    config_id: str = Field(..., description="Configuration identifier for the DBC service")
    tenant_id: str = Field(..., description="Tenant identifier for multi-tenant support")
    agent_config_path: str = Field(
        default="default_cwd_agent_config.yml",
        description="Path to the CWDAgent configuration file"
    )
    llm_callable: Callable = Field(..., description="Callable function for LLM operations")
    s3_callable: Callable = Field(..., description="Callable function for S3 operations")
    sql_callable: Callable = Field(..., description="Callable function for SQL execution")
    
    class Config:
        arbitrary_types_allowed = True