import uuid
from enum import StrEnum, auto
from typing import List, Optional

from pydantic import BaseModel, Field

from dataqa.core.agent.cwd_agent.config import DialectConfig

# Import the core asset models to use in IngestionData
from dataqa.core.data_models.asset_models import DatabaseSchema, Examples, Rules


class FileType(StrEnum):
    """Enumeration for specifying which asset types to fetch."""

    RULES = auto()
    SCHEMA = auto()
    EXAMPLES = auto()


class IngestionData(BaseModel):
    """
    Defines the structured data object returned by the `s3_retrieval`.
    """

    rules: Optional[Rules] = None
    schema: Optional[DatabaseSchema] = None
    examples: Optional[Examples] = None

    class Config:
        arbitrary_types_allowed = True


class UsecaseConfig(BaseModel):
    """
    High-level configuration for a specific use case from the DBC service.
    """

    config_id: uuid.UUID
    tenant_id: uuid.UUID
    usecase_name: str
    usecase_description: str
    sql_dialect: DialectConfig = Field(
        default_factory=DialectConfig,
        description="SQL dialect configuration for the use case.",
    )


class ConversationTurn(BaseModel):
    """
    A single turn in the conversation history from a DBCRequest.
    """

    question_id: uuid.UUID = Field(
        ..., description="Unique identifier for this specific question."
    )
    query: str = Field(
        ..., description="The user query from this conversation turn."
    )
    output_text: str = Field(
        ..., description="The final text response from the turn."
    )
    output_dataframes: List[str] = Field(
        default_factory=list, description="file names for the dataframes"
    )


class DBCRequest(BaseModel):
    """
    The standardized request format from the DBC service.
    """

    user_query: str = Field(
        ..., description="The natural language query from the user."
    )
    conversation_id: uuid.UUID = Field(
        ..., description="Unique identifier for the conversation session."
    )
    question_id: uuid.UUID = Field(
        ..., description="Unique identifier for this specific question."
    )
    conversation_history: List[ConversationTurn] = Field(
        default_factory=list,
        description="Previous conversation turns for context.",
    )


class StatusResponse(BaseModel):
    """
    An intermediate status and its streaming message in the agent's execution trace.
    """

    name: str = Field(..., description="Name of the processing step.")
    message: str = Field(..., description="A text message to be streamed.")


class StepResponse(BaseModel):
    """
    An intermediate processing step in the agent's execution trace.
    """

    name: str = Field(..., description="Name of the processing step.")
    content: str = Field(
        default="", description="A summary of what happened in this step."
    )


class DBCResponse(BaseModel):
    """
    The standardized response format from the DataQA library to the DBC service.
    """

    text: str = Field(
        ..., description="The main text response to the user query."
    )
    output_df_names: List[str] = Field(
        default_factory=list,
        description="List of S3 paths to dataframes generated.",
    )
    output_image_names: List[str] = Field(
        default_factory=list,
        description="List of S3 paths to images/plots generated.",
    )
    steps: List[StepResponse] = Field(
        default_factory=list,
        description="A list of intermediate processing steps for transparency.",
    )
