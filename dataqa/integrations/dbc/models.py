# dataqa/integrations/dbc/models.py
from typing import List, Optional, Set
from pydantic import BaseModel, Field
from enum import StrEnum, auto
import uuid

# Import the core asset models to use in IngestionData
from dataqa.core.data_models.asset_models import Rules, DatabaseSchema, Examples

class FileType(StrEnum):
    """Enumeration for specifying which asset types to fetch."""
    RULES = auto()
    SCHEMA = auto()
    EXAMPLES = auto()

class IngestionData(BaseModel):
    """
    Defines the structured data object returned by the `asset_callable`.
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
    tenant_id: str
    usecase_name: str
    usecase_description: str

class ConversationTurn(BaseModel):
    """
    A single turn in the conversation history from a DBCRequest.
    """
    query: str = Field(..., description="The user query from this conversation turn.")
    output_text: str = Field(
        ...,
        description="The final text response from the turn, including dataframe summaries."
    )

class DBCRequest(BaseModel):
    """
    The standardized request format from the DBC service.
    """
    user_query: str = Field(..., description="The natural language query from the user.")
    conversation_id: str = Field(..., description="Unique identifier for the conversation session.")
    question_id: str = Field(..., description="Unique identifier for this specific question.")
    conversation_history: List[ConversationTurn] = Field(
        default_factory=list,
        description="Previous conversation turns for context."
    )

class StepResponse(BaseModel):
    """
    An intermediate processing step in the agent's execution trace.
    """
    name: str = Field(..., description="Name of the processing step.")
    content: str = Field(default="", description="A summary of what happened in this step.")

class DBCResponse(BaseModel):
    """
    The standardized response format from the DataQA library to the DBC service.
    """
    text: str = Field(..., description="The main text response to the user query.")
    output_df_names: List[str] = Field(
        default_factory=list,
        description="List of S3 paths to dataframes generated."
    )
    output_image_names: List[str] = Field(
        default_factory=list,
        description="List of S3 paths to images/plots generated."
    )
    steps: List[StepResponse] = Field(
        default_factory=list,
        description="A list of intermediate processing steps for transparency."
    )