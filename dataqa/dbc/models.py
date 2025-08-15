from typing import List
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """
    Represents a single conversation turn in the history.
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
    """
    name: str = Field(..., description="Name or identifier of the processing step")
    content: str = Field(default="", description="Content or details of the step")


class DBCResponse(BaseModel):
    """
    Standardized response format from DataQA library to DBC service.
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