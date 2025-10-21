from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from pydantic import BaseModel, Field

# --- Core Data Contracts ---


class CoreConversationTurn(BaseModel):
    """A single, generic turn in a conversation's history."""

    query: str = Field(..., description="The user's query for this turn.")
    output_text: str = Field(
        ..., description="The final text response from the agent for this turn."
    )


class CoreRequest(BaseModel):
    """A generic request to process a query via a DataQA client."""

    user_query: str = Field(
        ..., description="The current natural language query from the user."
    )
    question_id: str = Field(
        ..., description="A unique identifier for the question."
    )
    conversation_id: str = Field(
        ..., description="A unique identifier for the conversation session."
    )
    history: List[CoreConversationTurn] = Field(
        default_factory=list,
        description="Previous turns in the conversation for context.",
    )


class CoreStep(BaseModel):
    """A generic representation of an intermediate processing step for debugging."""

    name: str = Field(
        ..., description="Name or identifier of the processing step."
    )
    content: str = Field(
        default="", description="Content or details of the step."
    )


class CoreStatus(BaseModel):
    """A generic representation for the status of the core agent during inference."""

    name: str = Field(
        ..., description="Name or identifier of the processing step."
    )
    message: str = Field(..., description="A text message to be streamed.")


class CoreResponse(BaseModel):
    """A generic response from a DataQA client."""

    text: str = Field(
        ..., description="The main text response to the user query."
    )
    output_dataframes: List[pd.DataFrame] = Field(
        default_factory=list,
        description="A list of pandas DataFrames generated as output.",
    )
    output_images: List[bytes] = Field(
        default_factory=list,
        description="A list of images (as bytes) generated as output.",
    )
    steps: List[CoreStep] = Field(
        default_factory=list,
        description="A list of intermediate processing steps for transparency.",
    )

    class Config:
        arbitrary_types_allowed = True


# --- Abstract Client Interface ---


class DataQAClient(ABC):
    """
    Abstract Base Class for a DataQA client.

    This defines the standard interface for interacting with the DataQA agentic system,
    regardless of the execution environment (local, DBC service, etc.).
    """

    @abstractmethod
    async def process_query(self, request: CoreRequest) -> CoreResponse:
        """
        Processes a user query and returns a structured response.
        """
        raise NotImplementedError
