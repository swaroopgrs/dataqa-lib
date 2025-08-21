from datetime import datetime
from typing import Any, List, Union

from pydantic import BaseModel, Field


class PipelineError(BaseModel):
    error_code: int
    error_type: str
    error_message: str


class PipelineInput(BaseModel):
    query: str = Field(description="the input query.")
    context: List[str] = (
        Field(  # TODO support a list of str as the conversation history
            default_factory=list, description="the conversation history."
        )
    )
    previous_rewritten_query: str = Field(
        default="",
        description="the `rewritten_query` field from the last state in the same conversation.",
    )
    datetime: str = Field(
        default=str(datetime.today()), description="current datetime"
    )


class PipelineOutput(BaseModel):
    rewritten_query: str = Field(
        default="None",
        description="""
            The newly generated rewritten query for the input query.
            Any rewriter components should always save rewritten query to this field.
        """,
    )
    code: str = Field(
        default="", description="the final generated code to be returned"
    )
    dataframe: List[str] = Field(default_factory=list)
    image_byte_str: List[str] = Field(default_factory=list)
    code_running_log: str = ""
    code_running_error: str = ""
    text: str = Field(
        default="", description="any textual output generated from LLM pipeline"
    )


class BasePipelineState(BaseModel):
    # static import fields
    input: PipelineInput = Field(description="the input to a pipeline")
    return_output: PipelineOutput = Field(
        default=None, description="The output that may be displayed to users."
    )

    # metadata
    total_time: float = Field(default=0, description="Pipeline running time")
    error: Union[PipelineError, None] = Field(
        default=None,
        description="Save the exception occured during pipeline execution",
    )
    full_state: Any = Field(
        default=None,
        description="Return full pipeline state for debugging and logging purpose",
    )
