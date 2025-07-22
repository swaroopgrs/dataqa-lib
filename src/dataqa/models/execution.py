"""Execution result data model for code execution operations."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExecutionResult(BaseModel):
    """Represents the result of executing generated code.
    
    This model captures both successful execution results and error information,
    providing a consistent interface for handling execution outcomes.
    """
    
    success: bool = Field(
        description="Whether the execution completed successfully"
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The resulting data from execution (e.g., DataFrame as dict, plot data)"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if execution failed"
    )
    execution_time: float = Field(
        description="Time taken to execute the code in seconds"
    )
    code_executed: str = Field(
        description="The actual code that was executed"
    )
    output_type: Optional[str] = Field(
        default=None,
        description="Type of output produced (e.g., 'dataframe', 'plot', 'scalar')"
    )
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )