"""Message data model for conversation handling."""

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """Represents a message in a conversation between user and agent.
    
    This model is used to maintain conversation history and context
    throughout agent interactions.
    """
    
    role: Literal["user", "assistant", "system"] = Field(
        description="The role of the message sender"
    )
    content: str = Field(
        description="The actual message content"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the message was created"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata associated with the message"
    )
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )