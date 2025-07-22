"""Document data model for knowledge base operations."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """Represents a document in the knowledge base.
    
    This model is used for storing and retrieving contextual information
    that helps ground the agent's responses in relevant domain knowledge.
    """
    
    content: str = Field(
        description="The main text content of the document"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the document (e.g., source, tags, date)"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding of the document content for similarity search"
    )
    source: str = Field(
        description="The source or origin of this document"
    )
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )