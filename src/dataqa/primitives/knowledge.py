"""Abstract base class for knowledge primitive implementations."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..exceptions import KnowledgeError
from ..models.document import Document


class KnowledgePrimitive(ABC):
    """Abstract base class for knowledge retrieval systems.
    
    This interface defines the contract for components that manage
    domain-specific knowledge and provide context for agent responses.
    Implementations might use vector databases, traditional search,
    or hybrid approaches.
    """
    
    @abstractmethod
    async def ingest(self, documents: List[Document]) -> None:
        """Ingest documents into the knowledge base.
        
        Args:
            documents: List of documents to add to the knowledge base
            
        Raises:
            KnowledgeError: If ingestion fails
        """
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        limit: int = 5,
        filters: Optional[dict] = None
    ) -> List[Document]:
        """Search for relevant documents based on a query.
        
        Args:
            query: The search query string
            limit: Maximum number of documents to return
            filters: Optional filters to apply to the search
            
        Returns:
            List of relevant documents ordered by relevance
            
        Raises:
            KnowledgeError: If search fails
        """
        pass
    
    @abstractmethod
    async def update(self, document_id: str, document: Document) -> None:
        """Update an existing document in the knowledge base.
        
        Args:
            document_id: Unique identifier for the document
            document: Updated document content
            
        Raises:
            KnowledgeError: If update fails or document not found
        """
        pass
    
    @abstractmethod
    async def delete(self, document_id: str) -> None:
        """Delete a document from the knowledge base.
        
        Args:
            document_id: Unique identifier for the document to delete
            
        Raises:
            KnowledgeError: If deletion fails or document not found
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> dict:
        """Get statistics about the knowledge base.
        
        Returns:
            Dictionary containing stats like document count, index size, etc.
        """
        pass