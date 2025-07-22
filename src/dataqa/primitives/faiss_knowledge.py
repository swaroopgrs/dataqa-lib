"""FAISS-based knowledge primitive implementation."""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..exceptions import KnowledgeError
from ..models.document import Document
from .knowledge import KnowledgePrimitive


class FAISSKnowledge(KnowledgePrimitive):
    """FAISS-based knowledge retrieval system.
    
    This implementation uses FAISS for efficient vector similarity search
    and sentence-transformers for generating embeddings. It supports
    persistence to disk for loading pre-built knowledge bases.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: Optional[Union[str, Path]] = None,
        embedding_dim: Optional[int] = None,
    ):
        """Initialize the FAISS knowledge primitive.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            index_path: Path to save/load the FAISS index and metadata
            embedding_dim: Dimension of embeddings (auto-detected if None)
        """
        self.model_name = model_name
        self.index_path = Path(index_path) if index_path else None
        self._embedding_model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.Index] = None
        self._documents: List[Document] = []
        self._embedding_dim = embedding_dim
        
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._embedding_model is None:
            try:
                self._embedding_model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise KnowledgeError(f"Failed to load embedding model '{self.model_name}': {e}")
        return self._embedding_model
    
    @property
    def index(self) -> faiss.Index:
        """Get or create the FAISS index."""
        if self._index is None:
            if self._embedding_dim is None:
                # Get embedding dimension from model
                self._embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            # Create a flat L2 index for simplicity
            self._index = faiss.IndexFlatL2(self._embedding_dim)
            
        return self._index
    
    async def ingest(self, documents: List[Document]) -> None:
        """Ingest documents into the knowledge base.
        
        Args:
            documents: List of documents to add to the knowledge base
            
        Raises:
            KnowledgeError: If ingestion fails
        """
        if not documents:
            return
            
        try:
            # Generate embeddings for documents that don't have them
            texts_to_embed = []
            doc_indices = []
            
            for i, doc in enumerate(documents):
                if doc.embedding is None:
                    texts_to_embed.append(doc.content)
                    doc_indices.append(i)
            
            if texts_to_embed:
                embeddings = self.embedding_model.encode(
                    texts_to_embed,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                # Update documents with embeddings
                for i, embedding in zip(doc_indices, embeddings):
                    documents[i].embedding = embedding.tolist()
            
            # Add embeddings to FAISS index
            embeddings_array = np.array([
                doc.embedding for doc in documents
            ], dtype=np.float32)
            
            self.index.add(embeddings_array)
            
            # Store documents
            self._documents.extend(documents)
            
        except Exception as e:
            raise KnowledgeError(f"Failed to ingest documents: {e}")
    
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
            filters: Optional filters to apply to the search (not implemented yet)
            
        Returns:
            List of relevant documents ordered by relevance
            
        Raises:
            KnowledgeError: If search fails
        """
        if not self._documents:
            return []
            
        try:
            # Generate embedding for query
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Search in FAISS index
            distances, indices = self.index.search(
                query_embedding.astype(np.float32), 
                min(limit, len(self._documents))
            )
            
            # Return documents ordered by relevance
            results = []
            for idx in indices[0]:
                if idx != -1:  # FAISS returns -1 for empty slots
                    results.append(self._documents[idx])
            
            # Apply filters if provided (basic implementation)
            if filters:
                filtered_results = []
                for doc in results:
                    match = True
                    for key, value in filters.items():
                        if key not in doc.metadata or doc.metadata[key] != value:
                            match = False
                            break
                    if match:
                        filtered_results.append(doc)
                results = filtered_results
            
            # Apply limit after filtering
            return results[:limit]
            
        except Exception as e:
            raise KnowledgeError(f"Failed to search documents: {e}")
    
    async def update(self, document_id: str, document: Document) -> None:
        """Update an existing document in the knowledge base.
        
        Args:
            document_id: Unique identifier for the document
            document: Updated document content
            
        Raises:
            KnowledgeError: If update fails or document not found
        """
        # Find document by ID (using source as ID for now)
        doc_index = None
        for i, doc in enumerate(self._documents):
            if doc.source == document_id:
                doc_index = i
                break
        
        if doc_index is None:
            raise KnowledgeError(f"Document with ID '{document_id}' not found")
        
        try:
            # Generate new embedding if content changed
            if document.embedding is None:
                embedding = self.embedding_model.encode(
                    [document.content],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                document.embedding = embedding[0].tolist()
            
            # Update document in list
            self._documents[doc_index] = document
            
            # Rebuild index (for simplicity - could be optimized)
            await self._rebuild_index()
            
        except Exception as e:
            raise KnowledgeError(f"Failed to update document '{document_id}': {e}")
    
    async def delete(self, document_id: str) -> None:
        """Delete a document from the knowledge base.
        
        Args:
            document_id: Unique identifier for the document to delete
            
        Raises:
            KnowledgeError: If deletion fails or document not found
        """
        # Find document by ID (using source as ID for now)
        doc_index = None
        for i, doc in enumerate(self._documents):
            if doc.source == document_id:
                doc_index = i
                break
        
        if doc_index is None:
            raise KnowledgeError(f"Document with ID '{document_id}' not found")
        
        try:
            # Remove document from list
            self._documents.pop(doc_index)
            
            # Rebuild index (for simplicity - could be optimized)
            await self._rebuild_index()
            
        except Exception as e:
            raise KnowledgeError(f"Failed to delete document '{document_id}': {e}")
    
    async def get_stats(self) -> dict:
        """Get statistics about the knowledge base.
        
        Returns:
            Dictionary containing stats like document count, index size, etc.
        """
        return {
            "document_count": len(self._documents),
            "index_size": self.index.ntotal if self._index else 0,
            "embedding_dimension": self._embedding_dim,
            "model_name": self.model_name,
            "index_path": str(self.index_path) if self.index_path else None,
        }
    
    async def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save the knowledge base to disk.
        
        Args:
            path: Path to save the knowledge base (uses self.index_path if None)
            
        Raises:
            KnowledgeError: If saving fails
        """
        save_path = Path(path) if path else self.index_path
        if save_path is None:
            raise KnowledgeError("No save path specified")
        
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            if self._index and self._index.ntotal > 0:
                faiss.write_index(self._index, str(save_path / "index.faiss"))
            
            # Save documents and metadata
            documents_data = [doc.model_dump() for doc in self._documents]
            with open(save_path / "documents.json", "w") as f:
                json.dump(documents_data, f, indent=2)
            
            # Save configuration
            config = {
                "model_name": self.model_name,
                "embedding_dim": self._embedding_dim,
                "document_count": len(self._documents),
            }
            with open(save_path / "config.json", "w") as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            raise KnowledgeError(f"Failed to save knowledge base: {e}")
    
    async def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """Load the knowledge base from disk.
        
        Args:
            path: Path to load the knowledge base from (uses self.index_path if None)
            
        Raises:
            KnowledgeError: If loading fails
        """
        load_path = Path(path) if path else self.index_path
        if load_path is None or not load_path.exists():
            raise KnowledgeError(f"Knowledge base path does not exist: {load_path}")
        
        try:
            # Load configuration
            config_path = load_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                self.model_name = config.get("model_name", self.model_name)
                self._embedding_dim = config.get("embedding_dim")
            
            # Load documents
            documents_path = load_path / "documents.json"
            if documents_path.exists():
                with open(documents_path) as f:
                    documents_data = json.load(f)
                self._documents = [Document(**doc_data) for doc_data in documents_data]
            
            # Load FAISS index
            index_path = load_path / "index.faiss"
            if index_path.exists():
                self._index = faiss.read_index(str(index_path))
            else:
                # Rebuild index from documents if index file doesn't exist
                await self._rebuild_index()
                
        except Exception as e:
            raise KnowledgeError(f"Failed to load knowledge base: {e}")
    
    async def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from current documents."""
        if not self._documents:
            return
            
        # Reset index
        if self._embedding_dim is None:
            self._embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self._index = faiss.IndexFlatL2(self._embedding_dim)
        
        # Add all embeddings
        embeddings = []
        for doc in self._documents:
            if doc.embedding is None:
                # Generate embedding if missing
                embedding = self.embedding_model.encode(
                    [doc.content],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                doc.embedding = embedding[0].tolist()
            embeddings.append(doc.embedding)
        
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self._index.add(embeddings_array)