"""Unit tests for FAISS knowledge primitive."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from dataqa.exceptions import KnowledgeError
from dataqa.models.document import Document
from dataqa.primitives.faiss_knowledge import FAISSKnowledge


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            content="Python is a programming language",
            source="doc1",
            metadata={"category": "programming", "language": "en"}
        ),
        Document(
            content="Machine learning uses algorithms to find patterns",
            source="doc2", 
            metadata={"category": "ml", "language": "en"}
        ),
        Document(
            content="FAISS is a library for similarity search",
            source="doc3",
            metadata={"category": "search", "language": "en"}
        )
    ]


@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer model."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
    return mock_model


@pytest.fixture
def faiss_knowledge():
    """Create FAISSKnowledge instance for testing."""
    return FAISSKnowledge(model_name="test-model", embedding_dim=384)


class TestFAISSKnowledge:
    """Test cases for FAISSKnowledge class."""
    
    def test_initialization(self):
        """Test FAISSKnowledge initialization."""
        kb = FAISSKnowledge(
            model_name="custom-model",
            index_path="/tmp/test",
            embedding_dim=512
        )
        
        assert kb.model_name == "custom-model"
        assert kb.index_path == Path("/tmp/test")
        assert kb._embedding_dim == 512
        assert kb._embedding_model is None
        assert kb._index is None
        assert kb._documents == []
    
    def test_initialization_defaults(self):
        """Test FAISSKnowledge initialization with defaults."""
        kb = FAISSKnowledge()
        
        assert kb.model_name == "all-MiniLM-L6-v2"
        assert kb.index_path is None
        assert kb._embedding_dim is None
    
    @patch('dataqa.primitives.faiss_knowledge.SentenceTransformer')
    def test_embedding_model_property(self, mock_transformer_class, faiss_knowledge):
        """Test lazy loading of embedding model."""
        mock_model = MagicMock()
        mock_transformer_class.return_value = mock_model
        
        # First access should create the model
        model = faiss_knowledge.embedding_model
        assert model is mock_model
        mock_transformer_class.assert_called_once_with("test-model")
        
        # Second access should return cached model
        model2 = faiss_knowledge.embedding_model
        assert model2 is mock_model
        assert mock_transformer_class.call_count == 1
    
    @patch('dataqa.primitives.faiss_knowledge.SentenceTransformer')
    def test_embedding_model_error(self, mock_transformer_class, faiss_knowledge):
        """Test embedding model loading error."""
        mock_transformer_class.side_effect = Exception("Model not found")
        
        with pytest.raises(KnowledgeError, match="Failed to load embedding model"):
            _ = faiss_knowledge.embedding_model
    
    @patch('dataqa.primitives.faiss_knowledge.faiss')
    def test_index_property(self, mock_faiss, faiss_knowledge):
        """Test FAISS index creation."""
        mock_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_index
        
        # Mock embedding model
        faiss_knowledge._embedding_model = MagicMock()
        faiss_knowledge._embedding_model.get_sentence_embedding_dimension.return_value = 384
        
        index = faiss_knowledge.index
        assert index is mock_index
        mock_faiss.IndexFlatL2.assert_called_once_with(384)
    
    @pytest.mark.asyncio
    @patch('dataqa.primitives.faiss_knowledge.SentenceTransformer')
    async def test_ingest_documents(self, mock_transformer_class, faiss_knowledge, sample_documents):
        """Test document ingestion."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        mock_transformer_class.return_value = mock_model
        
        mock_index = MagicMock()
        faiss_knowledge._index = mock_index
        
        # Test ingestion
        await faiss_knowledge.ingest(sample_documents)
        
        # Verify embeddings were generated and added
        mock_model.encode.assert_called_once()
        mock_index.add.assert_called_once()
        assert len(faiss_knowledge._documents) == 3
        
        # Verify embeddings were added to documents
        for doc in faiss_knowledge._documents:
            assert doc.embedding is not None
            assert len(doc.embedding) == 384
    
    @pytest.mark.asyncio
    async def test_ingest_empty_documents(self, faiss_knowledge):
        """Test ingesting empty document list."""
        await faiss_knowledge.ingest([])
        assert len(faiss_knowledge._documents) == 0
    
    @pytest.mark.asyncio
    @patch('dataqa.primitives.faiss_knowledge.SentenceTransformer')
    async def test_ingest_documents_with_embeddings(self, mock_transformer_class, faiss_knowledge):
        """Test ingesting documents that already have embeddings."""
        # Create document with existing embedding
        doc = Document(
            content="Test content",
            source="test",
            embedding=[0.1] * 384
        )
        
        mock_model = MagicMock()
        mock_transformer_class.return_value = mock_model
        
        mock_index = MagicMock()
        faiss_knowledge._index = mock_index
        
        await faiss_knowledge.ingest([doc])
        
        # Should not call encode since embedding already exists
        mock_model.encode.assert_not_called()
        mock_index.add.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('dataqa.primitives.faiss_knowledge.SentenceTransformer')
    async def test_ingest_error(self, mock_transformer_class, faiss_knowledge, sample_documents):
        """Test ingestion error handling."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_transformer_class.return_value = mock_model
        
        with pytest.raises(KnowledgeError, match="Failed to ingest documents"):
            await faiss_knowledge.ingest(sample_documents)
    
    @pytest.mark.asyncio
    @patch('dataqa.primitives.faiss_knowledge.SentenceTransformer')
    async def test_search_documents(self, mock_transformer_class, faiss_knowledge, sample_documents):
        """Test document search."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = [
            np.random.rand(3, 384).astype(np.float32),  # For ingestion
            np.random.rand(1, 384).astype(np.float32)   # For search query
        ]
        mock_transformer_class.return_value = mock_model
        
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.1, 0.2, 0.3]]),  # distances
            np.array([[0, 1, 2]])         # indices
        )
        faiss_knowledge._index = mock_index
        
        # Ingest documents first
        await faiss_knowledge.ingest(sample_documents)
        
        # Test search
        results = await faiss_knowledge.search("test query", limit=2)
        
        assert len(results) == 2  # Limited by limit parameter
        assert all(isinstance(doc, Document) for doc in results)
        mock_index.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_empty_knowledge_base(self, faiss_knowledge):
        """Test search with empty knowledge base."""
        results = await faiss_knowledge.search("test query")
        assert results == []
    
    @pytest.mark.asyncio
    @patch('dataqa.primitives.faiss_knowledge.SentenceTransformer')
    async def test_search_with_filters(self, mock_transformer_class, faiss_knowledge, sample_documents):
        """Test search with metadata filters."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = [
            np.random.rand(3, 384).astype(np.float32),  # For ingestion
            np.random.rand(1, 384).astype(np.float32)   # For search query
        ]
        mock_transformer_class.return_value = mock_model
        
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.1, 0.2, 0.3]]),  # distances
            np.array([[0, 1, 2]])         # indices
        )
        faiss_knowledge._index = mock_index
        
        # Ingest documents first
        await faiss_knowledge.ingest(sample_documents)
        
        # Test search with filters
        results = await faiss_knowledge.search(
            "test query", 
            filters={"category": "programming"}
        )
        
        assert len(results) == 1
        assert results[0].metadata["category"] == "programming"
    
    @pytest.mark.asyncio
    @patch('dataqa.primitives.faiss_knowledge.SentenceTransformer')
    async def test_search_error(self, mock_transformer_class, faiss_knowledge, sample_documents):
        """Test search error handling."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Search failed")
        mock_transformer_class.return_value = mock_model
        
        faiss_knowledge._documents = sample_documents
        
        with pytest.raises(KnowledgeError, match="Failed to search documents"):
            await faiss_knowledge.search("test query")
    
    @pytest.mark.asyncio
    @patch('dataqa.primitives.faiss_knowledge.SentenceTransformer')
    async def test_update_document(self, mock_transformer_class, faiss_knowledge, sample_documents):
        """Test document update."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = [
            np.random.rand(3, 384).astype(np.float32),  # For ingestion
            np.random.rand(1, 384).astype(np.float32)   # For update
        ]
        mock_transformer_class.return_value = mock_model
        
        mock_index = MagicMock()
        faiss_knowledge._index = mock_index
        
        # Ingest documents first
        await faiss_knowledge.ingest(sample_documents)
        
        # Update document
        updated_doc = Document(
            content="Updated Python content",
            source="doc1",
            metadata={"category": "programming", "updated": True}
        )
        
        await faiss_knowledge.update("doc1", updated_doc)
        
        # Verify document was updated
        found_doc = next(doc for doc in faiss_knowledge._documents if doc.source == "doc1")
        assert found_doc.content == "Updated Python content"
        assert found_doc.metadata["updated"] is True
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_document(self, faiss_knowledge):
        """Test updating non-existent document."""
        doc = Document(content="test", source="test")
        
        with pytest.raises(KnowledgeError, match="Document with ID 'nonexistent' not found"):
            await faiss_knowledge.update("nonexistent", doc)
    
    @pytest.mark.asyncio
    @patch('dataqa.primitives.faiss_knowledge.SentenceTransformer')
    async def test_delete_document(self, mock_transformer_class, faiss_knowledge, sample_documents):
        """Test document deletion."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        mock_transformer_class.return_value = mock_model
        
        mock_index = MagicMock()
        faiss_knowledge._index = mock_index
        
        # Ingest documents first
        await faiss_knowledge.ingest(sample_documents)
        initial_count = len(faiss_knowledge._documents)
        
        # Delete document
        await faiss_knowledge.delete("doc1")
        
        # Verify document was deleted
        assert len(faiss_knowledge._documents) == initial_count - 1
        assert not any(doc.source == "doc1" for doc in faiss_knowledge._documents)
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, faiss_knowledge):
        """Test deleting non-existent document."""
        with pytest.raises(KnowledgeError, match="Document with ID 'nonexistent' not found"):
            await faiss_knowledge.delete("nonexistent")
    
    @pytest.mark.asyncio
    async def test_get_stats(self, faiss_knowledge, sample_documents):
        """Test getting knowledge base statistics."""
        # Mock index
        mock_index = MagicMock()
        mock_index.ntotal = 3
        faiss_knowledge._index = mock_index
        faiss_knowledge._documents = sample_documents
        faiss_knowledge._embedding_dim = 384
        
        stats = await faiss_knowledge.get_stats()
        
        expected_stats = {
            "document_count": 3,
            "index_size": 3,
            "embedding_dimension": 384,
            "model_name": "test-model",
            "index_path": None,
        }
        
        assert stats == expected_stats
    
    @pytest.mark.asyncio
    @patch('dataqa.primitives.faiss_knowledge.faiss')
    async def test_save_knowledge_base(self, mock_faiss, faiss_knowledge, sample_documents):
        """Test saving knowledge base to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_kb"
            
            # Setup knowledge base
            mock_index = MagicMock()
            mock_index.ntotal = 3
            faiss_knowledge._index = mock_index
            faiss_knowledge._documents = sample_documents
            faiss_knowledge._embedding_dim = 384
            
            # Save knowledge base
            await faiss_knowledge.save(save_path)
            
            # Verify files were created
            assert (save_path / "documents.json").exists()
            assert (save_path / "config.json").exists()
            mock_faiss.write_index.assert_called_once()
            
            # Verify document data
            with open(save_path / "documents.json") as f:
                saved_docs = json.load(f)
            assert len(saved_docs) == 3
            
            # Verify config data
            with open(save_path / "config.json") as f:
                config = json.load(f)
            assert config["model_name"] == "test-model"
            assert config["embedding_dim"] == 384
            assert config["document_count"] == 3
    
    @pytest.mark.asyncio
    async def test_save_no_path(self, faiss_knowledge):
        """Test saving without specifying path."""
        with pytest.raises(KnowledgeError, match="No save path specified"):
            await faiss_knowledge.save()
    
    @pytest.mark.asyncio
    @patch('dataqa.primitives.faiss_knowledge.faiss')
    async def test_load_knowledge_base(self, mock_faiss, faiss_knowledge):
        """Test loading knowledge base from disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            load_path = Path(temp_dir) / "test_kb"
            load_path.mkdir()
            
            # Create test files
            documents_data = [
                {
                    "content": "Test content",
                    "source": "test",
                    "metadata": {"category": "test"},
                    "embedding": [0.1] * 384
                }
            ]
            
            with open(load_path / "documents.json", "w") as f:
                json.dump(documents_data, f)
            
            config_data = {
                "model_name": "loaded-model",
                "embedding_dim": 384,
                "document_count": 1
            }
            
            with open(load_path / "config.json", "w") as f:
                json.dump(config_data, f)
            
            # Create dummy index file
            (load_path / "index.faiss").touch()
            
            # Mock FAISS loading
            mock_index = MagicMock()
            mock_faiss.read_index.return_value = mock_index
            
            # Load knowledge base
            await faiss_knowledge.load(load_path)
            
            # Verify data was loaded
            assert faiss_knowledge.model_name == "loaded-model"
            assert faiss_knowledge._embedding_dim == 384
            assert len(faiss_knowledge._documents) == 1
            assert faiss_knowledge._documents[0].content == "Test content"
            assert faiss_knowledge._index is mock_index
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_path(self, faiss_knowledge):
        """Test loading from non-existent path."""
        with pytest.raises(KnowledgeError, match="Knowledge base path does not exist"):
            await faiss_knowledge.load("/nonexistent/path")
    
    @pytest.mark.asyncio
    @patch('dataqa.primitives.faiss_knowledge.SentenceTransformer')
    async def test_rebuild_index(self, mock_transformer_class, faiss_knowledge, sample_documents):
        """Test rebuilding FAISS index."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_transformer_class.return_value = mock_model
        
        # Set documents without embeddings
        doc_without_embedding = Document(content="test", source="test")
        faiss_knowledge._documents = [doc_without_embedding]
        faiss_knowledge._embedding_dim = 384
        
        # Rebuild index
        await faiss_knowledge._rebuild_index()
        
        # Verify embedding was generated
        assert faiss_knowledge._documents[0].embedding is not None
        assert faiss_knowledge._index is not None