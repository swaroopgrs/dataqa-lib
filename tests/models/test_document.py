"""Tests for Document data model."""

import pytest
from pydantic import ValidationError

from src.dataqa.models.document import Document


class TestDocument:
    """Test cases for Document model."""
    
    def test_document_creation_with_required_fields(self):
        """Test creating a document with only required fields."""
        document = Document(
            content="This is a test document",
            source="test_source"
        )
        
        assert document.content == "This is a test document"
        assert document.source == "test_source"
        assert document.metadata == {}
        assert document.embedding is None
    
    def test_document_creation_with_all_fields(self):
        """Test creating a document with all fields."""
        content = "Sample document content"
        metadata = {"author": "test", "category": "example"}
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        source = "test_database"
        
        document = Document(
            content=content,
            metadata=metadata,
            embedding=embedding,
            source=source
        )
        
        assert document.content == content
        assert document.metadata == metadata
        assert document.embedding == embedding
        assert document.source == source
    
    def test_document_content_required(self):
        """Test that content field is required."""
        with pytest.raises(ValidationError):
            Document(source="test")
    
    def test_document_source_required(self):
        """Test that source field is required."""
        with pytest.raises(ValidationError):
            Document(content="test content")
    
    def test_document_metadata_defaults_to_empty_dict(self):
        """Test that metadata defaults to empty dictionary."""
        document = Document(content="test", source="test")
        assert document.metadata == {}
    
    def test_document_embedding_validation(self):
        """Test embedding field validation."""
        # Valid embedding (list of floats)
        document = Document(
            content="test",
            source="test",
            embedding=[1.0, 2.0, 3.0]
        )
        assert document.embedding == [1.0, 2.0, 3.0]
        
        # None embedding should be allowed
        document = Document(content="test", source="test", embedding=None)
        assert document.embedding is None
    
    def test_document_json_serialization(self):
        """Test that document can be serialized to JSON."""
        document = Document(
            content="test content",
            source="test_source",
            metadata={"key": "value"},
            embedding=[0.1, 0.2, 0.3]
        )
        
        json_data = document.model_dump()
        
        assert json_data["content"] == "test content"
        assert json_data["source"] == "test_source"
        assert json_data["metadata"] == {"key": "value"}
        assert json_data["embedding"] == [0.1, 0.2, 0.3]
    
    def test_document_from_dict(self):
        """Test creating document from dictionary."""
        data = {
            "content": "document content",
            "source": "api",
            "metadata": {"type": "manual", "version": 1},
            "embedding": [1.1, 2.2, 3.3]
        }
        
        document = Document(**data)
        
        assert document.content == "document content"
        assert document.source == "api"
        assert document.metadata == {"type": "manual", "version": 1}
        assert document.embedding == [1.1, 2.2, 3.3]
    
    def test_document_complex_metadata(self):
        """Test document with complex metadata structures."""
        complex_metadata = {
            "tags": ["important", "database"],
            "created_at": "2024-01-01",
            "nested": {
                "level1": {
                    "level2": "deep_value"
                }
            },
            "numbers": [1, 2, 3]
        }
        
        document = Document(
            content="test",
            source="test",
            metadata=complex_metadata
        )
        
        assert document.metadata == complex_metadata