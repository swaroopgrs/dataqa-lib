"""Tests for primitive interface definitions."""

import pytest
from abc import ABC

from src.dataqa.primitives.knowledge import KnowledgePrimitive
from src.dataqa.primitives.executor import ExecutorPrimitive
from src.dataqa.primitives.llm import LLMInterface


class TestPrimitiveInterfaces:
    """Test cases for primitive interface definitions."""
    
    def test_knowledge_primitive_is_abstract(self):
        """Test that KnowledgePrimitive is an abstract base class."""
        assert issubclass(KnowledgePrimitive, ABC)
        
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            KnowledgePrimitive()
    
    def test_executor_primitive_is_abstract(self):
        """Test that ExecutorPrimitive is an abstract base class."""
        assert issubclass(ExecutorPrimitive, ABC)
        
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            ExecutorPrimitive()
    
    def test_llm_interface_is_abstract(self):
        """Test that LLMInterface is an abstract base class."""
        assert issubclass(LLMInterface, ABC)
        
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            LLMInterface()
    
    def test_knowledge_primitive_has_required_methods(self):
        """Test that KnowledgePrimitive defines all required abstract methods."""
        required_methods = [
            'ingest', 'search', 'update', 'delete', 'get_stats'
        ]
        
        for method_name in required_methods:
            assert hasattr(KnowledgePrimitive, method_name)
            method = getattr(KnowledgePrimitive, method_name)
            assert getattr(method, '__isabstractmethod__', False)
    
    def test_executor_primitive_has_required_methods(self):
        """Test that ExecutorPrimitive defines all required abstract methods."""
        required_methods = [
            'execute_sql', 'execute_python', 'get_schema', 
            'list_tables', 'get_sample_data', 'validate_code'
        ]
        
        for method_name in required_methods:
            assert hasattr(ExecutorPrimitive, method_name)
            method = getattr(ExecutorPrimitive, method_name)
            assert getattr(method, '__isabstractmethod__', False)
    
    def test_llm_interface_has_required_methods(self):
        """Test that LLMInterface defines all required abstract methods."""
        required_methods = [
            'generate_code', 'analyze_query', 'format_response',
            'generate_clarification', 'validate_generated_code', 'get_model_info'
        ]
        
        for method_name in required_methods:
            assert hasattr(LLMInterface, method_name)
            method = getattr(LLMInterface, method_name)
            assert getattr(method, '__isabstractmethod__', False)


class TestConcreteImplementations:
    """Test that concrete implementations must implement all abstract methods."""
    
    def test_incomplete_knowledge_primitive_fails(self):
        """Test that incomplete KnowledgePrimitive implementation fails."""
        
        class IncompleteKnowledge(KnowledgePrimitive):
            async def ingest(self, documents):
                pass
            # Missing other required methods
        
        with pytest.raises(TypeError):
            IncompleteKnowledge()
    
    def test_incomplete_executor_primitive_fails(self):
        """Test that incomplete ExecutorPrimitive implementation fails."""
        
        class IncompleteExecutor(ExecutorPrimitive):
            async def execute_sql(self, sql, parameters=None):
                pass
            # Missing other required methods
        
        with pytest.raises(TypeError):
            IncompleteExecutor()
    
    def test_incomplete_llm_interface_fails(self):
        """Test that incomplete LLMInterface implementation fails."""
        
        class IncompleteLLM(LLMInterface):
            async def generate_code(self, query, context=None, code_type="sql", conversation_history=None):
                pass
            # Missing other required methods
        
        with pytest.raises(TypeError):
            IncompleteLLM()
    
    def test_complete_implementations_work(self):
        """Test that complete implementations can be instantiated."""
        
        class CompleteKnowledge(KnowledgePrimitive):
            async def ingest(self, documents):
                pass
            async def search(self, query, limit=5, filters=None):
                return []
            async def update(self, document_id, document):
                pass
            async def delete(self, document_id):
                pass
            async def get_stats(self):
                return {}
        
        class CompleteExecutor(ExecutorPrimitive):
            async def execute_sql(self, sql, parameters=None):
                pass
            async def execute_python(self, code, context=None):
                pass
            async def get_schema(self, table_name=None):
                return {}
            async def list_tables(self):
                return []
            async def get_sample_data(self, table_name, limit=5):
                pass
            async def validate_code(self, code, code_type):
                return True
            async def generate_visualization(self, data, chart_type="auto", **kwargs):
                pass
        
        class CompleteLLM(LLMInterface):
            async def generate_code(self, query, context=None, code_type="sql", conversation_history=None):
                return ""
            async def analyze_query(self, query, conversation_history=None):
                return {}
            async def format_response(self, results, query, conversation_history=None):
                return ""
            async def generate_clarification(self, query, ambiguities, conversation_history=None):
                return ""
            async def validate_generated_code(self, code, code_type, context=None):
                return {}
            async def get_model_info(self):
                return {}
        
        # These should not raise errors
        knowledge = CompleteKnowledge()
        executor = CompleteExecutor()
        llm = CompleteLLM()
        
        assert isinstance(knowledge, KnowledgePrimitive)
        assert isinstance(executor, ExecutorPrimitive)
        assert isinstance(llm, LLMInterface)