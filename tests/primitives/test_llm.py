"""Tests for LLM interface implementations."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.dataqa.config.models import LLMConfig, LLMProvider
from src.dataqa.exceptions import LLMError
from src.dataqa.models.message import Message
from src.dataqa.primitives.llm import OpenAILLM, create_llm_interface


class TestOpenAILLM:
    """Test cases for OpenAI LLM implementation."""

    @pytest.fixture
    def llm_config(self):
        """Create a test LLM configuration."""
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-api-key",
            temperature=0.1,
            max_tokens=1000,
            timeout=30.0,
            max_retries=3
        )

    @pytest.fixture
    def sample_messages(self):
        """Create sample conversation messages."""
        return [
            Message(
                role="user",
                content="What is the average sales by region?",
                timestamp=datetime.now()
            ),
            Message(
                role="assistant", 
                content="I'll help you analyze sales by region.",
                timestamp=datetime.now()
            )
        ]

    def test_init_valid_config(self, llm_config):
        """Test successful initialization with valid config."""
        llm = OpenAILLM(llm_config)
        assert llm.config == llm_config
        assert llm.client is not None

    def test_init_invalid_provider(self):
        """Test initialization fails with invalid provider."""
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key="test-key"
        )
        
        with pytest.raises(LLMError, match="Invalid provider for OpenAI LLM"):
            OpenAILLM(config)

    def test_init_missing_api_key(self):
        """Test initialization fails without API key."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key=None
        )
        
        with pytest.raises(LLMError, match="OpenAI API key is required"):
            OpenAILLM(config)

    def test_format_conversation_history_empty(self, llm_config):
        """Test formatting empty conversation history."""
        llm = OpenAILLM(llm_config)
        result = llm._format_conversation_history(None)
        assert result == "No previous conversation."
        
        result = llm._format_conversation_history([])
        assert result == "No previous conversation."

    def test_format_conversation_history_with_messages(self, llm_config, sample_messages):
        """Test formatting conversation history with messages."""
        llm = OpenAILLM(llm_config)
        result = llm._format_conversation_history(sample_messages)
        
        assert "User: What is the average sales by region?" in result
        assert "Assistant: I'll help you analyze sales by region." in result

    def test_build_messages(self, llm_config):
        """Test building message list for API."""
        llm = OpenAILLM(llm_config)
        messages = llm._build_messages("System prompt", "User prompt")
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User prompt"

    @pytest.mark.asyncio
    async def test_make_api_call_success(self, llm_config):
        """Test successful API call."""
        llm = OpenAILLM(llm_config)
        
        # Mock the OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated response"
        
        with patch.object(llm.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            messages = [{"role": "user", "content": "test"}]
            result = await llm._make_api_call(messages)
            
            assert result == "Generated response"
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_api_call_empty_response(self, llm_config):
        """Test API call with empty response."""
        llm = OpenAILLM(llm_config)
        
        mock_response = MagicMock()
        mock_response.choices = []
        
        with patch.object(llm.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            messages = [{"role": "user", "content": "test"}]
            
            with pytest.raises(LLMError, match="No response choices returned"):
                await llm._make_api_call(messages)

    @pytest.mark.asyncio
    async def test_make_api_call_rate_limit_error(self, llm_config):
        """Test API call with rate limit error."""
        llm = OpenAILLM(llm_config)
        
        with patch.object(llm.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            # Create a mock response object
            mock_response = MagicMock()
            mock_response.request = MagicMock()
            
            from openai import RateLimitError
            mock_create.side_effect = RateLimitError("Rate limit exceeded", response=mock_response, body=None)
            
            messages = [{"role": "user", "content": "test"}]
            
            with pytest.raises(LLMError, match="Rate limit exceeded"):
                await llm._make_api_call(messages)

    @pytest.mark.asyncio
    async def test_make_api_call_authentication_error(self, llm_config):
        """Test API call with authentication error."""
        llm = OpenAILLM(llm_config)
        
        with patch.object(llm.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            # Create a mock response object
            mock_response = MagicMock()
            mock_response.request = MagicMock()
            
            from openai import AuthenticationError
            mock_create.side_effect = AuthenticationError("Invalid API key", response=mock_response, body=None)
            
            messages = [{"role": "user", "content": "test"}]
            
            with pytest.raises(LLMError, match="Invalid API key or authentication failed"):
                await llm._make_api_call(messages)

    @pytest.mark.asyncio
    async def test_generate_code_sql_success(self, llm_config, sample_messages):
        """Test successful SQL code generation."""
        llm = OpenAILLM(llm_config)
        
        mock_response = "```sql\nSELECT region, AVG(sales) FROM sales_table GROUP BY region;\n```"
        
        with patch.object(llm, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            result = await llm.generate_code(
                query="What is the average sales by region?",
                context="Table: sales_table (region, sales)",
                code_type="sql",
                conversation_history=sample_messages
            )
            
            assert "SELECT region, AVG(sales)" in result
            assert "GROUP BY region" in result
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_code_python_success(self, llm_config):
        """Test successful Python code generation."""
        llm = OpenAILLM(llm_config)
        
        mock_response = "```python\nimport pandas as pd\ndf.groupby('region')['sales'].mean()\n```"
        
        with patch.object(llm, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            result = await llm.generate_code(
                query="Calculate average sales by region",
                code_type="python"
            )
            
            assert "import pandas as pd" in result
            assert "groupby('region')" in result
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_code_unsupported_type(self, llm_config):
        """Test code generation with unsupported code type."""
        llm = OpenAILLM(llm_config)
        
        with pytest.raises(LLMError, match="Unsupported code type"):
            await llm.generate_code(
                query="Test query",
                code_type="javascript"
            )

    @pytest.mark.asyncio
    async def test_generate_code_empty_response(self, llm_config):
        """Test code generation with empty response."""
        llm = OpenAILLM(llm_config)
        
        with patch.object(llm, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = ""
            
            with pytest.raises(LLMError, match="Generated code is empty"):
                await llm.generate_code(
                    query="Test query",
                    code_type="sql"
                )

    @pytest.mark.asyncio
    async def test_analyze_query_success(self, llm_config, sample_messages):
        """Test successful query analysis."""
        llm = OpenAILLM(llm_config)
        
        analysis_response = {
            "intent": "Calculate average sales by region",
            "query_type": "analysis",
            "entities": ["sales", "region"],
            "complexity": "simple",
            "requires_clarification": False,
            "suggested_approach": "Use GROUP BY to aggregate sales by region"
        }
        
        mock_response = f"```json\n{json.dumps(analysis_response)}\n```"
        
        with patch.object(llm, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            result = await llm.analyze_query(
                query="What is the average sales by region?",
                conversation_history=sample_messages
            )
            
            assert result["intent"] == "Calculate average sales by region"
            assert result["query_type"] == "analysis"
            assert "sales" in result["entities"]
            assert result["complexity"] == "simple"
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_query_invalid_json(self, llm_config):
        """Test query analysis with invalid JSON response."""
        llm = OpenAILLM(llm_config)
        
        with patch.object(llm, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Invalid JSON response"
            
            result = await llm.analyze_query("Test query")
            
            # Should return fallback analysis
            assert result["intent"] == "Data analysis request"
            assert result["query_type"] == "analysis"
            assert result["complexity"] == "moderate"

    @pytest.mark.asyncio
    async def test_format_response_success(self, llm_config, sample_messages):
        """Test successful response formatting."""
        llm = OpenAILLM(llm_config)
        
        mock_response = "The analysis shows that the average sales by region are: North: $50K, South: $45K, East: $55K, West: $48K."
        
        with patch.object(llm, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            results = {
                "data": [
                    {"region": "North", "avg_sales": 50000},
                    {"region": "South", "avg_sales": 45000}
                ]
            }
            
            result = await llm.format_response(
                results=results,
                query="What is the average sales by region?",
                conversation_history=sample_messages
            )
            
            assert "average sales by region" in result
            assert "North: $50K" in result
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_clarification_success(self, llm_config, sample_messages):
        """Test successful clarification generation."""
        llm = OpenAILLM(llm_config)
        
        mock_response = "Could you please clarify:\n1. Which time period should I analyze?\n2. Do you want to include all product categories?"
        
        with patch.object(llm, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            result = await llm.generate_clarification(
                query="Show me the sales data",
                ambiguities=["Time period not specified", "Product category unclear"],
                conversation_history=sample_messages
            )
            
            assert "time period" in result.lower()
            assert "product categories" in result.lower()
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_generated_code_success(self, llm_config):
        """Test successful code validation."""
        llm = OpenAILLM(llm_config)
        
        validation_response = {
            "is_valid": True,
            "issues": [],
            "security_concerns": [],
            "suggestions": ["Consider adding error handling"],
            "risk_level": "low"
        }
        
        mock_response = f"```json\n{json.dumps(validation_response)}\n```"
        
        with patch.object(llm, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            result = await llm.validate_generated_code(
                code="SELECT * FROM users WHERE id = 1;",
                code_type="sql",
                context="Database schema information"
            )
            
            assert result["is_valid"] is True
            assert result["risk_level"] == "low"
            assert len(result["suggestions"]) == 1
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_generated_code_unsupported_type(self, llm_config):
        """Test code validation with unsupported code type."""
        llm = OpenAILLM(llm_config)
        
        with pytest.raises(LLMError, match="Unsupported code type for validation"):
            await llm.validate_generated_code(
                code="console.log('test');",
                code_type="javascript"
            )

    @pytest.mark.asyncio
    async def test_validate_generated_code_invalid_json(self, llm_config):
        """Test code validation with invalid JSON response."""
        llm = OpenAILLM(llm_config)
        
        with patch.object(llm, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Invalid JSON response"
            
            result = await llm.validate_generated_code(
                code="SELECT * FROM users;",
                code_type="sql"
            )
            
            # Should return fallback validation
            assert result["is_valid"] is True
            assert result["risk_level"] == "low"
            assert result["issues"] == []

    @pytest.mark.asyncio
    async def test_get_model_info(self, llm_config):
        """Test getting model information."""
        llm = OpenAILLM(llm_config)
        
        info = await llm.get_model_info()
        
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4"
        assert info["temperature"] == 0.1
        assert info["max_tokens"] == 1000
        assert "code_generation" in info["capabilities"]
        assert "query_analysis" in info["capabilities"]


class TestCreateLLMInterface:
    """Test cases for LLM interface factory function."""

    def test_create_openai_interface(self):
        """Test creating OpenAI interface."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key"
        )
        
        interface = create_llm_interface(config)
        assert isinstance(interface, OpenAILLM)

    def test_create_unsupported_interface(self):
        """Test creating interface with unsupported provider."""
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key="test-key"
        )
        
        with pytest.raises(LLMError, match="Unsupported LLM provider"):
            create_llm_interface(config)


@pytest.mark.integration
class TestOpenAILLMIntegration:
    """Integration tests for OpenAI LLM (requires API key)."""

    @pytest.fixture
    def real_llm_config(self):
        """Create a real LLM configuration (requires OPENAI_API_KEY env var)."""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY environment variable not set")
        
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",  # Use cheaper model for testing
            api_key=api_key,
            temperature=0.1,
            max_tokens=100
        )

    @pytest.mark.asyncio
    async def test_real_code_generation(self, real_llm_config):
        """Test real code generation with OpenAI API."""
        llm = OpenAILLM(real_llm_config)
        
        result = await llm.generate_code(
            query="Select all users from users table",
            code_type="sql"
        )
        
        assert "SELECT" in result.upper()
        assert "users" in result.lower()

    @pytest.mark.asyncio
    async def test_real_query_analysis(self, real_llm_config):
        """Test real query analysis with OpenAI API."""
        llm = OpenAILLM(real_llm_config)
        
        result = await llm.analyze_query(
            query="What are the top 5 products by sales?"
        )
        
        assert "intent" in result
        assert "query_type" in result
        assert isinstance(result["entities"], list)