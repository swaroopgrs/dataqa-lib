"""Abstract base class for LLM interface implementations."""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

import openai
from openai import AsyncOpenAI

from ..config.models import LLMConfig, LLMProvider
from ..exceptions import LLMError, RetryableError
from ..logging_config import get_primitive_logger
from ..models.message import Message
from ..utils.retry import retry_async, retry_on_rate_limit


class LLMInterface(ABC):
    """Abstract base class for Large Language Model interfaces.

    This interface defines the contract for components that interact
    with language models to generate code, analyze queries, and
    provide natural language responses. Implementations might use
    OpenAI, Anthropic, local models, or other LLM providers.
    """

    @abstractmethod
    async def generate_code(
        self,
        query: str,
        context: str | None = None,
        code_type: str = "sql",
        conversation_history: list[Message] | None = None
    ) -> str:
        """Generate code based on a natural language query.

        Args:
            query: The natural language query from the user
            context: Optional context information (schema, examples, etc.)
            code_type: Type of code to generate ('sql', 'python')
            conversation_history: Previous messages for context

        Returns:
            Generated code as a string

        Raises:
            LLMError: If code generation fails
        """
        pass

    @abstractmethod
    async def analyze_query(
        self,
        query: str,
        conversation_history: list[Message] | None = None
    ) -> dict[str, Any]:
        """Analyze a user query to understand intent and requirements.

        Args:
            query: The natural language query from the user
            conversation_history: Previous messages for context

        Returns:
            Dictionary containing analysis results (intent, entities, etc.)

        Raises:
            LLMError: If query analysis fails
        """
        pass

    @abstractmethod
    async def format_response(
        self,
        results: dict[str, Any],
        query: str,
        conversation_history: list[Message] | None = None
    ) -> str:
        """Format execution results into a natural language response.

        Args:
            results: The results from code execution
            query: The original user query
            conversation_history: Previous messages for context

        Returns:
            Formatted natural language response

        Raises:
            LLMError: If response formatting fails
        """
        pass

    @abstractmethod
    async def generate_clarification(
        self,
        query: str,
        ambiguities: list[str],
        conversation_history: list[Message] | None = None
    ) -> str:
        """Generate clarifying questions for ambiguous queries.

        Args:
            query: The ambiguous query from the user
            ambiguities: List of identified ambiguities
            conversation_history: Previous messages for context

        Returns:
            Clarifying question(s) as a string

        Raises:
            LLMError: If clarification generation fails
        """
        pass

    @abstractmethod
    async def validate_generated_code(
        self,
        code: str,
        code_type: str,
        context: str | None = None
    ) -> dict[str, Any]:
        """Validate generated code for correctness and safety.

        Args:
            code: The generated code to validate
            code_type: Type of code ('sql', 'python')
            context: Optional context for validation

        Returns:
            Dictionary containing validation results (is_valid, issues, etc.)

        Raises:
            LLMError: If validation fails
        """
        pass

    @abstractmethod
    async def get_model_info(self) -> dict[str, Any]:
        """Get information about the underlying model.

        Returns:
            Dictionary containing model information (name, version, capabilities)
        """
        pass


class OpenAILLM(LLMInterface):
    """OpenAI LLM implementation with structured prompts and context injection."""

    def __init__(self, config: LLMConfig):
        """Initialize OpenAI LLM with configuration.

        Args:
            config: LLM configuration containing API key, model, etc.

        Raises:
            LLMError: If configuration is invalid or API key is missing
        """
        if config.provider != LLMProvider.OPENAI:
            raise LLMError(
                f"Invalid provider for OpenAI LLM: {config.provider}",
                user_message="Configuration error: Invalid LLM provider specified.",
                error_code="INVALID_LLM_PROVIDER"
            )

        if not config.api_key:
            raise LLMError(
                "OpenAI API key is required",
                user_message="OpenAI API key is not configured. Please set your API key.",
                error_code="MISSING_API_KEY",
                recovery_suggestions=[
                    "Set the OPENAI_API_KEY environment variable",
                    "Check your configuration file for the correct API key setting"
                ]
            )

        self.config = config
        self.logger = get_primitive_logger("llm", f"openai-{config.model}")
        
        try:
            self.client = AsyncOpenAI(
                api_key=config.api_key.get_secret_value(),
                base_url=config.api_base,
                timeout=config.timeout,
                max_retries=config.max_retries
            )
            self.logger.info(f"OpenAI LLM initialized with model {config.model}")
        except Exception as e:
            raise LLMError(
                f"Failed to initialize OpenAI client: {e}",
                user_message="Failed to connect to OpenAI service. Please check your configuration.",
                error_code="CLIENT_INIT_FAILED",
                original_error=e
            )

        # Prompt templates for different operations
        self.code_generation_template = """You are a data analysis expert. Generate {code_type} code to answer the user's question.

Context Information:
{context}

Conversation History:
{history}

User Query: {query}

Requirements:
- Generate only the {code_type} code, no explanations
- Ensure the code is safe and follows best practices
- Use appropriate error handling
- For SQL: Use standard SQL syntax compatible with DuckDB
- For Python: Use pandas for data manipulation, matplotlib/seaborn for visualization

Generated {code_type_upper} Code:"""

        self.query_analysis_template = """Analyze the following user query to understand their intent and requirements.

Conversation History:
{history}

User Query: {query}

Provide analysis in JSON format with the following structure:
{{
    "intent": "description of what the user wants to accomplish",
    "query_type": "data_retrieval|analysis|visualization|modification",
    "entities": ["list", "of", "key", "entities"],
    "complexity": "simple|moderate|complex",
    "requires_clarification": true/false,
    "suggested_approach": "brief description of recommended approach"
}}

Analysis:"""

        self.response_formatting_template = """Format the execution results into a clear, natural language response for the user.

Original Query: {query}

Execution Results:
{results}

Conversation History:
{history}

Requirements:
- Provide a clear, concise summary of the results
- Highlight key insights or patterns
- Use natural language that's easy to understand
- If there are visualizations, describe what they show
- If there are errors, explain them in user-friendly terms

Response:"""

        self.clarification_template = """The user's query is ambiguous. Generate clarifying questions to better understand their needs.

User Query: {query}

Identified Ambiguities:
{ambiguities}

Conversation History:
{history}

Generate 1-3 specific clarifying questions that will help resolve the ambiguities:

Questions:"""

        self.code_validation_template = """Validate the following {code_type} code for correctness and safety.

Code to Validate:
{code}

Context:
{context}

Provide validation results in JSON format:
{{
    "is_valid": true/false,
    "issues": ["list", "of", "identified", "issues"],
    "security_concerns": ["list", "of", "security", "issues"],
    "suggestions": ["list", "of", "improvement", "suggestions"],
    "risk_level": "low|medium|high"
}}

Validation Results:"""

    def _format_conversation_history(self, history: list[Message] | None) -> str:
        """Format conversation history for prompt inclusion."""
        if not history:
            return "No previous conversation."

        formatted = []
        for msg in history[-5:]:  # Include last 5 messages for context
            role = msg.role.title()
            formatted.append(f"{role}: {msg.content}")

        return "\n".join(formatted)

    def _build_messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        """Build message list for OpenAI API."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    @retry_on_rate_limit(max_attempts=3, base_delay=2.0)
    async def _make_api_call(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Make API call with error handling and retries."""
        try:
            self.logger.debug(f"Making OpenAI API call with model {self.config.model}")
            
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **self.config.extra_params,
                **kwargs
            )

            if not response.choices:
                raise LLMError(
                    "No response choices returned from OpenAI API",
                    user_message="The AI service didn't provide a response. Please try again.",
                    error_code="NO_RESPONSE_CHOICES"
                )

            content = response.choices[0].message.content
            if not content:
                raise LLMError(
                    "Empty response content from OpenAI API",
                    user_message="The AI service provided an empty response. Please try again.",
                    error_code="EMPTY_RESPONSE"
                )

            self.logger.debug(f"Received response with {len(content)} characters")
            return content.strip()

        except openai.RateLimitError as e:
            self.logger.warning(f"Rate limit hit: {e}")
            raise RetryableError(
                f"Rate limit exceeded: {e}",
                user_message="The AI service is currently busy. Please wait a moment and try again.",
                error_code="RATE_LIMIT_EXCEEDED",
                retry_after=60.0,
                max_retries=5,
                original_error=e
            )

        except openai.APITimeoutError as e:
            self.logger.error(f"API timeout: {e}")
            raise RetryableError(
                f"API request timed out: {e}",
                user_message="The AI service took too long to respond. Please try again.",
                error_code="API_TIMEOUT",
                retry_after=5.0,
                max_retries=3,
                original_error=e
            )

        except openai.APIConnectionError as e:
            self.logger.error(f"API connection error: {e}")
            raise RetryableError(
                f"Failed to connect to OpenAI API: {e}",
                user_message="Unable to connect to the AI service. Please check your internet connection.",
                error_code="CONNECTION_ERROR",
                retry_after=10.0,
                max_retries=3,
                original_error=e
            )

        except openai.AuthenticationError as e:
            self.logger.error(f"Authentication error: {e}")
            raise LLMError(
                f"Invalid API key or authentication failed: {e}",
                user_message="Authentication failed. Please check your API key configuration.",
                error_code="AUTH_ERROR",
                recovery_suggestions=[
                    "Verify your OpenAI API key is correct",
                    "Check if your API key has sufficient permissions",
                    "Ensure your account has available credits"
                ],
                original_error=e
            )

        except openai.BadRequestError as e:
            self.logger.error(f"Bad request error: {e}")
            raise LLMError(
                f"Invalid request to OpenAI API: {e}",
                user_message="The request format is invalid. Please try rephrasing your question.",
                error_code="BAD_REQUEST",
                recovery_suggestions=[
                    "Try a shorter or simpler question",
                    "Check if your input contains any unusual characters"
                ],
                original_error=e
            )

        except Exception as e:
            self.logger.error(f"Unexpected error in OpenAI API call: {e}")
            raise LLMError(
                f"Unexpected error: {e}",
                user_message="An unexpected error occurred with the AI service. Please try again.",
                error_code="UNEXPECTED_ERROR",
                original_error=e
            )

    async def generate_code(
        self,
        query: str,
        context: str | None = None,
        code_type: str = "sql",
        conversation_history: list[Message] | None = None
    ) -> str:
        """Generate code based on a natural language query."""
        if code_type.lower() not in ["sql", "python"]:
            raise LLMError(f"Unsupported code type: {code_type}")

        history_str = self._format_conversation_history(conversation_history)
        context_str = context or "No additional context provided."

        prompt = self.code_generation_template.format(
            code_type=code_type.lower(),
            code_type_upper=code_type.upper(),
            context=context_str,
            history=history_str,
            query=query
        )

        system_prompt = f"You are an expert data analyst specializing in {code_type.upper()} code generation."
        messages = self._build_messages(system_prompt, prompt)

        try:
            response = await self._make_api_call(messages)

            # Extract code from response (remove markdown formatting if present)
            code = response
            if f"```{code_type.lower()}" in code:
                code = code.split(f"```{code_type.lower()}")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()

            if not code:
                raise LLMError(
                    "Generated code is empty",
                    user_message="The AI service didn't generate any code. Please try rephrasing your question.",
                    error_code="EMPTY_CODE_GENERATION"
                )

            self.logger.info(f"Generated {code_type} code for query: {query[:50]}...")
            return code

        except Exception as e:
            self.logger.error(f"Failed to generate {code_type} code: {e}")
            raise LLMError(
                f"Code generation failed: {e}",
                user_message="Failed to generate code for your request. Please try a different approach.",
                error_code="CODE_GENERATION_FAILED",
                original_error=e
            )

    async def analyze_query(
        self,
        query: str,
        conversation_history: list[Message] | None = None
    ) -> dict[str, Any]:
        """Analyze a user query to understand intent and requirements."""
        history_str = self._format_conversation_history(conversation_history)

        prompt = self.query_analysis_template.format(
            history=history_str,
            query=query
        )

        system_prompt = "You are an expert at analyzing data-related queries and understanding user intent."
        messages = self._build_messages(system_prompt, prompt)

        try:
            response = await self._make_api_call(messages)

            # Extract JSON from response
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()

            try:
                analysis = json.loads(json_str)

                # Validate required fields
                required_fields = ["intent", "query_type", "entities", "complexity", "requires_clarification"]
                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = None

                self.logger.info(f"Analyzed query: {query[:50]}... -> {analysis['query_type']}")
                return analysis

            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON response, using fallback: {e}")
                return {
                    "intent": "Data analysis request",
                    "query_type": "analysis",
                    "entities": [],
                    "complexity": "moderate",
                    "requires_clarification": False,
                    "suggested_approach": "Generate appropriate code to answer the query"
                }

        except Exception as e:
            self.logger.error(f"Failed to analyze query: {e}")
            raise LLMError(
                f"Query analysis failed: {e}",
                user_message="Failed to analyze your question. Please try rephrasing it.",
                error_code="QUERY_ANALYSIS_FAILED",
                original_error=e
            )

    async def format_response(
        self,
        results: dict[str, Any],
        query: str,
        conversation_history: list[Message] | None = None
    ) -> str:
        """Format execution results into a natural language response."""
        history_str = self._format_conversation_history(conversation_history)

        # Format results for prompt
        results_str = json.dumps(results, indent=2, default=str)

        prompt = self.response_formatting_template.format(
            query=query,
            results=results_str,
            history=history_str
        )

        system_prompt = "You are an expert at explaining data analysis results in clear, accessible language."
        messages = self._build_messages(system_prompt, prompt)

        try:
            response = await self._make_api_call(messages)
            self.logger.info(f"Formatted response for query: {query[:50]}...")
            return response

        except Exception as e:
            self.logger.error(f"Failed to format response: {e}")
            raise LLMError(
                f"Response formatting failed: {e}",
                user_message="Failed to format the response. Please try again.",
                error_code="RESPONSE_FORMATTING_FAILED",
                original_error=e
            )

    async def generate_clarification(
        self,
        query: str,
        ambiguities: list[str],
        conversation_history: list[Message] | None = None
    ) -> str:
        """Generate clarifying questions for ambiguous queries."""
        history_str = self._format_conversation_history(conversation_history)
        ambiguities_str = "\n".join(f"- {amb}" for amb in ambiguities)

        prompt = self.clarification_template.format(
            query=query,
            ambiguities=ambiguities_str,
            history=history_str
        )

        system_prompt = "You are an expert at identifying ambiguities and asking clarifying questions."
        messages = self._build_messages(system_prompt, prompt)

        try:
            response = await self._make_api_call(messages)
            self.logger.info(f"Generated clarification for query: {query[:50]}...")
            return response

        except Exception as e:
            self.logger.error(f"Failed to generate clarification: {e}")
            raise LLMError(
                f"Clarification generation failed: {e}",
                user_message="Failed to generate clarifying questions. Please try again.",
                error_code="CLARIFICATION_FAILED",
                original_error=e
            )

    async def validate_generated_code(
        self,
        code: str,
        code_type: str,
        context: str | None = None
    ) -> dict[str, Any]:
        """Validate generated code for correctness and safety."""
        if code_type.lower() not in ["sql", "python"]:
            raise LLMError(f"Unsupported code type for validation: {code_type}")

        context_str = context or "No additional context provided."

        prompt = self.code_validation_template.format(
            code_type=code_type.lower(),
            code=code,
            context=context_str
        )

        system_prompt = f"You are an expert at validating {code_type.upper()} code for correctness and security."
        messages = self._build_messages(system_prompt, prompt)

        try:
            response = await self._make_api_call(messages)

            # Extract JSON from response
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()

            try:
                validation = json.loads(json_str)

                # Ensure required fields exist
                required_fields = ["is_valid", "issues", "security_concerns", "suggestions", "risk_level"]
                for field in required_fields:
                    if field not in validation:
                        if field == "is_valid":
                            validation[field] = True
                        elif field == "risk_level":
                            validation[field] = "low"
                        else:
                            validation[field] = []

                self.logger.info(f"Validated {code_type} code: {validation['is_valid']}, risk: {validation['risk_level']}")
                return validation

            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse validation JSON, using fallback: {e}")
                return {
                    "is_valid": True,
                    "issues": [],
                    "security_concerns": [],
                    "suggestions": [],
                    "risk_level": "low"
                }

        except Exception as e:
            self.logger.error(f"Failed to validate code: {e}")
            raise LLMError(
                f"Code validation failed: {e}",
                user_message="Failed to validate the generated code. Please try again.",
                error_code="CODE_VALIDATION_FAILED",
                original_error=e
            )

    async def get_model_info(self) -> dict[str, Any]:
        """Get information about the underlying model."""
        return {
            "provider": "openai",
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
            "capabilities": [
                "code_generation",
                "query_analysis",
                "response_formatting",
                "clarification_generation",
                "code_validation"
            ]
        }


def create_llm_interface(config: LLMConfig) -> LLMInterface:
    """Factory function to create LLM interface based on configuration.

    Args:
        config: LLM configuration

    Returns:
        Appropriate LLM interface implementation

    Raises:
        LLMError: If provider is not supported
    """
    if config.provider == LLMProvider.OPENAI:
        return OpenAILLM(config)
    else:
        raise LLMError(f"Unsupported LLM provider: {config.provider}")
