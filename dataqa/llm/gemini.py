# dataqa/llm/gemini.py

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field

from dataqa.llm.base_llm import (
    BaseLLM,
    LLMConfig,
    LLMError,
    LLMMetaData,
    LLMOutput,
)
from dataqa.utils.prompt_utils import messages_to_serializable

logger = logging.getLogger(__name__)


class GeminiLLMConfig(LLMConfig):
    api_key: str = Field(description="The API key for the Gemini model.")
    temperature: float = Field(default=0.0)


class GeminiLLM(BaseLLM):
    config_base_model = GeminiLLMConfig
    config: GeminiLLMConfig

    def _get_model(self, **kwargs) -> ChatGoogleGenerativeAI:
        """Initializes and returns the LangChain ChatGoogleGenerativeAI model client."""
        llm = ChatGoogleGenerativeAI(
            model=self.config.model,
            google_api_key=self.config.api_key,
            temperature=self.config.temperature,
            convert_system_message_to_human=True, # Gemini doesn't have a distinct system role
        )

        with_structured_output = kwargs.get(
            "with_structured_output", self.config.with_structured_output
        )
        if with_structured_output:
            llm = llm.with_structured_output(with_structured_output, method="json")

        return llm

    async def ainvoke(self, messages: List[Any], max_retry: int = 3, **kwargs) -> LLMOutput:
        t_start = time.time()
        from_component = kwargs.get("from_component", "")
        generation = ""
        metadata = None
        error = None

        try:
            model = self._get_model(**kwargs)
            response = await model.ainvoke(messages)
            
            generation = response if kwargs.get("with_structured_output") else response.content

            # Extract metadata if available
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                 metadata = LLMMetaData(
                    request_id=response.id if hasattr(response, 'id') else 'N/A',
                    model=self.config.model,
                    num_retries=0, # Langchain handles retries, not easily visible here
                    latency=time.time() - t_start,
                    input_token=response.usage_metadata.get("prompt_token_count", 0),
                    output_token=response.usage_metadata.get("candidates_token_count", 0),
                 )

        except Exception as e:
            logger.error(f"Gemini LLM call failed: {e}", exc_info=True)
            error = LLMError(error_code=500, error_type=type(e).__name__, error_message=str(e))
            generation = f"LLM call failed with error: {e}"

        return LLMOutput(
            prompt=messages_to_serializable(messages),
            generation=generation,
            from_component=from_component,
            metadata=metadata,
            error=error,
        )