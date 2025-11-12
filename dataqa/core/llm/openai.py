import logging
import random
import time
from typing import Any, Dict, Optional

import openai
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import Field

from dataqa.core.llm.base_llm import (
    BaseLLM,
    LLMConfig,
    LLMError,
    LLMOutput,
)
from dataqa.core.utils.prompt_utils import messages_to_serializable

logger = logging.getLogger(__name__)


class AzureOpenAIConfig(LLMConfig):
    api_version: str
    api_type: str
    base_url: str = Field(
        default="base_url",
        description="""
            The base URL of AzureOpenAI.
            It should be provided either
              - when define AzureOpenAIConfig
              - call AzureOpenAI.invoke()
        """,
    )
    api_key: str = Field(
        default="api_key",
        description="""
            The API_KEY of AzureOpenAI.
            It should be provided either
              - when define AzureOpenAIConfig
              - call AzureOpenAI.invoke()
        """,
    )
    temperature: float = Field(default=1)
    num_response: int = Field(  # TODO how to generate multiple responses
        default=1, description="The number of LLM response to be generated"
    )
    max_completion_tokens: int = Field(
        default=5000,
        description="The maximum output tokens",  # TODO o1 requires a different attribute "max_completion_token"
    )
    frequency_penalty: float = Field(
        default=0, description="[-2, 2]. Penalty against repeating tokens."
    )
    oai_params: Optional[Dict[str, Any]] = Field(default={})
    azure_model_params: Optional[Dict[str, Any]] = Field(default={})


class AzureOpenAI(BaseLLM):
    config_base_model = AzureOpenAIConfig
    config: AzureOpenAIConfig

    def _get_model(self, **kwargs):
        with_structured_output = kwargs.get(
            "with_structured_output", self.config.with_structured_output
        )
        if kwargs.get("token") is not None and kwargs.get("token") != "":
            llm = AzureChatOpenAI(
                azure_endpoint=kwargs.get("base_url") or self.config.base_url,
                azure_deployment=self.config.model,
                api_key=kwargs.get("api_key") or self.config.api_key,
                api_version=self.config.api_version,
                default_headers={
                    "Authorization": f"Bearer {kwargs.get('token')}"
                },
                openai_api_type=self.config.api_type,
                temperature=self.config.temperature,
                n=self.config.num_response,
                include_response_headers=with_structured_output is None,
                frequency_penalty=self.config.frequency_penalty,
                model_kwargs={
                    "max_completion_tokens": self.config.max_completion_tokens,
                },
                **self.config.oai_params,
                **self.config.azure_model_params,
            )
        else:
            llm = AzureChatOpenAI(
                azure_endpoint=kwargs.get("base_url") or self.config.base_url,
                azure_deployment=self.config.model,
                api_key=kwargs.get("api_key") or self.config.api_key,
                api_version=self.config.api_version,
                openai_api_type=self.config.api_type,
                temperature=self.config.temperature,
                n=self.config.num_response,
                include_response_headers=with_structured_output is None,
                frequency_penalty=self.config.frequency_penalty,
                model_kwargs={
                    "max_completion_tokens": self.config.max_completion_tokens,
                },
                **self.config.oai_params,
                **self.config.azure_model_params,
            )

        if with_structured_output is not None:
            llm = llm.with_structured_output(
                with_structured_output,
                include_raw=True,
                method="json_schema",
            )
        return llm

    async def ainvoke(self, messages, max_retry: int = 5, **kwargs):
        t = time.time()
        from_component = kwargs.get("from_component", "")

        generation = ""
        metadata = None
        error = None
        logger.info("invoking llm with retry...")
        error_msgs = []
        """Attempts to catch common exceptions raised that occur when invoking Azure"""
        for i in range(max_retry):
            try:
                response = await self._get_model(**kwargs).ainvoke(messages)
                if (
                    kwargs.get(
                        "with_structured_output",
                        self.config.with_structured_output,
                    )
                    is not None
                ):
                    if response["parsing_error"]:
                        generation = str(response)
                    else:
                        generation = response["parsed"]
                        metadata = {
                            "request_id": response["raw"].id,
                            "model": response["raw"].response_metadata[
                                "model_name"
                            ],
                            "latency": time.time() - t,
                            "num_retries": i,
                            "input_token": response["raw"].usage_metadata[
                                "input_tokens"
                            ],
                            "output_token": response["raw"].usage_metadata[
                                "output_tokens"
                            ],
                        }
                else:
                    generation = response.content
                break
            # Break on non-recoverable exceptions
            except (
                ValueError,
                openai.BadRequestError,
                openai.AuthenticationError,
                openai.APIConnectionError,
                openai.PermissionDeniedError,
                openai.APIError,
            ) as e:  # ValueErrors can be Langchain during content filter
                logger.exception(
                    f"exception calling LLM try {i + 1}", exc_info=e
                )
                error_msgs.append(e)
                error = LLMError(
                    error_code=0, error_type="LLM Error", error_message=str(e)
                )
                break
            # Retry on recoverable exceptions (RateLimit, APITimeOut etc.)
            except Exception as e:
                logger.exception(
                    f"exception calling LLM try {i + 1}", exc_info=e
                )
                error_msgs.append(e)
                # record latest error
                error = LLMError(
                    error_code=0, error_type="LLM Error", error_message=str(e)
                )
                wait_time = (2**i) + random.random()
                logger.info(f"Wait {wait_time}s before retry")
                time.sleep(wait_time)
                continue

        if error:
            logger.error("errors calling LLM %s", error_msgs)
        return LLMOutput(
            prompt=messages_to_serializable(messages),
            generation=generation,
            from_component=from_component,
            metadata=metadata,
            error=error,
        )


class OpenAIEmbedding:
    embedding_model_client = None

    def _get_model(self, **kwargs):
        if self.embedding_model_client is None:
            if (
                kwargs.get("openai_api_token") is not None
                and kwargs.get("openai_api_token") != ""
            ):
                llm = AzureOpenAIEmbeddings(
                    openai_api_key=kwargs.get("openai_api_key"),
                    openai_api_version=kwargs.get("openai_api_version"),
                    default_headers={
                        "Authorization": f"Bearer {kwargs.get('openai_api_token')}"
                    },
                    azure_endpoint=kwargs.get("azure_endpoint"),
                    model=kwargs.get("embedding_model_name"),
                )
            else:
                llm = AzureOpenAIEmbeddings(
                    openai_api_key=kwargs.get("openai_api_key"),
                    openai_api_version=kwargs.get("openai_api_version"),
                    azure_endpoint=kwargs.get("azure_endpoint"),
                    model=kwargs.get("embedding_model_name"),
                )
            self.embedding_model_client = llm
        return self.embedding_model_client

    async def __call__(self, query: str, **kwargs):
        res = await self._get_model(**kwargs).aembed_query(query)
        return res
