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
        The default azure openai url.
        It should be provided either
        through this config field or through config
        call AzureOpenAI.invoke()
        """
    )
    api_key: str = Field(
        default="api_key",
        description="""
        The default azure openai key.
        It should be provided either
        through this config field or through config
        call AzureOpenAI.invoke()
        """
    )
    temperature: float = Field(default=1)
    num_response: int = Field( # TODO how to generate multiple responses
        default=1, description="The number of llm response to be generated"
    )
    max_completion_tokens: int = Field(
        default=5000,
        description="The maximum output tokens", # TODO o1 requires a different attribute "max_completion_token"
    )
    frequency_penalty: float = Field(
        default=None, description="[-2, 2]. Penalty against repeating tokens."
    )
    oai_params: Optional[dict] = Field(default={})
    azure_model_params: Optional[dict] = Field(default={},)


class AzureOpenAI(BaseLLM):
    config_base_model = AzureOpenAIConfig
    config: AzureOpenAIConfig

    def _get_model(self, **kwargs):
        with_structured_output = kwargs.get(
            "with_structured_output", self.config.with_structured_output
        )
        llm = AzureChatOpenAI(
            azure_deployment=self.config.model,
            azure_endpoint=kwargs.get("base_url") or self.config.base_url,
            api_version=self.config.api_version,
            api_key=kwargs.get("api_key") or self.config.api_key,
            openai_api_type=self.config.api_type,
            n=self.config.num_response,
            temperature=self.config.temperature,
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
        logger.info(f"invoking llm with retry...")
        error_msgs = []
        # attempts to catch common exceptions raised that occur when invoking Azure
        for i in range(max_retry):
            try:
                response = await self._get_model(**kwargs).ainvoke(messages)
                if not kwargs.get(
                    "with_structured_output", self.config.with_structured_output
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
                            "input_tokens": response["raw"].usage_metadata[
                                "input_tokens"
                            ],
                            "output_token": response["raw"].usage_metadata[
                                "output_token"
                            ],
                        }
                else:
                    generation = response.content
                break
            except (
                ValueError,
                openai.BadRequestError,
                openai.AuthenticationError,
                openai.PermissionDeniedError,
                openai.APIError,
            ) as e:
                logger.exception(
                    f"error calling llm try {i + 1}", exc_info=e
                )
                error_msgs.append(e)
                error = LLMError(
                    error_code=0, error_type="LLM Errrpor", error_message=str(e)
                )
                break
            except Exception as e:
                logger.exception(
                    f"error calling llm try {i + 1}", exc_info=e
                )
                error_msgs.append(e)
                # record latest error
                error = LLMError(
                    error_code=0, error_type="LLM Error", error_message=str(e)
                )
                wait_time = (2**i) + random.random()
                logger.info(f"retrying after wait {wait_time}")
                time.sleep(wait_time)
                continue
        if error:
            logger.error(f"errors calling llm: {error_msgs}")
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
            llm = AzureOpenAIEmbeddings(
                openai_api_key=kwargs.get("openai_api_key"),
                openai_api_version=kwargs.get("openai_api_version"),
                azure_endpoint=kwargs.get("azure_endpoint"),
                model=kwargs.get("embedding_model_name"),
            )
            self.embedding_model_client = llm
        return self.embedding_model_client

    async def __call__(self, query: str, **kwargs):
        response = await self._get_model(**kwargs).aembed_query(query)
        return response