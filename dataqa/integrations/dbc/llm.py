# dataqa/integrations/dbc/llm.py
import asyncio
from typing import Any, Callable, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI

from dataqa.core.llm.base_llm import BaseLLM, LLMConfig, LLMOutput
from dataqa.core.utils.prompt_utils import messages_to_serializable


class DBCProxyChatModel(SimpleChatModel):
    """
    A proxy LangChain Chat Model that wraps the DBC `llm_invoke_with_retries` function.
    """

    dbc_invoke_function: Callable = Field(
        ..., description="The async llm_invoke_with_retries function from DBC."
    )
    model_name: str = "dbc-proxy-model"
    delegate_model: BaseChatModel = Field(
        ...,
        description="A real chat model instance to delegate binding logic to.",
    )

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "dbc-proxy-chat-model"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        THE FIX IS HERE: Implement the required synchronous _call method.
        This method wraps the asynchronous _agenerate method.
        """
        # Create a new asyncio event loop to run the async code
        # in this synchronous context.
        result = asyncio.run(self._agenerate(messages, stop, None, **kwargs))
        return result.generations[0].message.content

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        The core async method that calls the DBC invocation function.
        """
        response_message = await self.dbc_invoke_function(
            model=self.model_name,
            messages=messages,
        )
        generation = ChatGeneration(message=response_message)
        return ChatResult(generations=[generation])

    def bind_tools(
        self,
        tools: List[BaseTool],
        **kwargs: Any,
    ) -> "DBCProxyChatModel":
        """
        Handles tool binding by delegating the complex logic to the internal delegate model.
        """
        tool_bound_delegate = self.delegate_model.bind_tools(tools, **kwargs)
        new_proxy = self.copy(update={"delegate_model": tool_bound_delegate})
        return new_proxy

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name}


class DBCLLMAdapter(BaseLLM):
    """
    The adapter that creates the DBCProxyChatModel.
    """

    config_base_model = LLMConfig

    def __init__(self, llm_callable: Callable, model_name: str = "dbc_model"):
        super().__init__(config=LLMConfig(model=model_name))
        self.llm_callable = llm_callable
        self._proxy_model: Optional[DBCProxyChatModel] = None

    def _get_model(self, **kwargs) -> BaseChatModel:
        """
        Returns an instance of our custom DBCProxyChatModel.
        """
        if self._proxy_model is None:
            delegate = AzureChatOpenAI(
                model="placeholder",
                api_key="placeholder",
                azure_endpoint="https://placeholder.openai.azure.com",
                api_version="placeholder",
            )

            self._proxy_model = DBCProxyChatModel(
                dbc_invoke_function=self.llm_callable,
                model_name=self.config.model,
                delegate_model=delegate,
            )
        return self._proxy_model

    async def ainvoke(self, messages: Any, **kwargs) -> LLMOutput:
        """
        For simple invocations (without tool binding).
        """
        try:
            response_message: BaseMessage = await self.llm_callable(
                model=self.config.model, messages=messages
            )
            generation = response_message.content

            return LLMOutput(
                prompt=messages_to_serializable(messages), generation=generation
            )
        except Exception as e:
            return LLMOutput(
                prompt=messages_to_serializable(messages),
                error={
                    "error_code": 500,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
