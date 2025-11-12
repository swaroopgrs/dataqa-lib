import types
from typing import Any, Callable, Optional, Type

# from langchain_core.callbacks import (
#     AsyncCallbackManagerForLLMRun,
#     CallbackManagerForLLMRun,
# )
from langchain_core.language_models import BaseChatModel

# from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# from langchain_core.prompt_values import ChatPromptValue
# from langchain_core.runnables import Runnable
# from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

from dataqa.core.llm.base_llm import BaseLLM, LLMConfig, LLMOutput
from dataqa.core.utils.prompt_utils import messages_to_serializable


class DBCLLMAdapter(BaseLLM):
    """
    The adapter that creates the DBCProxyChatModel.
    """

    @property
    def config_base_model(self) -> Type[LLMConfig]:
        return LLMConfig

    def __init__(self, llm_callable: Callable, model_name: str = "dbc_model"):
        super().__init__(config=LLMConfig(model=model_name))
        self.llm_callable = llm_callable
        self._proxy_model: Optional[BaseChatModel] = None

    def _get_model(self, **kwargs) -> BaseChatModel:
        """
        Returns an instance of our custom DBCProxyChatModel.
        """
        if self._proxy_model is None:
            # The delegate is a real langchain model. It's used to correctly
            # format tool and structured output calls, but its own `invoke`
            # method will never be called
            delegate = AzureChatOpenAI(
                model="gpt-4.1-2025-04-14",
                openai_api_key="placeholder",
                openai_api_type="azure_ad",
                temperature=0,
                azure_endpoint="https://placeholder.openai.azure.com",
                api_version="2024-08-01-preview",
            )

            async def custom_agenerate(
                self_instance, messages, stop, run_manager, **kwargs
            ):
                try:
                    response_message = await self.llm_callable(
                        model=self_instance.deployment_name,
                        messages=messages,
                        **kwargs,
                    )
                    generation = ChatGeneration(message=response_message)
                    return ChatResult(generations=[generation])
                except Exception as e:
                    print(f"{e}")
                    raise

            delegate._agenerate = types.MethodType(custom_agenerate, delegate)

            self._proxy_model = delegate
        return self._proxy_model

    async def ainvoke(self, messages: Any, **kwargs) -> LLMOutput:
        """
        Single unified method for all LLM calls. Uses the proxy model to handle everything consistently.
        """
        # t = time.time()
        from_component = kwargs.pop("from_component", "")
        try:
            # get the base proxy model
            model = self._get_model()
            # dbc_kwargs = {k:v for k, v in kwargs.items() if k not in}

            # apply structured output if requested for this specific call
            with_structured_output = kwargs.pop(
                "with_structured_output", self.config.with_structured_output
            )
            if with_structured_output:
                model = model.with_structured_output(
                    with_structured_output,
                    # include_raw=True,
                    method="json_schema",
                )

            # invoke the model
            response = await model.ainvoke(messages)
            if isinstance(response, BaseModel):
                generation = response
            elif hasattr(response, "tool_calls") and response.tool_calls:
                generation = response.tool_calls[0]["args"]
            else:
                generation = response.content

            # metadata = {
            #     # "request_id": response["raw"].id,
            #     # "model": response["raw"].response_metadata[
            #     #     "model_name"
            #     # ],
            #     # "latency": time.time() - t,
            #     # "input_token": response["raw"].usage_metadata[
            #     #     "input_tokens"
            #     # ],
            #     # "output_token": response["raw"].usage_metadata[
            #     #     "output_tokens"
            #     # ],
            # }

            return LLMOutput(
                prompt=messages_to_serializable(messages),
                generation=generation,
                # metadata=metadata,
                from_component=from_component,
            )
        except Exception as e:
            return LLMOutput(
                prompt=messages_to_serializable(messages),
                generation="",
                error={
                    "error_code": 500,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
                from_component=from_component,
            )
