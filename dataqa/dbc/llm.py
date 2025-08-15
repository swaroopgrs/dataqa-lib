from typing import Any, Callable, List
from langchain_core.messages.utils import AnyMessage

from dataqa.llm.base_llm import BaseLLM, LLMConfig, LLMOutput
from dataqa.utils.prompt_utils import messages_to_serializable

class DBCLLMAdapter(BaseLLM):
    config_base_model = LLMConfig

    def __init__(self, llm_callable: Callable, model_name: str = "dbc_model"):
        """
        An adapter to make the DBC-provided LLM callable conform to the BaseLLM interface.
        
        Args:
            llm_callable: A function that takes `messages` and returns a response.
            model_name: An identifier for the model being used.
        """
        super().__init__(config=LLMConfig(model=model_name))
        self.llm_callable = llm_callable

    async def ainvoke(self, messages: List[AnyMessage], **kwargs) -> LLMOutput:
        """
        Invokes the DBC callable and wraps the result in an LLMOutput object.
        """
        # Note: This assumes the llm_callable is async. If not, it needs to be wrapped.
        # It also assumes the callable handles its own retries, timeouts, etc.
        try:
            # We assume the callable takes a list of message-like objects or dicts
            serialized_messages = messages_to_serializable(messages)
            response_content = await self.llm_callable(messages=serialized_messages)
            
            # Here, we assume the response is the raw string content.
            # This part may need adjustment based on the actual return type of the callable.
            generation = response_content

            return LLMOutput(
                prompt=serialized_messages,
                generation=generation
                # Metadata would be harder to get unless the callable returns it.
            )
        except Exception as e:
            # Package any exception into the LLMOutput error field
            return LLMOutput(
                prompt=messages_to_serializable(messages),
                error={"error_code": 500, "error_type": type(e).__name__, "error_message": str(e)}
            )