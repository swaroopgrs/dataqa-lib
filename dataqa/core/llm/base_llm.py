from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from langchain_core.messages.utils import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel, Field

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM]]
_StrOrDictOrPydantic = Union[str, Dict[str, Any], Type[_BM], List]


class LLMConfig(BaseModel):
    model: str = Field(
        description="The model name, such as deployment_name for oai llm, such as `gpt-4o-2024-05-06`, but NOT model_name like `gpt-4o`"
    )
    with_structured_output: Optional[Union[None, _DictOrPydanticClass]] = Field(
        default=None,
        description="""
        Parse raw llm generations to structured output.
        The input is a dict or a BaseModel class.
        """,
    )
    # with_tools TODO


class LLMMetaData(BaseModel):
    request_id: str = Field(
        description="A unique identifier for this LLM call. Usually provided by the LLM provider."
    )
    model: str = Field(description="The model name used in this LLM call.")
    num_generations: int = Field(
        default=1,
        description="The number of generations requested in this LLM call.",
    )
    num_retries: int = Field(
        description="The number of retries in this LLM call."
    )
    start_timestamp: Union[None, str] = Field(
        default=None,
        description="The timestamp to send request. The preferred format is `%a, %d %b %Y %H:%M:%S %Z`, e.g. `Tue, 04 Mar 2025 20:54:30 GMT`",
    )
    end_timestamp: Union[None, str] = Field(
        default=None,
        description="The timestamp to receive response. In the same format as `start timestamp`",
    )
    latency: Union[None, float] = Field(
        default=None,
        description="The latency between start and end timestamps in milliseconds",
    )
    input_token: int = Field(description="The number of input tokens")
    output_token: int = Field(
        description="The number of LLM completion tokens summed over all responses"
    )
    reasoning_token: Optional[int] = Field(
        default=0,
        description="The number of reasoning tokens. Use for reasoning models only such as GPT-O1.",
    )
    cost: Union[None, float] = Field(
        default=None, description="The cost of this LLM call in dollars."
    )
    ratelimit_tokens: Union[None, int] = Field(
        default=None,
        description="The maximum number of tokens to reach rate limit",
    )
    ratelimit_requests: Union[None, int] = Field(
        default=None,
        description="The maximum number of requests to reach rate limit",
    )
    ratelimit_remaining_tokens: Union[None, int] = Field(
        default=None,
        description="The number of remaining tokens to reach rate limit. By default",
    )
    ratelimit_remaining_requests: Union[None, int] = Field(
        default=None,
        description="The number of remaining requests to reach rate limit",
    )


class LLMError(BaseModel):
    error_code: int
    error_type: str
    error_message: str


class LLMOutput(BaseModel):
    prompt: _StrOrDictOrPydantic = Field(description="The input prompt")
    generation: _StrOrDictOrPydantic = Field(
        default="",
        description="""
        The LLM generations.
        Parsed to Dict or Pydantic BaseModel is the structured output is required.
        """,
    )
    from_component: Optional[str] = Field(
        default="",
        description="""
        The name of component that triggers this LLM call.
        Set to empty if the component name is provided.
        """,
    )
    metadata: Union[None, LLMMetaData] = Field(
        default=None, description="Token usage, cost, latency, ratelimit, ..."
    )
    error: Optional[LLMError] = None


class BaseLLM(RunnableCallable, ABC):
    def __init__(self, config: Union[LLMConfig, Dict] = None, **kwargs):
        self.config = config
        if isinstance(config, Dict):
            self.config = self.config_base_model(**kwargs)
        if self.config is None:
            self.config = self.config_base_model(**kwargs)
        super().__init__(
            func=self._func,
            afunc=self._afunc,
            name="base_retry_node",
            trace=False,
            **kwargs,
        )

    @property
    @abstractmethod
    def config_base_model(self):
        raise NotImplementedError

    def invoke(self, messages: List[AnyMessage], **kwargs) -> LLMOutput:
        raise NotImplementedError

    async def ainvoke(self, messages: List[AnyMessage], **kwargs) -> LLMOutput:
        raise NotImplementedError

    def stream(self, messages: List[AnyMessage], **kwargs):
        raise NotImplementedError

    async def astream(self, messages: List[AnyMessage], **kwargs):
        raise NotImplementedError

    def _func(
        self,
        input: Union[List[AnyMessage], dict[str, Any]],
        config: RunnableConfig,
    ) -> Any:
        raise NotImplementedError("_func not implemented")

    async def _afunc(
        self,
        input: Union[List[AnyMessage], dict[str, Any]],
        config: RunnableConfig,
    ) -> Any:
        raise NotImplementedError("_afunc not implemented")
