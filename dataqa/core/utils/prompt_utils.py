from typing import Dict, List, Literal, Union

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages.base import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel


class Prompt(BaseModel):
    role: Literal["system", "user", "assistant"] = "system"
    content: str


prompt_type = Union[
    str, Prompt, Dict, List[str], List[Prompt], List[Dict], ChatPromptTemplate
]


def messages_to_serializable(messages: LanguageModelInput) -> List:
    if isinstance(messages, Dict) and "raw" in messages:
        messages = messages["raw"]
    if isinstance(messages, str):
        return [messages]
    output = []
    if isinstance(messages, PromptValue):
        messages = messages.to_messages()
    for msg in messages:
        if isinstance(msg, BaseMessage):
            output.append(msg.to_json()["kwargs"])
        else:
            output.append(msg)
    return output


def build_prompt(
    prompt: prompt_type,
) -> ChatPromptTemplate:
    if isinstance(prompt, ChatPromptTemplate):
        return prompt

    if not isinstance(prompt, list):
        prompt = [prompt]

    messages = []

    for msg in prompt:
        if isinstance(msg, str):
            messages.append(("system", msg))
        elif isinstance(msg, dict):
            if "content" not in msg:
                raise ValueError(
                    "`content` is required to build a prompt from a dictionary."
                )
            messages.append((msg.get("role", "system"), msg["content"]))
        elif isinstance(msg, Prompt):
            messages.append((msg.role, msg.construct))
        else:
            raise ValueError(
                f"Type {type(msg)} is not supported to build a prompt"
            )

    return ChatPromptTemplate.from_messages(messages=messages)

