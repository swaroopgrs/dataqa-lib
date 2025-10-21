import json
from typing import Any, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field, ValidationError

from dataqa.core.llm.base_llm import (
    BaseLLM,
    LLMConfig,
    LLMError,
    LLMOutput,
)
from dataqa.core.utils.prompt_utils import messages_to_serializable


# Helper to extract JSON from a string that might have surrounding text
def extract_json_from_string(text: str) -> str:
    try:
        # Find the start of the JSON block
        json_start = text.find("{")
        if json_start == -1:
            return text  # No JSON object found

        # Find the end of the JSON block
        json_end = text.rfind("}")
        if json_end == -1:
            return text  # No JSON object found

        return text[json_start : json_end + 1]
    except Exception:
        return text  # Return original text on any error


class GeminiConfig(LLMConfig):
    api_key: str = Field(description="The Google API key for Gemini.")
    temperature: float = Field(default=0.0)


class GeminiLLM(BaseLLM):
    config_base_model = GeminiConfig
    config: GeminiConfig

    def _get_model(self, **kwargs) -> ChatGoogleGenerativeAI:
        api_key = self.config.api_key
        if not api_key or "${" in api_key:
            raise ValueError("Gemini API key not set.")

        # For structured output, we will request JSON directly
        response_format = (
            "json" if kwargs.get("with_structured_output") else "text"
        )

        return ChatGoogleGenerativeAI(
            model=self.config.model,
            google_api_key=api_key,
            temperature=self.config.temperature,
            convert_system_message_to_human=True,
            # Tell Gemini to output JSON if requested
            response_mime_type=f"application/{response_format}",
        )

    async def ainvoke(self, messages: List[Any], **kwargs) -> LLMOutput:
        serialized_messages = messages_to_serializable(messages)
        output_schema = kwargs.get("with_structured_output")

        try:
            model = self.get_model(**kwargs)
            base_messages: List[BaseMessage] = messages.to_messages()

            # 2. Rebuild the list, merging system content into the first human message.
            final_messages_for_llm = []
            system_content_parts = []
            for msg in base_messages:
                if msg.type == "system":
                    system_content_parts.append(msg.content)
                elif msg.type == "human":
                    # If there's preceding system content, prepend it.
                    if system_content_parts:
                        full_content = (
                            "\n\n".join(system_content_parts)
                            + "\n\n"
                            + msg.content
                        )
                        final_messages_for_llm.append(
                            HumanMessage(content=full_content)
                        )
                        system_content_parts = []  # Clear after use
                    else:
                        final_messages_for_llm.append(msg)
                else:  # Pass through AI messages as is
                    final_messages_for_llm.append(msg)
            # --- END CRITICAL FIX ---

            # Now, invoke the model with the correctly formatted message list
            response: AIMessage = await model.ainvoke(final_messages_for_llm)
            raw_text_output = response.content

            if output_schema:
                json_string = extract_json_from_string(raw_text_output)
                try:
                    parsed_output = output_schema.model_validate_json(
                        json_string
                    )
                    generation = parsed_output
                except (ValidationError, json.JSONDecodeError) as e:
                    generation = f"Pydantic Validation Error: {e}\n---RAW OUTPUT---\n{raw_text_output}"
            else:
                generation = raw_text_output

            return LLMOutput(prompt=serialized_messages, generation=generation)
        except Exception as e:
            import traceback

            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            error_obj = LLMError(
                error_code=500,
                error_type=type(e).__name__,
                error_message=error_msg,
            )
            return LLMOutput(prompt=serialized_messages, error=error_obj)
