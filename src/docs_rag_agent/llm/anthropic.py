from typing import cast

import anthropic
from anthropic.types import TextBlock

from docs_rag_agent.llm.base import LLMError, LLMRateLimitError, LLMResponse, Message


class AnthropicClient:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        system_messages = [m.content for m in messages if m.role == "system"]
        other_messages = [m for m in messages if m.role != "system"]

        system_prompt = "\n\n".join(system_messages) if system_messages else None

        api_messages = [{"role": m.role, "content": m.content} for m in other_messages]

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,  # type: ignore[arg-type]
                messages=api_messages,  # type: ignore[arg-type]
            )
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API error: {e}") from e

        content_block = cast(TextBlock, response.content[0])
        return LLMResponse(
            content=content_block.text,
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
