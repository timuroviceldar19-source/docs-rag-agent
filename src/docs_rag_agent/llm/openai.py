from collections.abc import Iterator

import openai
from openai import OpenAI

from docs_rag_agent.llm.base import (
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMStreamChunk,
    Message,
)


class OpenAIClient:
    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=api_messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}") from e

        content = response.choices[0].message.content or ""

        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        return LLMResponse(
            content=content,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def generate_stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[LLMStreamChunk]:
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )  # type: ignore[call-overload]
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}") from e

        input_tokens = 0
        output_tokens = 0
        try:
            for chunk in stream:
                # Token deltas: chunks with at least one choice.
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    text = delta.content or ""
                    if text:
                        yield LLMStreamChunk(text=text)
                # Final usage frame: choices is empty, usage is set.
                if chunk.usage is not None:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}") from e

        yield LLMStreamChunk(
            is_final=True,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
