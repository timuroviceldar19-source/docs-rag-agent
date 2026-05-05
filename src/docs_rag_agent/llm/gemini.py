import google.genai as genai  # type: ignore[import-untyped]
import google.genai.errors as genai_errors  # type: ignore[import-untyped]
import google.genai.types as genai_types  # type: ignore[import-untyped]

from docs_rag_agent.llm.base import LLMError, LLMRateLimitError, LLMResponse, Message


class GeminiClient:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = genai.Client(api_key=api_key)
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

        system_instruction = "\n\n".join(system_messages) if system_messages else None

        contents = [
            {"role": "model" if m.role == "assistant" else m.role, "parts": [{"text": m.content}]}
            for m in other_messages
        ]

        config = genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction=system_instruction,
        )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except genai_errors.APIError as e:
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            if code == 429:
                raise LLMRateLimitError(str(e)) from e
            raise LLMError(f"Gemini API error: {e}") from e

        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage and usage.prompt_token_count else 0
        output_tokens = (
            usage.candidates_token_count if usage and usage.candidates_token_count else 0
        )

        return LLMResponse(
            content=response.text or "",
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
