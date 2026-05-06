from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass
class Message:
    role: Literal["user", "assistant", "system"]
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass
class LLMStreamChunk:
    """One chunk of a streamed LLM response.

    A stream is a sequence of token chunks (`text != ""`, `is_final=False`)
    followed by exactly one terminal chunk (`is_final=True`, carrying the
    final usage and model name; `text` is empty).
    """

    text: str = ""
    is_final: bool = False
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@runtime_checkable
class LLMClient(Protocol):
    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse: ...

    def generate_stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[LLMStreamChunk]: ...


class LLMError(Exception):
    """Base exception for LLM provider failures, normalized across providers."""


class LLMRateLimitError(LLMError):
    """Raised when the provider returns a rate-limit / quota / 429 error."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
