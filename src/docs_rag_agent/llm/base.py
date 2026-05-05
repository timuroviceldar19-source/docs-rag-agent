from __future__ import annotations

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


@runtime_checkable
class LLMClient(Protocol):
    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


class LLMError(Exception):
    """Base exception for LLM provider failures, normalized across providers."""


class LLMRateLimitError(LLMError):
    """Raised when the provider returns a rate-limit / quota / 429 error."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
