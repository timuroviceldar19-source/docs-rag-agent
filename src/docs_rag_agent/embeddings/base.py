from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    vector_size: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...
