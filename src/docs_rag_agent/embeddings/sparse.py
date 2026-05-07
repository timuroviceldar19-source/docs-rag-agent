"""Sparse text embedder for hybrid retrieval (BM25)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from fastembed import SparseTextEmbedding


@dataclass
class SparseVector:
    """A sparse vector: parallel arrays of token-id indices and weights."""

    indices: list[int]
    values: list[float]


@runtime_checkable
class SparseEmbedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[SparseVector]: ...


class FastEmbedSparseEmbedder:
    """BM25 token-frequency vectorizer via fastembed.

    Pairs naturally with ``FastEmbedLocalEmbedder`` (dense) for two-stage
    hybrid retrieval. The model produces document-side BM25 weights; at
    search time Qdrant applies IDF when the sparse vector is configured
    with ``Modifier.IDF``.
    """

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self._model = SparseTextEmbedding(model_name=model_name)
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(
                indices=[int(i) for i in e.indices],
                values=[float(v) for v in e.values],
            )
            for e in self._model.embed(texts)
        ]
