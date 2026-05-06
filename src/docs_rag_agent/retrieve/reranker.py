from __future__ import annotations

from typing import Protocol

from fastembed.rerank.cross_encoder import TextCrossEncoder

from docs_rag_agent.retrieve.store import SearchResult


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]: ...


class CrossEncoderReranker:
    """Two-stage retrieval reranker using a cross-encoder model.

    Pairs (query, candidate.text) are scored jointly — much higher quality
    than the bi-encoder used for first-stage retrieval, but quadratic in
    candidate count, so call only on a small over-fetched shortlist.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model = TextCrossEncoder(model_name=model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not candidates:
            return []
        scores = list(self._model.rerank(query, [c.text for c in candidates]))
        scored = [
            SearchResult(
                id=c.id,
                text=c.text,
                score=float(s),
                metadata=c.metadata,
            )
            for c, s in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
