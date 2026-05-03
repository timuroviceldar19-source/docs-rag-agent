from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from docs_rag_agent.llm import LLMClient, Message
from docs_rag_agent.retrieve import SearchResult


@dataclass
class EvalRecord:
    id: str
    question: str
    expected_source_contains: str
    reference_answer: str


@dataclass
class EvalResult:
    question_id: str
    question: str
    hit: bool
    reciprocal_rank: float
    faithfulness: float | None = None


def compute_hit(results: list[SearchResult], expected_contains: str) -> bool:
    """True if any result's source path contains expected_contains."""
    return any(expected_contains in r.metadata.get("source", "") for r in results)


def compute_reciprocal_rank(results: list[SearchResult], expected_contains: str) -> float:
    """1/rank of the first result whose source path contains expected_contains, or 0.0."""
    for rank, r in enumerate(results, 1):
        if expected_contains in r.metadata.get("source", ""):
            return 1.0 / rank
    return 0.0


def compute_faithfulness(
    question: str,
    context: str,
    answer: str,
    llm: LLMClient,
) -> float:
    """LLM-as-judge: score how faithfully the answer is grounded in context (0.0–1.0).

    Returns 0.0 on parse failure — never raises.
    """
    prompt = (
        "Rate the faithfulness of the answer to the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        'Respond with ONLY JSON: {"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}'
    )
    messages = [Message(role="user", content=prompt)]
    response = llm.generate(messages, max_tokens=100, temperature=0.0)
    try:
        raw = cast(dict[str, Any], json.loads(response.content))
        return max(0.0, min(1.0, float(raw.get("score", 0.0))))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return 0.0


def summarize(results: list[EvalResult]) -> dict[str, float]:
    """Aggregate hit rate, MRR, and mean faithfulness (if available)."""
    n = len(results)
    if n == 0:
        return {"hit_rate": 0.0, "mrr": 0.0, "mean_faithfulness": 0.0}
    faith_scores = [r.faithfulness for r in results if r.faithfulness is not None]
    return {
        "hit_rate": sum(1 for r in results if r.hit) / n,
        "mrr": sum(r.reciprocal_rank for r in results) / n,
        "mean_faithfulness": sum(faith_scores) / len(faith_scores) if faith_scores else 0.0,
    }
