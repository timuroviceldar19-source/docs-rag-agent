import json
from pathlib import Path

import pytest

from docs_rag_agent.eval import (
    compute_faithfulness,
    compute_hit,
    compute_reciprocal_rank,
)
from docs_rag_agent.llm.base import LLMResponse, Message
from docs_rag_agent.retrieve import SearchResult


def _make_result(source: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        id="x", text="text", score=score, metadata={"source": source, "heading": ""}
    )


# --- compute_hit ---

def test_compute_hit_found() -> None:
    results = [_make_result("docs/tutorial/path-params.md")]
    assert compute_hit(results, "path-params") is True


def test_compute_hit_not_found() -> None:
    results = [_make_result("docs/tutorial/other.md")]
    assert compute_hit(results, "path-params") is False


# --- compute_reciprocal_rank ---

def test_rr_first_position() -> None:
    results = [_make_result("path-params.md"), _make_result("other.md", 0.8)]
    assert compute_reciprocal_rank(results, "path-params") == pytest.approx(1.0)


def test_rr_second_position() -> None:
    results = [_make_result("other.md", 0.9), _make_result("path-params.md", 0.8)]
    assert compute_reciprocal_rank(results, "path-params") == pytest.approx(0.5)


# --- compute_faithfulness ---

def test_compute_faithfulness_with_fake_llm() -> None:
    class FakeJudgeLLM:
        def generate(
            self,
            messages: list[Message],
            *,
            max_tokens: int = 100,
            temperature: float = 0.0,
        ) -> LLMResponse:
            return LLMResponse(
                content='{"score": 0.85, "reasoning": "Mostly grounded."}',
                model="fake",
                input_tokens=5,
                output_tokens=5,
            )

    score = compute_faithfulness("Q?", "some context", "some answer", FakeJudgeLLM())
    assert score == pytest.approx(0.85)


# --- eval_dataset.json ---

def test_eval_dataset_is_valid() -> None:
    data = json.loads(Path("data/eval_dataset.json").read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 10
    for item in data:
        assert {"id", "question", "expected_source_contains", "reference_answer"} <= item.keys()
        assert isinstance(item["question"], str) and len(item["question"]) > 10
