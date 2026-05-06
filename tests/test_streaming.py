"""Tests for the SSE streaming layer."""
from __future__ import annotations

import json
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

import docs_rag_agent.api.main as main_module
from docs_rag_agent.api.main import app
from docs_rag_agent.api.sse import sse_event
from docs_rag_agent.embeddings import FastEmbedLocalEmbedder
from docs_rag_agent.llm.base import (
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMStreamChunk,
    Message,
)
from docs_rag_agent.retrieve import Document, VectorStore

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class StreamingFakeLLM:
    """LLM that yields a fixed token sequence and a final usage chunk.

    Also supports the non-streaming path so it can stand in for /query and
    /agent's blocking endpoints inside the same test session.
    """

    def __init__(self, tokens: list[str] | None = None) -> None:
        self.tokens = tokens if tokens is not None else ["Hello", " ", "world", "!"]

    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        return LLMResponse(
            content="".join(self.tokens),
            model="fake-model",
            input_tokens=10,
            output_tokens=len(self.tokens),
        )

    def generate_stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[LLMStreamChunk]:
        for tok in self.tokens:
            yield LLMStreamChunk(text=tok)
        yield LLMStreamChunk(
            is_final=True,
            model="fake-model",
            input_tokens=10,
            output_tokens=len(self.tokens),
        )


class RateLimitedStreamingLLM:
    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        raise LLMRateLimitError("quota exceeded", retry_after=12.0)

    def generate_stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[LLMStreamChunk]:
        raise LLMRateLimitError("quota exceeded", retry_after=12.0)
        yield  # pragma: no cover  (unreachable, makes this a generator)


class BrokenStreamingLLM:
    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        raise LLMError("upstream broke")

    def generate_stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[LLMStreamChunk]:
        raise LLMError("upstream broke")
        yield  # pragma: no cover


class JSONLoopFakeLLM:
    """Fake LLM that drives the agent loop: first turn returns a search call,
    second turn returns a final_answer."""

    def __init__(self) -> None:
        self._turn = 0

    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self._turn += 1
        if self._turn == 1:
            content = json.dumps(
                {
                    "thought": "let me search",
                    "action": "search_docs",
                    "action_input": {"query": "fastapi", "top_k": 3},
                }
            )
        else:
            content = json.dumps(
                {"thought": "I have enough", "final_answer": "FastAPI is a framework."}
            )
        return LLMResponse(
            content=content,
            model="fake-model",
            input_tokens=5,
            output_tokens=20,
        )

    def generate_stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[LLMStreamChunk]:
        # Not used by /agent/stream — the agent uses .generate per turn.
        raise NotImplementedError
        yield  # pragma: no cover


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder() -> FastEmbedLocalEmbedder:
    return FastEmbedLocalEmbedder()


@pytest.fixture(scope="module")
def populated_store(embedder: FastEmbedLocalEmbedder) -> VectorStore:
    client = QdrantClient(":memory:")
    store = VectorStore(client=client, collection="test_stream", embedder=embedder)
    store.ensure_collection()
    store.upsert(
        [
            Document(
                id=str(uuid.uuid4()),
                text="FastAPI is a modern, fast web framework for building APIs.",
                metadata={"source": "index.md", "heading": "Introduction"},
            ),
            Document(
                id=str(uuid.uuid4()),
                text="Install FastAPI via pip: pip install fastapi.",
                metadata={"source": "tutorial/index.md", "heading": "Installation"},
            ),
        ]
    )
    return store


def _patched_client(
    populated_store: VectorStore,
    llm_factory: object,
    rerank: bool = False,
) -> TestClient:
    main_module.get_vector_store = lambda: populated_store  # type: ignore[assignment]
    main_module.get_llm_client = llm_factory  # type: ignore[assignment]
    main_module.get_reranker = lambda: None  # type: ignore[assignment]
    return TestClient(app, raise_server_exceptions=False, headers={"X-API-Key": "dev-key"})


def _parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    """Parse an SSE response body into a list of (event, data-dict) tuples."""
    events: list[tuple[str, dict[str, object]]] = []
    event_name = "message"
    data_lines: list[str] = []
    for line in body.split("\n"):
        line = line.rstrip("\r")
        if line == "":
            if data_lines:
                events.append((event_name, json.loads("\n".join(data_lines))))
            event_name = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    return events


# ---------------------------------------------------------------------------
# sse_event helper
# ---------------------------------------------------------------------------


def test_sse_event_format() -> None:
    out = sse_event("token", {"text": "hi"})
    assert out == 'event: token\ndata: {"text": "hi"}\n\n'


def test_sse_event_unicode_not_escaped() -> None:
    out = sse_event("x", {"v": "привет"})
    assert "привет" in out


# ---------------------------------------------------------------------------
# /query/stream
# ---------------------------------------------------------------------------


def test_query_stream_emits_chunks_token_end(populated_store: VectorStore) -> None:
    c = _patched_client(populated_store, lambda: StreamingFakeLLM(["Foo", " ", "bar"]))
    response = c.post(
        "/query/stream",
        json={"question": "What is FastAPI?", "top_k": 2},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    names = [e[0] for e in events]
    # Exactly one chunks frame, then >=1 token frames, then exactly one end.
    assert names[0] == "chunks"
    assert names[-1] == "end"
    assert names.count("chunks") == 1
    assert names.count("end") == 1
    assert names.count("token") >= 1

    # Reassembled text matches the fake's tokens.
    tokens = [e[1]["text"] for e in events if e[0] == "token"]
    assert "".join(tokens) == "Foo bar"  # type: ignore[arg-type]

    end_event = next(e[1] for e in events if e[0] == "end")
    assert end_event["model"] == "fake-model"
    assert end_event["input_tokens"] == 10
    assert end_event["output_tokens"] == 3


def test_query_stream_chunks_payload_has_sources(populated_store: VectorStore) -> None:
    c = _patched_client(populated_store, lambda: StreamingFakeLLM())
    response = c.post(
        "/query/stream",
        json={"question": "What is FastAPI?", "top_k": 2},
    )
    events = _parse_sse(response.text)
    chunks_event = next(e[1] for e in events if e[0] == "chunks")
    chunks = chunks_event["chunks"]
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for ch in chunks:  # type: ignore[union-attr]
        assert {"text", "source", "heading", "score"} <= set(ch.keys())


def test_query_stream_rate_limit_emits_error_event(populated_store: VectorStore) -> None:
    c = _patched_client(populated_store, lambda: RateLimitedStreamingLLM())
    response = c.post(
        "/query/stream",
        json={"question": "What is FastAPI?", "top_k": 2},
    )
    # Status is 200 because errors are surfaced as SSE frames, not HTTP errors.
    assert response.status_code == 200
    events = _parse_sse(response.text)
    error_events = [e for e in events if e[0] == "error"]
    assert len(error_events) == 1
    assert "rate limit" in str(error_events[0][1]["detail"]).lower()
    # No 'end' frame on a rate-limited stream.
    assert not any(e[0] == "end" for e in events)


def test_query_stream_llm_error_emits_error_event(populated_store: VectorStore) -> None:
    c = _patched_client(populated_store, lambda: BrokenStreamingLLM())
    response = c.post(
        "/query/stream",
        json={"question": "What is FastAPI?", "top_k": 2},
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    error_events = [e for e in events if e[0] == "error"]
    assert len(error_events) == 1
    assert error_events[0][1]["detail"] == "Upstream LLM request failed"


def test_query_stream_validates_payload(populated_store: VectorStore) -> None:
    c = _patched_client(populated_store, lambda: StreamingFakeLLM())
    response = c.post("/query/stream", json={"question": "hi"})  # too short
    assert response.status_code == 422


def test_query_stream_requires_api_key(populated_store: VectorStore) -> None:
    main_module.get_vector_store = lambda: populated_store  # type: ignore[assignment]
    main_module.get_llm_client = lambda: StreamingFakeLLM()  # type: ignore[assignment]
    main_module.get_reranker = lambda: None  # type: ignore[assignment]
    c = TestClient(app, raise_server_exceptions=False)  # no header
    response = c.post(
        "/query/stream",
        json={"question": "What is FastAPI?", "top_k": 2},
    )
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# /agent/stream
# ---------------------------------------------------------------------------


def test_agent_stream_emits_steps_then_final(populated_store: VectorStore) -> None:
    c = _patched_client(populated_store, lambda: JSONLoopFakeLLM())
    response = c.post(
        "/agent/stream",
        json={"question": "What is FastAPI?", "max_iterations": 3},
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    names = [e[0] for e in events]
    # At least 2 steps (search + final_answer) followed by exactly one final.
    assert names.count("final") == 1
    assert names[-1] == "final"
    assert names.count("step") >= 2

    final = next(e[1] for e in events if e[0] == "final")
    assert final["answer"] == "FastAPI is a framework."
    assert isinstance(final["chunks"], list)
    assert final["model"] == "fake-model"
