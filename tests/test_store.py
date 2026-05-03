import uuid

import pytest
from qdrant_client import QdrantClient

from docs_rag_agent.embeddings import FastEmbedLocalEmbedder
from docs_rag_agent.retrieve import Document, SearchResult, VectorStore


@pytest.fixture(scope="module")
def embedder() -> FastEmbedLocalEmbedder:
    return FastEmbedLocalEmbedder()


@pytest.fixture(scope="module")
def store(embedder: FastEmbedLocalEmbedder) -> VectorStore:
    client = QdrantClient(":memory:")
    vs = VectorStore(client=client, collection="test_col", embedder=embedder)
    vs.ensure_collection()
    return vs


def test_ensure_collection_is_idempotent(store: VectorStore) -> None:
    store.ensure_collection()  # second call must not raise
    store.ensure_collection()  # third call too


def test_upsert_and_search_basic(store: VectorStore) -> None:
    docs = [
        Document(id=str(uuid.uuid4()), text="FastAPI is a modern Python web framework."),
        Document(id=str(uuid.uuid4()), text="Qdrant is a vector similarity search engine."),
        Document(id=str(uuid.uuid4()), text="Python is great for machine learning."),
    ]
    store.upsert(docs)

    results = store.search("web framework", top_k=1)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert "FastAPI" in results[0].text


def test_search_returns_scores_between_0_and_1(store: VectorStore) -> None:
    results = store.search("search engine", top_k=3)
    for r in results:
        assert 0.0 <= r.score <= 1.0


def test_upsert_with_metadata(store: VectorStore) -> None:
    doc = Document(
        id=str(uuid.uuid4()),
        text="Dependency injection in FastAPI.",
        metadata={"source": "fastapi_docs", "section": "tutorial"},
    )
    store.upsert([doc])

    results = store.search("dependency injection fastapi", top_k=1)
    assert results[0].metadata.get("source") == "fastapi_docs"
    assert results[0].metadata.get("section") == "tutorial"
