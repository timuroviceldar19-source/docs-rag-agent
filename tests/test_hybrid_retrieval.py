"""Tests for hybrid (dense + BM25 + RRF) retrieval in VectorStore."""
from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

from docs_rag_agent.embeddings import (
    FastEmbedLocalEmbedder,
    FastEmbedSparseEmbedder,
)
from docs_rag_agent.retrieve import Document, VectorStore
from docs_rag_agent.retrieve.store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME


@pytest.fixture(scope="module")
def dense_embedder() -> FastEmbedLocalEmbedder:
    return FastEmbedLocalEmbedder()


@pytest.fixture(scope="module")
def sparse_embedder() -> FastEmbedSparseEmbedder:
    return FastEmbedSparseEmbedder()


@pytest.fixture
def hybrid_store(
    dense_embedder: FastEmbedLocalEmbedder,
    sparse_embedder: FastEmbedSparseEmbedder,
) -> VectorStore:
    client = QdrantClient(":memory:")
    store = VectorStore(
        client=client,
        collection="test_hybrid",
        embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        hybrid_fetch_k=10,
    )
    store.ensure_collection()
    store.upsert(
        [
            Document(
                id=str(uuid.uuid4()),
                text="FastAPI declares path parameters with type annotations in route functions.",
                metadata={"source": "tutorial/path-params.md", "heading": "Path"},
            ),
            Document(
                id=str(uuid.uuid4()),
                text="Pydantic models define request bodies for FastAPI endpoints.",
                metadata={"source": "tutorial/body.md", "heading": "Body"},
            ),
            Document(
                id=str(uuid.uuid4()),
                text="Query parameters in FastAPI are declared as default function arguments.",
                metadata={"source": "tutorial/query-params.md", "heading": "Query"},
            ),
        ]
    )
    return store


def test_is_hybrid_flag(hybrid_store: VectorStore) -> None:
    assert hybrid_store.is_hybrid is True


def test_dense_only_store_reports_not_hybrid(
    dense_embedder: FastEmbedLocalEmbedder,
) -> None:
    client = QdrantClient(":memory:")
    store = VectorStore(client=client, collection="test_dense", embedder=dense_embedder)
    assert store.is_hybrid is False


def test_hybrid_collection_has_dense_and_sparse_configs(hybrid_store: VectorStore) -> None:
    info = hybrid_store._client.get_collection("test_hybrid")
    vectors = info.config.params.vectors
    # Named-vector dict shape — assert dense leg exists.
    assert isinstance(vectors, dict)
    assert DENSE_VECTOR_NAME in vectors
    sparse = info.config.params.sparse_vectors
    assert sparse is not None and SPARSE_VECTOR_NAME in sparse


def test_hybrid_search_returns_results(hybrid_store: VectorStore) -> None:
    results = hybrid_store.search("path parameters in FastAPI", top_k=3)
    assert len(results) > 0
    assert all(isinstance(r.text, str) and r.text for r in results)


def test_hybrid_search_surfaces_keyword_match(hybrid_store: VectorStore) -> None:
    """A unique keyword in the doc should bring its source to the top."""
    results = hybrid_store.search("Pydantic models request bodies", top_k=3)
    sources = [r.metadata.get("source", "") for r in results]
    assert any("body.md" in s for s in sources), sources


def test_dense_only_search_still_works(
    dense_embedder: FastEmbedLocalEmbedder,
) -> None:
    """Backward compat: VectorStore without sparse embedder uses unnamed dense layout."""
    client = QdrantClient(":memory:")
    store = VectorStore(client=client, collection="test_dense_compat", embedder=dense_embedder)
    store.ensure_collection()
    store.upsert(
        [
            Document(
                id=str(uuid.uuid4()),
                text="FastAPI is a fast Python web framework.",
                metadata={"source": "index.md", "heading": "Intro"},
            ),
        ]
    )
    results = store.search("python web framework", top_k=1)
    assert len(results) == 1
    assert "FastAPI" in results[0].text
