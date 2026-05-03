import pytest

from docs_rag_agent.embeddings import Embedder, FastEmbedLocalEmbedder


@pytest.fixture(scope="module")
def embedder() -> FastEmbedLocalEmbedder:
    return FastEmbedLocalEmbedder()


def test_embedder_satisfies_protocol(embedder: FastEmbedLocalEmbedder) -> None:
    assert isinstance(embedder, Embedder)


def test_embed_returns_correct_shape(embedder: FastEmbedLocalEmbedder) -> None:
    texts = ["Hello world", "FastAPI is fast"]
    vectors = embedder.embed(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == embedder.vector_size
    assert len(vectors[1]) == embedder.vector_size


def test_embed_returns_plain_float_list(embedder: FastEmbedLocalEmbedder) -> None:
    vectors = embedder.embed(["test"])
    assert isinstance(vectors, list)
    assert isinstance(vectors[0], list)
    assert isinstance(vectors[0][0], float)


def test_vector_size_positive(embedder: FastEmbedLocalEmbedder) -> None:
    assert embedder.vector_size > 0
