import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from docs_rag_agent.api import dependencies
from docs_rag_agent.api.main import app
from docs_rag_agent.embeddings import FastEmbedLocalEmbedder
from docs_rag_agent.llm.base import LLMResponse, Message
from docs_rag_agent.retrieve import VectorStore

# --- Fake LLM client ---

class FakeLLMClient:
    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        return LLMResponse(
            content="This is a test answer.",
            model="fake-model",
            input_tokens=10,
            output_tokens=20,
        )


# --- Fixtures ---

@pytest.fixture(scope="module")
def embedder() -> FastEmbedLocalEmbedder:
    return FastEmbedLocalEmbedder()


@pytest.fixture(scope="module")
def populated_store(embedder: FastEmbedLocalEmbedder) -> VectorStore:
    client = QdrantClient(":memory:")
    store = VectorStore(client=client, collection="test", embedder=embedder)
    store.ensure_collection()
    import uuid

    from docs_rag_agent.retrieve import Document
    store.upsert([
        Document(
            id=str(uuid.uuid4()),
            text="FastAPI is a modern, fast web framework for building APIs with Python.",
            metadata={"source": "index.md", "heading": "Introduction"},
        ),
        Document(
            id=str(uuid.uuid4()),
            text="You can install FastAPI using pip: pip install fastapi",
            metadata={"source": "tutorial/index.md", "heading": "Installation"},
        ),
    ])
    return store


@pytest.fixture(scope="module")
def client(
    populated_store: VectorStore,
    embedder: FastEmbedLocalEmbedder,
) -> TestClient:
    # Override the lru_cache singletons before creating the TestClient
    dependencies.get_vector_store.cache_clear()
    dependencies.get_llm_client.cache_clear()
    dependencies.get_embedder.cache_clear()

    # Monkey-patch the cached functions to return our fakes
    dependencies.get_vector_store.__wrapped__ = lambda: populated_store  # type: ignore[attr-defined]
    dependencies.get_llm_client.__wrapped__ = lambda: FakeLLMClient()  # type: ignore[attr-defined]

    # A cleaner approach: re-register via module attribute replacement
    import docs_rag_agent.api.main as main_module
    main_module.get_vector_store = lambda: populated_store  # type: ignore[assignment]
    main_module.get_llm_client = lambda: FakeLLMClient()  # type: ignore[assignment]

    return TestClient(app)


# --- Tests ---

def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_200(client: TestClient) -> None:
    response = client.post("/query", json={"question": "How to install FastAPI?"})
    assert response.status_code == 200


def test_query_response_shape(client: TestClient) -> None:
    response = client.post("/query", json={"question": "What is FastAPI?"})
    data = response.json()
    assert "answer" in data
    assert "chunks" in data
    assert "model" in data
    assert "input_tokens" in data
    assert "output_tokens" in data


def test_query_answer_is_string(client: TestClient) -> None:
    response = client.post("/query", json={"question": "How to install FastAPI?"})
    assert isinstance(response.json()["answer"], str)
    assert len(response.json()["answer"]) > 0


def test_query_chunks_are_list(client: TestClient) -> None:
    response = client.post("/query", json={"question": "FastAPI installation"})
    chunks = response.json()["chunks"]
    assert isinstance(chunks, list)


def test_query_too_short_returns_422(client: TestClient) -> None:
    response = client.post("/query", json={"question": "Hi"})
    assert response.status_code == 422


def test_query_top_k_too_large_returns_422(client: TestClient) -> None:
    response = client.post("/query", json={"question": "What is FastAPI?", "top_k": 99})
    assert response.status_code == 422
