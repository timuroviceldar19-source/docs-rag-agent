import uuid

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from docs_rag_agent.api import dependencies
from docs_rag_agent.api.main import app
from docs_rag_agent.embeddings import FastEmbedLocalEmbedder
from docs_rag_agent.llm.base import LLMResponse, Message
from docs_rag_agent.retrieve import VectorStore

# --- Fake LLM client ---

class FakeAgentLLMClient:
    def __init__(self) -> None:
        self._call_count = 0

    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self._call_count += 1
        if self._call_count == 1:
            return LLMResponse(
                content=(
                    '{"thought": "I need to search the docs.", '
                    '"action": "search_docs", "action_input": '
                    '{"query": "FastAPI basics", "top_k": 2}}'
                ),
                model="fake-agent-model",
                input_tokens=50,
                output_tokens=30,
            )
        return LLMResponse(
            content=(
                '{"thought": "I found enough information.", '
                '"final_answer": "FastAPI is a modern web framework '
                'for building APIs with Python."}'
            ),
            model="fake-agent-model",
            input_tokens=80,
            output_tokens=20,
        )


# --- Fixtures ---

@pytest.fixture(scope="module")
def embedder() -> FastEmbedLocalEmbedder:
    return FastEmbedLocalEmbedder()


@pytest.fixture(scope="module")
def populated_store(embedder: FastEmbedLocalEmbedder) -> VectorStore:
    client = QdrantClient(":memory:")
    store = VectorStore(client=client, collection="test-agent", embedder=embedder)
    store.ensure_collection()
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
def agent_client(
    populated_store: VectorStore,
    embedder: FastEmbedLocalEmbedder,
) -> TestClient:
    dependencies.get_vector_store.cache_clear()
    dependencies.get_llm_client.cache_clear()
    dependencies.get_embedder.cache_clear()

    import docs_rag_agent.api.main as main_module
    main_module.get_vector_store = lambda: populated_store  # type: ignore[assignment]
    main_module.get_llm_client = lambda: FakeAgentLLMClient()  # type: ignore[assignment]

    return TestClient(app)


# --- Tests ---

def test_healthz_still_passes(agent_client: TestClient) -> None:
    assert agent_client.get("/healthz").status_code == 200


def test_agent_returns_200(agent_client: TestClient) -> None:
    response = agent_client.post("/agent", json={"question": "What is FastAPI?"})
    assert response.status_code == 200


def test_agent_response_shape(agent_client: TestClient) -> None:
    data = agent_client.post("/agent", json={"question": "What is FastAPI?"}).json()
    for key in ("answer", "steps", "chunks", "model", "input_tokens", "output_tokens"):
        assert key in data


def test_agent_answer_is_non_empty_string(agent_client: TestClient) -> None:
    data = agent_client.post("/agent", json={"question": "What is FastAPI?"}).json()
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


def test_agent_steps_is_non_empty_list(agent_client: TestClient) -> None:
    data = agent_client.post("/agent", json={"question": "FastAPI install"}).json()
    assert isinstance(data["steps"], list)
    assert len(data["steps"]) > 0


def test_agent_chunks_is_list(agent_client: TestClient) -> None:
    data = agent_client.post("/agent", json={"question": "FastAPI install"}).json()
    assert isinstance(data["chunks"], list)


def test_agent_question_too_short_returns_422(agent_client: TestClient) -> None:
    response = agent_client.post("/agent", json={"question": "Hi"})
    assert response.status_code == 422


def test_agent_max_iterations_too_large_returns_422(agent_client: TestClient) -> None:
    response = agent_client.post(
        "/agent", json={"question": "What is FastAPI?", "max_iterations": 99}
    )
    assert response.status_code == 422
