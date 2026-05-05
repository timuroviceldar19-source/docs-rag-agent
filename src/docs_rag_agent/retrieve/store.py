from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from docs_rag_agent.embeddings.base import Embedder


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    vector: list[float] | None = None


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict[str, str]


class VectorStore:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embedder: Embedder,
    ) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._embedder.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, documents: list[Document], batch_size: int = 250) -> None:
        """Embed and upsert documents in batches. Assigns a UUID id if doc.id is empty."""
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc.text for doc in batch]
            vectors = self._embedder.embed(texts)

            points = [
                PointStruct(
                    id=doc.id if doc.id else str(uuid.uuid4()),
                    vector=vector,
                    payload={"text": doc.text, **doc.metadata},
                )
                for doc, vector in zip(batch, vectors, strict=True)
            ]
            self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Embed query and return top-k nearest documents."""
        query_vector = self._embedder.embed([query])[0]
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
        )
        return [
            SearchResult(
                id=str(hit.id),
                text=str(hit.payload.get("text", "") if hit.payload else ""),
                score=hit.score,
                metadata={k: str(v) for k, v in (hit.payload or {}).items() if k != "text"},
            )
            for hit in response.points
        ]

