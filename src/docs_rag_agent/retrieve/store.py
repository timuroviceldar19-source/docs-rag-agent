from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from docs_rag_agent.embeddings.base import Embedder
from docs_rag_agent.embeddings.sparse import SparseEmbedder

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"


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
    """Qdrant-backed vector store.

    When ``sparse_embedder`` is None: classic single-vector dense layout
    (unnamed cosine vector). When ``sparse_embedder`` is provided: hybrid
    layout with named ``dense`` (cosine) + sparse ``bm25`` (IDF modifier),
    and ``search`` runs a server-side RRF fusion over both.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embedder: Embedder,
        sparse_embedder: SparseEmbedder | None = None,
        hybrid_fetch_k: int = 50,
    ) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._sparse_embedder = sparse_embedder
        self._hybrid_fetch_k = hybrid_fetch_k

    @property
    def is_hybrid(self) -> bool:
        return self._sparse_embedder is not None

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist.

        Layout depends on whether a sparse embedder is configured. If the
        collection already exists we leave it alone — switching layouts
        requires dropping and re-creating (with a fresh ingest).
        """
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            return

        if self._sparse_embedder is None:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(
                    size=self._embedder.vector_size,
                    distance=qm.Distance.COSINE,
                ),
            )
        else:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    DENSE_VECTOR_NAME: qm.VectorParams(
                        size=self._embedder.vector_size,
                        distance=qm.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: qm.SparseVectorParams(
                        modifier=qm.Modifier.IDF,
                    ),
                },
            )

    def upsert(self, documents: list[Document], batch_size: int = 250) -> None:
        """Embed and upsert documents in batches. Assigns a UUID id if doc.id is empty."""
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc.text for doc in batch]
            dense_vectors = self._embedder.embed(texts)
            sparse_vectors = (
                self._sparse_embedder.embed(texts)
                if self._sparse_embedder is not None
                else None
            )

            points: list[qm.PointStruct] = []
            for idx, doc in enumerate(batch):
                point_id = doc.id if doc.id else str(uuid.uuid4())
                payload = {"text": doc.text, **doc.metadata}
                if sparse_vectors is None:
                    points.append(
                        qm.PointStruct(
                            id=point_id, vector=dense_vectors[idx], payload=payload
                        )
                    )
                else:
                    sv = sparse_vectors[idx]
                    points.append(
                        qm.PointStruct(
                            id=point_id,
                            vector={
                                DENSE_VECTOR_NAME: dense_vectors[idx],
                                SPARSE_VECTOR_NAME: qm.SparseVector(
                                    indices=sv.indices,
                                    values=sv.values,
                                ),
                            },
                            payload=payload,
                        )
                    )
            self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return top-k results.

        Hybrid mode (sparse embedder configured): server-side RRF fusion of
        dense + BM25 candidates. Dense-only mode: plain cosine.
        """
        if self._sparse_embedder is None:
            query_vector = self._embedder.embed([query])[0]
            response = self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                limit=top_k,
            )
        else:
            dense_vec = self._embedder.embed([query])[0]
            sparse_vec = self._sparse_embedder.embed([query])[0]
            response = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    qm.Prefetch(
                        query=dense_vec,
                        using=DENSE_VECTOR_NAME,
                        limit=self._hybrid_fetch_k,
                    ),
                    qm.Prefetch(
                        query=qm.SparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values,
                        ),
                        using=SPARSE_VECTOR_NAME,
                        limit=self._hybrid_fetch_k,
                    ),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
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
