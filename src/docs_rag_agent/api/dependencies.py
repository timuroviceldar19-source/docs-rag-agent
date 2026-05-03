from __future__ import annotations

import functools

from qdrant_client import QdrantClient

from docs_rag_agent.config import Settings
from docs_rag_agent.embeddings import FastEmbedLocalEmbedder
from docs_rag_agent.llm import LLMClient, build_llm_client
from docs_rag_agent.retrieve import VectorStore


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@functools.lru_cache(maxsize=1)
def get_embedder() -> FastEmbedLocalEmbedder:
    settings = get_settings()
    return FastEmbedLocalEmbedder(settings.embedding_model)


@functools.lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url)


@functools.lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    client = get_qdrant_client()
    embedder = get_embedder()
    settings = get_settings()
    return VectorStore(
        client=client,
        collection=settings.qdrant_collection,
        embedder=embedder,
    )


@functools.lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    settings = get_settings()
    return build_llm_client(settings)
