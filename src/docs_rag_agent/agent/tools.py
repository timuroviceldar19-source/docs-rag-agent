from __future__ import annotations

from docs_rag_agent.retrieve import Reranker, SearchResult, VectorStore

TOOL_SCHEMA = """Tool: search_docs
Description: Search FastAPI documentation chunks for relevant information.
Parameters:
  query (str): The search query string.
  top_k (int, default=3): Number of results to return. Clamped to 1-10.
"""


def execute_search(
    store: VectorStore,
    query: str,
    top_k: int = 3,
    reranker: Reranker | None = None,
    fetch_k: int = 20,
) -> list[SearchResult]:
    k = min(max(top_k, 1), 10)
    if reranker is None:
        return store.search(query, top_k=k)
    candidates = store.search(query, top_k=max(fetch_k, k))
    return reranker.rerank(query, candidates, top_k=k)
