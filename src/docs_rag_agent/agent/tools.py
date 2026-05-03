from __future__ import annotations

from docs_rag_agent.retrieve import SearchResult, VectorStore

TOOL_SCHEMA = """Tool: search_docs
Description: Search FastAPI documentation chunks for relevant information.
Parameters:
  query (str): The search query string.
  top_k (int, default=3): Number of results to return. Clamped to 1-10.
"""


def execute_search(store: VectorStore, query: str, top_k: int = 3) -> list[SearchResult]:
    return store.search(query, top_k=min(max(top_k, 1), 10))
