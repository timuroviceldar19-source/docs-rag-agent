# Roadmap

## Week 1: Working RAG pipeline
**Goal:** Build the core retrieval and generation loop.
- **Deliverables:**
    - Foundation Scaffold (Project skeleton, CI, docs)
    - Configuration layer + LLM provider abstraction
    - Local embeddings with `fastembed` + Qdrant integration
    - Ingest CLI for FastAPI documentation
    - FastAPI endpoint `/query` for basic RAG
- **Done when:**
    - `curl /query` returns an LLM-generated answer with at least one citation chunk from the FastAPI documentation.

## Week 2: Production polish
**Goal:** Improve robustness and observability.
- **Deliverables:**
    - ReAct-style agent loop with web-search fallback tool
    - Structured outputs for consistent API responses
    - Evaluations (RAGAS or custom golden set)
    - Tracing integration with Langfuse
    - Dockerfile for the application and updated docker-compose
- **Done when:**
    - The agent can fallback to web search when documentation is insufficient.
    - Evaluation metrics show a clear improvement over the baseline RAG.

## Week 3: Visible product
**Goal:** Create a user-facing interface and demo.
- **Deliverables:**
    - Minimal Streamlit or Next.js frontend
    - Deployment to Fly.io or Render with a public URL
    - Animated demo GIF for the README
    - Blog post draft (Habr/Medium) explaining the architecture
- **Done when:**
    - A public URL is accessible and functional.
    - The demo GIF accurately reflects the user experience.

## Week 4: Polish + side project
**Goal:** Finalize the portfolio piece and extend impact.
- **Deliverables:**
    - GitHub profile README updated with the project
    - Second smaller repository (MCP server or specialized eval CLI)
    - LinkedIn profile updated with project achievements
    - Job applications sent out
- **Done when:**
    - The project is showcased prominently on GitHub and LinkedIn.
    - All week 4 deliverables are completed.

## Known issues (technical debt, surfaced during E2E UI testing on 2026-05-04)

- **No deduplication across agent iterations.** A 5-iteration ReAct run with `top_k=3` accumulates up to 15 chunks in `result.sources`, often dominated by a single source file when the agent re-issues similar queries. Fix: dedupe by `(source, heading, text-prefix)` in `agent/loop.py` before returning, and ideally feed the agent a "you've already seen X" signal in observations.
- **Agent wastes iterations on dead-end queries.** Vague questions (e.g. `"FastAPI documentation"`) cause 5 fruitless `search_docs` calls and ~4k tokens burned, ending in `"Could not find a definitive answer within the iteration limit."` Fix in agent system prompt: "If two consecutive searches return the same sources, either reformulate the query substantially or give up early with a partial answer."
- ~~**Backend conflates third-party rate limits with internal errors.**~~ **Resolved 2026-05-04.** All three LLM providers (`gemini`, `anthropic`, `openai`) now raise domain exceptions `LLMRateLimitError` / `LLMError` from `llm/base.py` instead of leaking SDK-specific errors. FastAPI exception handlers map `LLMRateLimitError → 503 (with Retry-After)` and `LLMError → 502`. Covered by 4 new tests in `tests/test_api.py`.

## Operational lessons (not bugs)

- **Qdrant volume is fragile across Docker restarts.** On 2026-05-04 a Docker Desktop restart left the persisted collection with all-zero vectors (`indexed_vectors_count=0`, `points_count=8862`, vectors in storage all zeros), causing every search to return the same chunks with `score=0.000`. Re-running `python -m docs_rag_agent.ingest.pipeline` recovered the index. Initially misdiagnosed as a code bug — the score-preservation chain in `store.search → execute_search → all_sources → CitedChunk` is in fact correct. Mitigation options for the future: (a) add a healthcheck that samples one vector and asserts it is non-zero, (b) move to Qdrant cloud or named volumes with safer persistence, (c) document the re-ingest command prominently in the README runbook.

Status: Weeks 1-3 complete (foundation, LLM/RAG, ingest, FastAPI, ReAct agent, evals, tracing, Docker, CI, public repo, Streamlit UI).

