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

- **Agent drops retrieval scores.** In `/agent` responses, all returned chunks have `score=0.000`, while `/query` returns proper scores (0.8+). Likely cause: in `agent/loop.py` or `agent/tools.py`, `SearchResult` objects are reconstructed without preserving the original score. Fix: pass the score through unchanged when collecting agent sources.
- **No deduplication across agent iterations.** A 5-iteration ReAct run with `top_k=3` returns 15 chunks, but observed runs all 15 from the same single source file. The agent re-searches with similar queries and accumulates duplicates. Fix: dedupe by `(source, heading, text-prefix)` key before exposing in `result.sources`, and ideally feed the agent a "you've already seen X" signal.
- **Agent wastes iterations on dead-end queries.** Vague questions (e.g. `"FastAPI documentation"`) cause 5 fruitless `search_docs` calls and ~4k tokens burned, ending in `"Could not find a definitive answer within the iteration limit."` Fix in agent system prompt: "If two consecutive searches return the same sources, either reformulate the query substantially or give up early with a partial answer."
- **Backend conflates third-party rate limits with internal errors.** `429 RESOURCE_EXHAUSTED` from Gemini propagates to the client as `500 Internal Server Error`. Fix: catch `google.genai.errors.ClientError` in `llm/gemini.py` (and equivalents in other providers), re-raise as `HTTPException(503, retry_after=...)` in the API layer.

Status: Weeks 1-3 complete (foundation, LLM/RAG, ingest, FastAPI, ReAct agent, evals, tracing, Docker, CI, public repo, Streamlit UI).

