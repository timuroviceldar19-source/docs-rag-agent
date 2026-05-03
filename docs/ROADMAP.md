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

Status: Week 1 — Foundation in progress
