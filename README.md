# docs-rag-agent

> Production-style RAG + ReAct agent over FastAPI documentation.
> Multi-provider LLM layer, local embeddings, Qdrant, and a CLI evaluation suite.

[![CI](https://github.com/timuroviceldar19-source/docs-rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/timuroviceldar19-source/docs-rag-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-57%20passed-brightgreen)

## Features

- **Multi-provider LLM** — Gemini, Anthropic, OpenAI via a shared `Protocol`; swap with one env var
- **Local embeddings** — `fastembed` (BAAI/bge-small-en-v1.5), no embedding API costs
- **Qdrant vector store** — similarity search with source + heading metadata
- **Ingest pipeline** — Markdown → chunk → embed → upsert, idempotent
- **ReAct agent** — multi-step reasoning loop with `search_docs` tool (`/agent` endpoint)
- **FastAPI layer** — `/query`, `/agent`, `/healthz` with full Pydantic v2 models
- **Retrieval evals** — hit rate, MRR, LLM-as-judge faithfulness (10-item golden dataset)
- **Optional Langfuse tracing** — zero cost when keys absent
- **Docker Compose** — `docker-compose up --build` starts Qdrant + API
- **57 tests** — pure unit tests, no external services, no API keys required

## Architecture

```mermaid
graph TD
    Client -->|HTTP POST| FastAPI

    subgraph "FastAPI (api/)"
        Q["/query"] --> RAG["RAG: retrieve + generate"]
        A["/agent"] --> React["ReAct loop"]
    end

    RAG --> VS[(Qdrant)]
    RAG --> LLM[LLMClient]
    React --> VS
    React --> LLM

    subgraph "LLM layer (llm/)"
        LLM --> Gemini
        LLM --> Anthropic
        LLM --> OpenAI
    end

    subgraph "Ingest (scripts/pipeline.py)"
        Docs["FastAPI docs (.md)"] --> Chunker --> Embedder --> VS
    end
```

## Quickstart

### Prerequisites

- Python 3.11
- Docker + Docker Compose
- A Gemini API key (or Anthropic/OpenAI — set `LLM_PROVIDER`)

### Install

```bash
git clone https://github.com/timuroviceldar19-source/docs-rag-agent.git
cd docs-rag-agent
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Set GEMINI_API_KEY (or ANTHROPIC_API_KEY / OPENAI_API_KEY)
# Optionally set LLM_PROVIDER (default: gemini)
```

### Start Qdrant

```bash
docker-compose up qdrant -d
```

### Ingest FastAPI docs

```bash
python scripts/pipeline.py --docs-dir data/fastapi-docs
```

### Run the API

```bash
uvicorn docs_rag_agent.api.main:app --reload
# → http://localhost:8000/docs
```

### Full stack with Docker

```bash
docker-compose up --build
```

## API Usage

**Query (single-step RAG):**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I declare path parameters?", "top_k": 5}'
```

**Agent (multi-step ReAct):**

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between path and query parameters?", "max_iterations": 5}'
```

**Retrieval eval:**

```bash
# Requires Qdrant running with ingested docs
python scripts/eval.py --top-k 5             # hit rate + MRR
python scripts/eval.py --top-k 5 --judge     # + LLM-as-judge faithfulness
```

## Running tests

```bash
pytest          # 57 tests, no network required
mypy src/       # strict type check, 25 source files
ruff check .    # linting
```

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.115 + Pydantic v2 |
| LLM | Gemini / Anthropic / OpenAI (swappable Protocol) |
| Embeddings | fastembed · BAAI/bge-small-en-v1.5 (local) |
| Vector DB | Qdrant |
| Config | pydantic-settings |
| CLI | Typer |
| Tests | pytest 8 · mypy --strict · ruff |
| Tracing | Langfuse (optional) |
| Container | Docker + Docker Compose |

## Project structure

```
src/docs_rag_agent/
├── api/          # FastAPI app, endpoints, dependency singletons
├── agent/        # ReAct loop and tool execution
├── llm/          # Multi-provider LLM abstraction (Protocol + 3 clients)
├── embeddings/   # fastembed local embedder
├── retrieve/     # Qdrant vector store wrapper
├── ingest/       # Markdown chunker and ingestion pipeline
├── eval.py       # Retrieval + faithfulness eval functions
├── config.py     # pydantic-settings configuration
└── tracing.py    # Optional Langfuse tracing
scripts/
├── pipeline.py   # Ingest CLI (Typer)
└── eval.py       # Eval CLI (Typer)
data/
└── eval_dataset.json  # 10 golden QA pairs
```

## License

MIT
