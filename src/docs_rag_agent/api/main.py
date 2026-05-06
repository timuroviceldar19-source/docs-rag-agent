import logging
from collections.abc import Iterator
from typing import Any, cast

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from docs_rag_agent.agent.loop import AgentResult, AgentStep, run_agent, run_agent_iter
from docs_rag_agent.api.dependencies import (
    get_llm_client,
    get_reranker,
    get_settings,
    get_vector_store,
    verify_api_key,
)
from docs_rag_agent.api.sse import sse_event
from docs_rag_agent.llm import LLMError, LLMRateLimitError, Message
from docs_rag_agent.retrieve import SearchResult
from docs_rag_agent.tracing import trace_agent, trace_query

logger = logging.getLogger(__name__)

app = FastAPI(
    title="docs-rag-agent",
    description="RAG + agent over FastAPI documentation.",
    version="0.1.0",
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)

class CitedChunk(BaseModel):
    text: str
    source: str
    heading: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    chunks: list[CitedChunk]
    model: str
    input_tokens: int
    output_tokens: int

class AgentStepOut(BaseModel):
    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None
    final_answer: str | None = None


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    max_iterations: int = Field(default=5, ge=1, le=10)


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStepOut]
    chunks: list[CitedChunk]   # reuse existing model
    model: str
    input_tokens: int
    output_tokens: int

@app.exception_handler(LLMRateLimitError)
def _handle_rate_limit(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, LLMRateLimitError)
    headers: dict[str, str] = {}
    if exc.retry_after is not None:
        headers["Retry-After"] = str(int(exc.retry_after))
    return JSONResponse(
        status_code=503,
        content={"detail": f"LLM provider rate limit: {exc}"},
        headers=headers,
    )


@app.exception_handler(LLMError)
def _handle_llm_error(_request: Request, exc: Exception) -> JSONResponse:
    logger.error("Upstream LLM error: %s", exc)
    return JSONResponse(status_code=502, content={"detail": "Upstream LLM request failed"})


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
def query(request: QueryRequest) -> QueryResponse:
    store = get_vector_store()
    llm = get_llm_client()
    reranker = get_reranker()
    settings = get_settings()

    if reranker is None:
        results = store.search(request.question, top_k=request.top_k)
    else:
        candidates = store.search(
            request.question,
            top_k=max(settings.rerank_fetch_k, request.top_k),
        )
        results = reranker.rerank(request.question, candidates, top_k=request.top_k)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")
    
    context = "\n\n---\n\n".join(
        f"[{r.metadata.get('source', 'unknown')} / {r.metadata.get('heading', 'none')}]\n{r.text}" 
        for r in results
    )
    
    messages = [
        Message(
            role="system",
            content=(
                "You are a helpful assistant answering questions about the FastAPI framework. "
                "Answer based only on the provided context. "
                "Cite the source file when you reference information."
            ),
        ),
        Message(
            role="user",
            content=f"Context:\n{context}\n\nQuestion: {request.question}",
        ),
    ]
    
    response = llm.generate(messages, max_tokens=1024, temperature=0.0)
    
    result = QueryResponse(
        answer=response.content,
        chunks=[
            CitedChunk(
                text=r.text, 
                source=r.metadata.get("source", "unknown"), 
                heading=r.metadata.get("heading", "none"), 
                score=r.score
            ) 
            for r in results
        ],
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens
    )
    trace_query(
        question=request.question,
        answer=response.content,
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        num_chunks=len(results),
    )
    return result

@app.post("/agent", response_model=AgentResponse, dependencies=[Depends(verify_api_key)])
def agent_endpoint(request: AgentRequest) -> AgentResponse:
    store = get_vector_store()
    llm = get_llm_client()
    reranker = get_reranker()
    result = run_agent(
        question=request.question,
        store=store,
        llm=llm,
        max_iterations=request.max_iterations,
        reranker=reranker,
    )
    response_obj = AgentResponse(
        answer=result.answer,
        steps=[
            AgentStepOut(
                thought=s.thought,
                action=s.action,
                action_input=s.action_input,
                observation=s.observation,
                final_answer=s.final_answer,
            )
            for s in result.steps
        ],
        chunks=[
            CitedChunk(
                text=r.text,
                source=r.metadata.get("source", ""),
                heading=r.metadata.get("heading", ""),
                score=r.score,
            )
            for r in result.sources
        ],
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
    trace_agent(
        question=request.question,
        answer=result.answer,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        num_steps=len(result.steps),
        num_chunks=len(result.sources),
    )
    return response_obj


# ---------------------------------------------------------------------------
# Streaming endpoints
# ---------------------------------------------------------------------------


def _retrieve_for_query(question: str, top_k: int) -> list[SearchResult]:
    """Run the same retrieval+rerank pipeline used by /query."""
    store = get_vector_store()
    reranker = get_reranker()
    settings = get_settings()
    if reranker is None:
        return store.search(question, top_k=top_k)
    candidates = store.search(question, top_k=max(settings.rerank_fetch_k, top_k))
    return reranker.rerank(question, candidates, top_k=top_k)


def _chunk_payload(r: SearchResult) -> dict[str, Any]:
    return {
        "text": r.text,
        "source": r.metadata.get("source", "unknown"),
        "heading": r.metadata.get("heading", "none"),
        "score": r.score,
    }


def _step_payload(s: AgentStep) -> dict[str, Any]:
    return {
        "thought": s.thought,
        "action": s.action,
        "action_input": s.action_input,
        "observation": s.observation,
        "final_answer": s.final_answer,
    }


@app.post("/query/stream", dependencies=[Depends(verify_api_key)])
def query_stream(request: QueryRequest) -> StreamingResponse:
    """SSE stream for /query.

    Frames:
      event: chunks  data: {"chunks": [...]}                 (once, up front)
      event: token   data: {"text": "..."}                   (many)
      event: end     data: {"model", "input_tokens", "output_tokens"}
      event: error   data: {"detail": "..."}                 (on failure)
    """
    llm = get_llm_client()
    results = _retrieve_for_query(request.question, request.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    context = "\n\n---\n\n".join(
        f"[{r.metadata.get('source', 'unknown')} / {r.metadata.get('heading', 'none')}]\n{r.text}"
        for r in results
    )
    messages = [
        Message(
            role="system",
            content=(
                "You are a helpful assistant answering questions about the FastAPI framework. "
                "Answer based only on the provided context. "
                "Cite the source file when you reference information."
            ),
        ),
        Message(
            role="user",
            content=f"Context:\n{context}\n\nQuestion: {request.question}",
        ),
    ]

    def generator() -> Iterator[str]:
        yield sse_event("chunks", {"chunks": [_chunk_payload(r) for r in results]})
        accumulated: list[str] = []
        model = ""
        input_tokens = 0
        output_tokens = 0
        try:
            for chunk in llm.generate_stream(messages, max_tokens=1024, temperature=0.0):
                if chunk.is_final:
                    model = chunk.model
                    input_tokens = chunk.input_tokens
                    output_tokens = chunk.output_tokens
                    continue
                if chunk.text:
                    accumulated.append(chunk.text)
                    yield sse_event("token", {"text": chunk.text})
        except LLMRateLimitError as e:
            yield sse_event("error", {"detail": f"LLM provider rate limit: {e}"})
            return
        except LLMError as e:
            logger.error("Upstream LLM error during stream: %s", e)
            yield sse_event("error", {"detail": "Upstream LLM request failed"})
            return

        yield sse_event(
            "end",
            {"model": model, "input_tokens": input_tokens, "output_tokens": output_tokens},
        )
        trace_query(
            question=request.question,
            answer="".join(accumulated),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            num_chunks=len(results),
        )

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/agent/stream", dependencies=[Depends(verify_api_key)])
def agent_stream(request: AgentRequest) -> StreamingResponse:
    """SSE stream for /agent.

    Frames:
      event: step   data: {<AgentStep>}                            (per ReAct turn)
      event: final  data: {"answer", "chunks", "model", ...usage}  (once at end)
      event: error  data: {"detail": "..."}                        (on failure)
    """
    store = get_vector_store()
    llm = get_llm_client()
    reranker = get_reranker()

    def generator() -> Iterator[str]:
        try:
            for kind, payload in run_agent_iter(
                question=request.question,
                store=store,
                llm=llm,
                max_iterations=request.max_iterations,
                reranker=reranker,
            ):
                if kind == "step":
                    yield sse_event("step", _step_payload(cast(AgentStep, payload)))
                else:
                    result = cast(AgentResult, payload)
                    yield sse_event(
                        "final",
                        {
                            "answer": result.answer,
                            "chunks": [_chunk_payload(r) for r in result.sources],
                            "model": result.model,
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                        },
                    )
                    trace_agent(
                        question=request.question,
                        answer=result.answer,
                        model=result.model,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        num_steps=len(result.steps),
                        num_chunks=len(result.sources),
                    )
        except LLMRateLimitError as e:
            yield sse_event("error", {"detail": f"LLM provider rate limit: {e}"})
        except LLMError as e:
            logger.error("Upstream LLM error during agent stream: %s", e)
            yield sse_event("error", {"detail": "Upstream LLM request failed"})

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
