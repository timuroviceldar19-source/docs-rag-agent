from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from docs_rag_agent.agent.loop import run_agent
from docs_rag_agent.api.dependencies import get_llm_client, get_vector_store
from docs_rag_agent.llm import Message
from docs_rag_agent.tracing import trace_agent, trace_query

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

@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    store = get_vector_store()
    llm = get_llm_client()
    
    results = store.search(request.question, top_k=request.top_k)
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

@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(request: AgentRequest) -> AgentResponse:
    store = get_vector_store()
    llm = get_llm_client()
    result = run_agent(
        question=request.question,
        store=store,
        llm=llm,
        max_iterations=request.max_iterations,
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
