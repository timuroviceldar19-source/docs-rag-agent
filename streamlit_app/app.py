"""Streamlit UI for docs-rag-agent."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-key")
TIMEOUT = 60.0
STREAM_TIMEOUT = 120.0

st.set_page_config(
    page_title="docs-rag-agent",
    page_icon="📚",
    layout="wide",
)

st.title("📚 docs-rag-agent")
st.caption("RAG + ReAct agent over FastAPI documentation")

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Mode",
        ["query", "agent"],
        help="`/query` is single-step RAG. `/agent` runs a multi-step ReAct loop.",
    )
    if mode == "query":
        top_k = st.slider("Top-K chunks", 1, 20, 5)
        max_iterations = 5
    else:
        max_iterations = st.slider("Max iterations", 1, 10, 5)
        top_k = 5
    stream = st.checkbox("Stream response", value=True)
    st.caption(f"Backend: {BACKEND_URL}")


# ---------------------------------------------------------------------------
# Backend calls
# ---------------------------------------------------------------------------


def call_query(question: str, top_k: int) -> dict[str, Any]:
    with httpx.Client(timeout=TIMEOUT, headers={"X-API-Key": API_KEY}) as client:
        response = client.post(
            f"{BACKEND_URL}/query",
            json={"question": question, "top_k": top_k},
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]


def call_agent(question: str, max_iterations: int) -> dict[str, Any]:
    with httpx.Client(timeout=TIMEOUT, headers={"X-API-Key": API_KEY}) as client:
        response = client.post(
            f"{BACKEND_URL}/agent",
            json={"question": question, "max_iterations": max_iterations},
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]


def _iter_sse(
    url: str, payload: dict[str, Any]
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (event_name, data_dict) tuples from an SSE stream."""
    headers = {"X-API-Key": API_KEY, "Accept": "text/event-stream"}
    with (
        httpx.Client(timeout=STREAM_TIMEOUT, headers=headers) as client,
        client.stream("POST", url, json=payload) as response,
    ):
        response.raise_for_status()
        event_name = "message"
        data_lines: list[str] = []
        for line in response.iter_lines():
            if line == "":
                if data_lines:
                    raw = "\n".join(data_lines)
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        data = {"raw": raw}
                    yield event_name, data
                event_name = "message"
                data_lines = []
                continue
            if line.startswith(":"):
                continue  # SSE comment / keep-alive
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_chunks(chunks: list[dict[str, Any]]) -> None:
    for i, chunk in enumerate(chunks, 1):
        label = f"#{i} · {chunk['source']} · score={chunk['score']:.3f}"
        with st.expander(label):
            if chunk.get("heading"):
                st.markdown(f"**{chunk['heading']}**")
            st.text(chunk["text"])


def render_steps(steps: list[dict[str, Any]]) -> None:
    for i, step in enumerate(steps, 1):
        title = f"Step {i}: {step.get('action') or 'final answer'}"
        with st.expander(title):
            st.markdown(f"**Thought:** {step['thought']}")
            if step.get("action"):
                st.markdown(f"**Action:** `{step['action']}`")
                st.json(step.get("action_input", {}))
            if step.get("observation"):
                st.markdown("**Observation:**")
                st.text(step["observation"])
            if step.get("final_answer"):
                st.markdown(f"**Final Answer:** {step['final_answer']}")


def render_metrics(model: str, input_tokens: int, output_tokens: int) -> None:
    cols = st.columns(3)
    cols[0].metric("Model", model)
    cols[1].metric("Input tokens", input_tokens)
    cols[2].metric("Output tokens", output_tokens)


# ---------------------------------------------------------------------------
# Stream handlers
# ---------------------------------------------------------------------------


def run_query_stream(question: str, top_k: int) -> None:
    payload = {"question": question, "top_k": top_k}
    st.subheader("Answer")
    answer_box = st.empty()
    accumulated: list[str] = []
    chunks: list[dict[str, Any]] = []
    final: dict[str, Any] = {}

    for event, data in _iter_sse(f"{BACKEND_URL}/query/stream", payload):
        if event == "chunks":
            chunks = data.get("chunks", [])
        elif event == "token":
            accumulated.append(data.get("text", ""))
            answer_box.markdown("".join(accumulated))
        elif event == "end":
            final = data
        elif event == "error":
            st.error(data.get("detail", "Unknown stream error"))
            return

    if not accumulated:
        answer_box.info("Empty response from model.")
    render_metrics(
        final.get("model", ""),
        int(final.get("input_tokens", 0)),
        int(final.get("output_tokens", 0)),
    )
    if chunks:
        st.subheader(f"Sources ({len(chunks)})")
        render_chunks(chunks)


def run_agent_stream(question: str, max_iterations: int) -> None:
    payload = {"question": question, "max_iterations": max_iterations}
    steps_box = st.empty()
    answer_box = st.empty()
    received_steps: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None

    for event, data in _iter_sse(f"{BACKEND_URL}/agent/stream", payload):
        if event == "step":
            received_steps.append(data)
            with steps_box.container():
                st.subheader(f"Reasoning steps ({len(received_steps)})")
                render_steps(received_steps)
        elif event == "final":
            final = data
        elif event == "error":
            st.error(data.get("detail", "Unknown stream error"))
            return

    if final is None:
        st.warning("Stream ended without a final answer.")
        return

    with answer_box.container():
        st.subheader("Answer")
        st.markdown(final.get("answer", ""))
        render_metrics(
            final.get("model", ""),
            int(final.get("input_tokens", 0)),
            int(final.get("output_tokens", 0)),
        )
        chunks = final.get("chunks", [])
        if chunks:
            st.subheader(f"Sources ({len(chunks)})")
            render_chunks(chunks)


def run_blocking(question: str) -> None:
    with st.spinner("Thinking..."):
        if mode == "query":
            result = call_query(question, top_k)
        else:
            result = call_agent(question, max_iterations)
    st.subheader("Answer")
    st.markdown(result["answer"])
    render_metrics(result["model"], result["input_tokens"], result["output_tokens"])
    if mode == "agent" and result.get("steps"):
        st.subheader(f"Reasoning steps ({len(result['steps'])})")
        render_steps(result["steps"])
    if result.get("chunks"):
        st.subheader(f"Sources ({len(result['chunks'])})")
        render_chunks(result["chunks"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

question = st.text_area(
    "Question",
    placeholder="How do I declare path parameters in FastAPI?",
    height=100,
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        if stream:
            if mode == "query":
                run_query_stream(question, top_k)
            else:
                run_agent_stream(question, max_iterations)
        else:
            run_blocking(question)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            st.warning("No relevant chunks found for this question. Try rephrasing.")
        else:
            st.error(f"Backend returned {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        st.error(
            f"Cannot reach backend at {BACKEND_URL}. "
            f"Is `uvicorn docs_rag_agent.api.main:app` running?\n\n{e}"
        )
