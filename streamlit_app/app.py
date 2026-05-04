"""Streamlit UI for docs-rag-agent."""
from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT = 60.0

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
    st.caption(f"Backend: {BACKEND_URL}")


def call_query(question: str, top_k: int) -> dict[str, Any]:
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(
            f"{BACKEND_URL}/query",
            json={"question": question, "top_k": top_k},
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]


def call_agent(question: str, max_iterations: int) -> dict[str, Any]:
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(
            f"{BACKEND_URL}/agent",
            json={"question": question, "max_iterations": max_iterations},
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]


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


question = st.text_area(
    "Question",
    placeholder="How do I declare path parameters in FastAPI?",
    height=100,
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Thinking..."):
            if mode == "query":
                result = call_query(question, top_k)
            else:
                result = call_agent(question, max_iterations)
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
    else:
        st.subheader("Answer")
        st.markdown(result["answer"])

        cols = st.columns(3)
        cols[0].metric("Model", result["model"])
        cols[1].metric("Input tokens", result["input_tokens"])
        cols[2].metric("Output tokens", result["output_tokens"])

        if mode == "agent" and result.get("steps"):
            st.subheader(f"Reasoning steps ({len(result['steps'])})")
            render_steps(result["steps"])

        if result.get("chunks"):
            st.subheader(f"Sources ({len(result['chunks'])})")
            render_chunks(result["chunks"])
