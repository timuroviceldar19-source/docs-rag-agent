from __future__ import annotations

import functools
import os
from typing import Any

_ENABLED: bool = (
    bool(os.getenv("LANGFUSE_SECRET_KEY")) and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
)


def is_tracing_enabled() -> bool:
    return _ENABLED


@functools.lru_cache(maxsize=1)
def _get_langfuse() -> Any:
    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]
        return Langfuse()
    except ImportError:
        return None


def trace_query(
    question: str,
    answer: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    num_chunks: int,
) -> None:
    if not _ENABLED:
        return
    lf = _get_langfuse()
    if lf is None:
        return
    trace = lf.trace(
        name="query",
        input={"question": question},
        output={"answer": answer[:200], "num_chunks": num_chunks},
    )
    trace.generation(
        name="llm_generate",
        model=model,
        input=question,
        output=answer[:200],
        usage={"input": input_tokens, "output": output_tokens},
    )
    lf.flush()


def trace_agent(
    question: str,
    answer: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    num_steps: int,
    num_chunks: int,
) -> None:
    if not _ENABLED:
        return
    lf = _get_langfuse()
    if lf is None:
        return
    lf.trace(
        name="agent",
        input={"question": question},
        output={"answer": answer[:200], "num_steps": num_steps, "num_chunks": num_chunks},
        metadata={
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    )
    lf.flush()
