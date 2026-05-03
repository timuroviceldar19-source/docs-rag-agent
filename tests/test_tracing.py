from docs_rag_agent.tracing import is_tracing_enabled, trace_agent, trace_query


def test_tracing_disabled_without_env_keys() -> None:
    # In test environment, LANGFUSE_SECRET_KEY is not set.
    assert not is_tracing_enabled()


def test_trace_query_is_noop_without_keys() -> None:
    # Must not raise.
    trace_query(
        question="What is FastAPI?",
        answer="FastAPI is a web framework.",
        model="test-model",
        input_tokens=10,
        output_tokens=5,
        num_chunks=3,
    )


def test_trace_agent_is_noop_without_keys() -> None:
    # Must not raise.
    trace_agent(
        question="What is FastAPI?",
        answer="FastAPI is a web framework.",
        model="test-model",
        input_tokens=20,
        output_tokens=10,
        num_steps=2,
        num_chunks=4,
    )


def test_tracing_module_exports_expected_symbols() -> None:
    import docs_rag_agent.tracing as tracing_module
    assert callable(tracing_module.trace_query)
    assert callable(tracing_module.trace_agent)
    assert callable(tracing_module.is_tracing_enabled)
