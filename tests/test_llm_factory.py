import pytest

from docs_rag_agent.config import Settings
from docs_rag_agent.llm import LLMClient, build_llm_client
from docs_rag_agent.llm.anthropic import AnthropicClient
from docs_rag_agent.llm.gemini import GeminiClient
from docs_rag_agent.llm.openai import OpenAIClient


def _gemini_settings(**overrides: str) -> Settings:
    return Settings(
        llm_provider="gemini",
        gemini_api_key="fake-gemini-key",
        **overrides,  # type: ignore[arg-type]
    )


def _anthropic_settings(**overrides: str) -> Settings:
    return Settings(
        llm_provider="anthropic",
        anthropic_api_key="fake-anthropic-key",
        **overrides,  # type: ignore[arg-type]
    )


def _openai_settings(**overrides: str) -> Settings:
    return Settings(
        llm_provider="openai",
        openai_api_key="fake-openai-key",
        **overrides,  # type: ignore[arg-type]
    )


def test_gemini_factory_returns_gemini_client() -> None:
    client = build_llm_client(_gemini_settings())
    assert isinstance(client, GeminiClient)


def test_anthropic_factory_returns_anthropic_client() -> None:
    client = build_llm_client(_anthropic_settings())
    assert isinstance(client, AnthropicClient)


def test_openai_factory_returns_openai_client() -> None:
    client = build_llm_client(_openai_settings())
    assert isinstance(client, OpenAIClient)


def test_all_clients_satisfy_llm_client_protocol() -> None:
    for settings in [_gemini_settings(), _anthropic_settings(), _openai_settings()]:
        client = build_llm_client(settings)
        assert isinstance(client, LLMClient)


def test_unknown_provider_raises_value_error() -> None:
    settings = Settings(llm_provider="cohere")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        build_llm_client(settings)


def test_gemini_missing_key_raises_value_error() -> None:
    settings = Settings(llm_provider="gemini", gemini_api_key="")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        build_llm_client(settings)


def test_anthropic_missing_key_raises_value_error() -> None:
    settings = Settings(llm_provider="anthropic", anthropic_api_key="")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_llm_client(settings)


def test_openai_missing_key_raises_value_error() -> None:
    settings = Settings(llm_provider="openai", openai_api_key="")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_llm_client(settings)
