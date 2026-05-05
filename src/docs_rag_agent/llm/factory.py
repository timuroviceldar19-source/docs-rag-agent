from docs_rag_agent.config import Settings
from docs_rag_agent.llm.anthropic import AnthropicClient
from docs_rag_agent.llm.base import LLMClient
from docs_rag_agent.llm.gemini import GeminiClient
from docs_rag_agent.llm.openai import OpenAIClient


def build_llm_client(settings: Settings) -> LLMClient:
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        return GeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        return AnthropicClient(api_key=settings.anthropic_api_key, model=settings.anthropic_model)
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return OpenAIClient(api_key=settings.openai_api_key, model=settings.openai_model)
    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        return OpenAIClient(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            base_url="https://api.groq.com/openai/v1",
        )
    raise ValueError(
        f"Unknown LLM_PROVIDER={settings.llm_provider!r}. Allowed: gemini, anthropic, openai, groq."
    )
