from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "gemini"
    api_key: str = "dev-key"  # Agent API Key
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "fastapi_docs"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_fetch_k: int = 20  # over-fetch from vector store, then rerank to top_k
    log_level: str = "INFO"
