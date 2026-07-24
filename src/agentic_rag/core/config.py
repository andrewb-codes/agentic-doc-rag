import os
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agentic Doc RAG"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"
    sql_log_level: str = "WARNING"

    database_url: str

    internal_api_key: str

    max_upload_size_bytes: int = 10 * 1024 * 1024

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "document_chunks"

    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"

    backend_cors_origins: str = ""

    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        if not self.backend_cors_origins:
            return []

        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]


settings = Settings()  # type: ignore[call-arg]
