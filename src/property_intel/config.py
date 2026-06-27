from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = (
        "postgresql+psycopg://property_intel:property_intel_dev_password"
        "@localhost:5432/property_intel"
    )
    data_raw_dir: Path = REPO_ROOT / "data" / "raw"
    data_processed_dir: Path = REPO_ROOT / "data" / "processed"
    log_level: str = "INFO"
    log_dir: Path = REPO_ROOT / "logs"
    openai_embedding_model: str = "text-embedding-3-small"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "document_chunks"
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-v3.5"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.0
    openai_max_tokens: int = 1024

    @staticmethod
    def _resolve(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    def resolved_data_raw_dir(self) -> Path:
        return self._resolve(self.data_raw_dir)

    def resolved_data_processed_dir(self) -> Path:
        return self._resolve(self.data_processed_dir)

    def resolved_log_dir(self) -> Path:
        return self._resolve(self.log_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
