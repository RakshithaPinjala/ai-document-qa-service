from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Document Q&A Service"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/rag_db"

    # Providers
    LLM_PROVIDER: str = "openai"
    EMBEDDING_PROVIDER: str = "openai"
    VECTOR_DB: str = "chroma"

    # API Keys
    OPENAI_API_KEY: Optional[str] = None

    # RAG Settings
    TOP_K: int = 5
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    SIMILARITY_THRESHOLD: float = 0.75

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
