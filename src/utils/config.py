"""
Configuration management for GigaCorp Support Agent.
Uses pydantic-settings for environment-based configuration.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # App Config
    app_name: str = "GigaCorp Support Agent"
    app_debug: bool = False
    
    # LLM Config (Groq)
    groq_api_key: str
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024
    
    # Embedding Config (Local HuggingFace)
    embedding_model: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    
    # Vector Store Config
    vector_store_path: str = "./data/vector_db"
    
    # Memory Config
    memory_type: str = "buffer_window"
    memory_k: int = 5
    
    # Retrieval Config
    retrieval_k: int = 4
    retrieval_score_threshold: Optional[float] = None


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()