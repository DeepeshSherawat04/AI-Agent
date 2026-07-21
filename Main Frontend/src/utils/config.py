"""
Configuration loader from environment variables.
Uses pydantic-settings for type-safe env var mapping.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Application configuration loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM API
    groq_api_key: str = ""
    openai_api_key: str = ""

    # App
    app_title: str = "TrulyIAS Scheduling Assistant"
    app_icon: str = "📅"
    debug: bool = False

    # Webhook
    webhook_url: str = ""

    # Database
    db_path: str = "data/appointments.db"

    # BUG FIX: Add checkpoint database path for LangGraph persistence
    checkpoint_db_path: str = "data/checkpoints.db"

    def get_llm_api_key(self) -> str:
        """Return the primary LLM API key (Groq preferred, fallback to OpenAI)."""
        if self.groq_api_key and self.groq_api_key.startswith("gsk_"):
            return self.groq_api_key
        if self.openai_api_key and self.openai_api_key.startswith("sk-"):
            return self.openai_api_key
        raise ValueError(
            "No valid LLM API key found. Set GROQ_API_KEY or OPENAI_API_KEY in .env"
        )

    def get_db_path(self) -> Path:
        """Return absolute path to appointments database."""
        path = Path(self.db_path)
        if not path.is_absolute():
            path = Path(__file__).parent.parent.parent / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # BUG FIX: Add missing method that persistence.py expects
    def get_checkpoint_db_path(self) -> Path:
        """Return absolute path to LangGraph checkpoint database."""
        path = Path(self.checkpoint_db_path)
        if not path.is_absolute():
            path = Path(__file__).parent.parent.parent / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# Singleton instance
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get or create the singleton AppConfig instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config