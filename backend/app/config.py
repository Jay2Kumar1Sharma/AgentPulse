from functools import lru_cache
from os import getenv

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Runtime configuration loaded from environment variables."""

    service_name: str = "AgentEval Dashboard API"
    database_url: str
    cors_origins: list[str]
    llm_provider: str
    llm_judge_enabled: bool
    llm_judge_model: str
    openai_api_key: str
    anthropic_api_key: str
    google_api_key: str

    def __init__(self) -> None:
        self.database_url = getenv("DATABASE_URL", "sqlite:///./agent_eval.db")
        self.cors_origins = self._parse_cors_origins(
            getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
        )
        self.llm_provider = getenv("LLM_PROVIDER", "openai").strip().lower()
        self.llm_judge_enabled = self._parse_bool(getenv("LLM_JUDGE_ENABLED", "false"))
        self.llm_judge_model = getenv("LLM_JUDGE_MODEL", "gpt-4o-mini").strip()
        self.openai_api_key = getenv("OPENAI_API_KEY", "").strip()
        self.anthropic_api_key = getenv("ANTHROPIC_API_KEY", "").strip()
        self.google_api_key = getenv("GOOGLE_API_KEY", "").strip()

    @staticmethod
    def _parse_cors_origins(raw_value: str) -> list[str]:
        return [origin.strip() for origin in raw_value.split(",") if origin.strip()]

    @staticmethod
    def _parse_bool(raw_value: str) -> bool:
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}

    def llm_api_key_for_provider(self) -> str:
        provider_keys = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.google_api_key,
            "google": self.google_api_key,
        }
        return provider_keys.get(self.llm_provider, "")


@lru_cache
def get_settings() -> Settings:
    return Settings()
