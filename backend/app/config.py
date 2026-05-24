from functools import lru_cache
from os import getenv

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Runtime configuration loaded from environment variables."""

    service_name: str = "AgentEval Dashboard API"
    database_url: str
    cors_origins: list[str]

    def __init__(self) -> None:
        self.database_url = getenv("DATABASE_URL", "sqlite:///./agent_eval.db")
        self.cors_origins = self._parse_cors_origins(
            getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
        )

    @staticmethod
    def _parse_cors_origins(raw_value: str) -> list[str]:
        return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
