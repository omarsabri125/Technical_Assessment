from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import ValidationError
from typing import Optional


class Settings(BaseSettings):
    APP_NAME: str
    OPEN_ROUTER_API_KEY: Optional[str] = None
    OPEN_ROUTER_MODEL_NAME: str = (
        "nvidia/nemotron-3-ultra-550b-a55b:free"
    )
    OPENROUTER_SITE_URL: str | None = None
    OPENROUTER_SITE_NAME: str = "FastAPI OpenRouter API"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    APP_DESCRIPTION: str = (
        "Streams model responses through OpenRouter."
    )
    APP_VERSION: str = "1.0.0"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as e:
        missing_fields = [err["loc"][0] for err in e.errors()]

        raise RuntimeError(
            f"❌ Missing required environment variables: {', '.join(missing_fields)}\n"
            f"👉 Make sure they exist in your .env file"
        )
