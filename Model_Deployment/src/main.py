from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import AsyncOpenAI

from .helpers.settings import get_settings
from .routes.chat import router as chat_router


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    default_headers: dict[str, str] = {}

    if settings.OPENROUTER_SITE_URL:
        default_headers["HTTP-Referer"] = (
            settings.OPENROUTER_SITE_URL
        )

    if settings.OPENROUTER_SITE_NAME:
        default_headers["X-OpenRouter-Title"] = (
            settings.OPENROUTER_SITE_NAME
        )

    client = AsyncOpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=(
            settings
            .OPEN_ROUTER_API_KEY
        ),
        default_headers=default_headers,
        timeout=300.0,
        max_retries=2,
    )

    app.state.openrouter_client = client

    try:
        yield
    finally:
        await client.close()


app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)


app.include_router(chat_router)


@app.get(
    "/health",
    tags=["Operations"],
)
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "provider": "OpenRouter",
        "model": settings.OPEN_ROUTER_MODEL_NAME,
    }