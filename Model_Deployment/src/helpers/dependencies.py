from fastapi import HTTPException, Request, status
from openai import AsyncOpenAI


def get_openrouter_client(
    request: Request,
) -> AsyncOpenAI:
    client: AsyncOpenAI | None = getattr(
        request.app.state,
        "openrouter_client",
        None,
    )

    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenRouter client is not initialized.",
        )

    return client