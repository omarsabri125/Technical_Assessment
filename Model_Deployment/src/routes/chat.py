from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from ..helpers.dependencies import get_openrouter_client
from ..helpers.settings import Settings, get_settings
from ..schemas.chat import ChatRequest
from ..services.stream_service import ChatService


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post(
    "/stream",
    response_class=StreamingResponse,
)
async def chat_stream(
    request: ChatRequest,
    client: Annotated[
        AsyncOpenAI,
        Depends(get_openrouter_client),
    ],
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> StreamingResponse:
    service = ChatService(
        client=client,
        settings=settings,
    )

    return StreamingResponse(
        service.stream_message(request.message),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )