import json
import time
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from ..helpers.settings import Settings


class ChatService:
    def __init__(
        self,
        client: AsyncOpenAI,
        settings: Settings,
    ) -> None:
        self.client = client
        self.settings = settings

    async def stream_message(
        self,
        message: str,
    ) -> AsyncIterator[str]:

        start_time = time.perf_counter()
        first_token_time = None

        input_tokens = 0
        output_tokens = 0

        try:
            stream = await self.client.chat.completions.create(
                model=self.settings.OPEN_ROUTER_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
                stream=True,
                stream_options={
                    "include_usage": True,
                },
            )

            async for chunk in stream:

                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

                if not chunk.choices:
                    continue

                token = chunk.choices[0].delta.content

                if token:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    token_data = json.dumps(
                        {
                            "token": token,
                        },
                        ensure_ascii=False,
                    )

                    yield (
                        "event: token\n"
                        f"data: {token_data}\n\n"
                    )

            end_time = time.perf_counter()

            if first_token_time is not None:
                ttft = first_token_time - start_time
                generation_time = end_time - first_token_time
            else:
                ttft = 0
                generation_time = 0

            total_latency = end_time - start_time

            metrics_data = json.dumps(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "ttft_ms": round(ttft * 1000, 2),
                    "generation_time_ms": round(
                        generation_time * 1000,
                        2,
                    ),
                    "total_latency_ms": round(
                        total_latency * 1000,
                        2,
                    ),
                },
                ensure_ascii=False,
            )

            yield (
                "event: metrics\n"
                f"data: {metrics_data}\n\n"
            )

            yield (
                "event: done\n"
                'data: {"status": "completed"}\n\n'
            )

        except Exception as exc:
            error_data = json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
            )

            yield (
                "event: error\n"
                f"data: {error_data}\n\n"
            )