import json
import time

from locust import HttpUser, between, task


class ChatUser(HttpUser):
    host = "http://127.0.0.1:5000"

    wait_time = between(1, 3)

    def record_metric(
        self,
        name: str,
        value_ms: float,
    ) -> None:
        self.environment.events.request.fire(
            request_type="LLM Metric",
            name=name,
            response_time=value_ms,
            response_length=0,
            response=None,
            context={},
            exception=None,
        )

    @task
    def stream_chat(self) -> None:
        start_time = time.perf_counter()
        first_token_time = None
        response_size = 0
        current_event = None

        with self.client.post(
            "/chat/stream",
            json={
                "message": "Explain FastAPI simply"
            },
            stream=True,
            catch_response=True,
            name="/chat/stream - Total Latency",
            timeout=(10, 180),
        ) as response:

            if response.status_code != 200:
                response.failure(
                    f"HTTP {response.status_code}"
                )
                return

            try:
                for line in response.iter_lines(
                    decode_unicode=True,
                ):
                    if not line:
                        continue

                    response_size += len(
                        line.encode("utf-8")
                    )

                    if line.startswith("event:"):
                        current_event = line.split(
                            ":",
                            maxsplit=1,
                        )[1].strip()

                        continue

                    if not line.startswith("data:"):
                        continue

                    raw_data = line.split(
                        ":",
                        maxsplit=1,
                    )[1].strip()

                    data = json.loads(raw_data)


                    if current_event == "token":
                        if first_token_time is None:
                            first_token_time = (
                                time.perf_counter()
                            )

                    elif current_event == "error":
                        response.failure(
                            data.get(
                                "message",
                                "Unknown stream error",
                            )
                        )
                        return

                 
                    elif current_event == "done":
                        break

                end_time = time.perf_counter()

                if first_token_time is None:
                    response.failure(
                        "No token received"
                    )
                    return

                ttft_ms = (
                    first_token_time - start_time
                ) * 1000

                generation_time_ms = (
                    end_time - first_token_time
                ) * 1000

                total_latency_ms = (
                    end_time - start_time
                ) * 1000

                response.request_meta[
                    "response_time"
                ] = total_latency_ms

                response.request_meta[
                    "response_length"
                ] = response_size

                response.success()

                self.record_metric(
                    name="/chat/stream - TTFT",
                    value_ms=ttft_ms,
                )

                self.record_metric(
                    name="/chat/stream - Generation Time",
                    value_ms=generation_time_ms,
                )

            except Exception as exc:
                response.failure(
                    f"{type(exc).__name__}: {exc}"
                )