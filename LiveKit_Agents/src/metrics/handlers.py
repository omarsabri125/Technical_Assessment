import logging
from collections.abc import Mapping
from typing import Any

from livekit.agents.metrics import (
    LLMMetrics,
    STTMetrics,
    TTSMetrics,
)


logger = logging.getLogger("agent-metrics")


def handle_llm_metrics(
    metrics: LLMMetrics,
) -> None:
    logger.info(
        "[LLM METRICS] "
        "prompt_tokens=%d | "
        "completion_tokens=%d | "
        "total_tokens=%d | "
        "tokens_per_second=%.2f | "
        "ttft=%.3fs | "
        "duration=%.3fs | "
        "cancelled=%s",
        getattr(metrics, "prompt_tokens", 0),
        getattr(metrics, "completion_tokens", 0),
        getattr(metrics, "total_tokens", 0),
        getattr(metrics, "tokens_per_second", 0.0),
        getattr(metrics, "ttft", 0.0),
        getattr(metrics, "duration", 0.0),
        getattr(metrics, "cancelled", False),
    )


def handle_stt_metrics(
    metrics: STTMetrics,
) -> None:
    duration = getattr(
        metrics,
        "duration",
        0.0,
    )

    audio_duration = getattr(
        metrics,
        "audio_duration",
        0.0,
    )

    real_time_factor = (
        duration / audio_duration
        if audio_duration > 0
        else 0.0
    )

    logger.info(
        "[STT METRICS] "
        "duration=%.3fs | "
        "audio_duration=%.3fs | "
        "real_time_factor=%.3f | "
        "streamed=%s",
        duration,
        audio_duration,
        real_time_factor,
        getattr(metrics, "streamed", False),
    )


def handle_tts_metrics(
    metrics: TTSMetrics,
) -> None:
    logger.info(
        "[TTS METRICS] "
        "ttfb=%.3fs | "
        "duration=%.3fs | "
        "audio_duration=%.3fs | "
        "characters=%d | "
        "streamed=%s | "
        "cancelled=%s",
        getattr(metrics, "ttfb", 0.0),
        getattr(metrics, "duration", 0.0),
        getattr(metrics, "audio_duration", 0.0),
        getattr(metrics, "characters_count", 0),
        getattr(metrics, "streamed", False),
        getattr(metrics, "cancelled", False),
    )


def handle_eou_metrics(
    metrics: Mapping[str, Any],
) -> None:
    transcription_delay = metrics.get(
        "transcription_delay"
    )

    end_of_turn_delay = metrics.get(
        "end_of_turn_delay"
    )

    if (
        transcription_delay is None
        and end_of_turn_delay is None
    ):
        return

    logger.info(
        "[EOU METRICS] "
        "transcription_delay=%s | "
        "end_of_turn_delay=%s",
        format_seconds(transcription_delay),
        format_seconds(end_of_turn_delay),
    )


def format_seconds(
    value: Any,
) -> str:
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return str(value)