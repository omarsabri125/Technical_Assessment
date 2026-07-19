import logging

from livekit.agents import (
    AgentSession,
    ConversationItemAddedEvent,
)
from livekit.agents.llm import ChatMessage

from ..metrics.handlers import (
    handle_eou_metrics,
    handle_llm_metrics,
    handle_stt_metrics,
    handle_tts_metrics,
)


logger = logging.getLogger("metrics-registry")


def register_metrics(
    session: AgentSession,
) -> None:
    """
    Register LLM, STT, TTS, and EOU metrics listeners.
    """

    session.llm.on(
        "metrics_collected",
        handle_llm_metrics,
    )

    session.stt.on(
        "metrics_collected",
        handle_stt_metrics,
    )

    session.tts.on(
        "metrics_collected",
        handle_tts_metrics,
    )

    def conversation_item_wrapper(
        event: ConversationItemAddedEvent,
    ) -> None:
        item = event.item

        if not isinstance(item, ChatMessage):
            return

        if item.role != "user":
            return

        item_metrics = getattr(
            item,
            "metrics",
            None,
        )

        if not item_metrics:
            return

        handle_eou_metrics(
            item_metrics
        )

    session.on(
        "conversation_item_added",
        conversation_item_wrapper,
    )

    logger.info(
        "[METRICS] Metrics listeners registered."
    )