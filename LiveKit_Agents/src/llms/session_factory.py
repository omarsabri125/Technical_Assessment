from livekit.agents import (
    AgentSession,
    TurnHandlingOptions,
    inference,
)

from ..helpers.settings import Settings
from ..llms.llm_factory import create_llm
from ..llms.stt_factory import create_stt
from ..llms.tts_factory import create_tts


def create_agent_session(
    settings: Settings,
) -> AgentSession:
    return AgentSession(
        stt=create_stt(settings),
        llm=create_llm(settings),
        tts=create_tts(settings),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),

            interruption={
                "enabled": True,
                "resume_false_interruption": True,
                "false_interruption_timeout": (
                    settings.FALSE_INTERRUPTION_TIMEOUT
                ),
            },
            preemptive_generation={
                "enabled": True,
                "max_retries": (
                    settings.PREEMPTIVE_GENERATION_MAX_RETRIES
                ),
            },
        ),
        aec_warmup_duration=settings.AEC_WARMUP_DURATION,
    )
