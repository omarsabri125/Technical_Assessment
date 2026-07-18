from livekit.agents.llm import LLM
from livekit.plugins import openai

from ..helpers.settings import Settings


def create_llm(settings: Settings) -> LLM:
    if settings.LLM_PROVIDER == "OPENROUTER":
        assert settings.OPENROUTER_API_KEY is not None

        return openai.LLM.with_openrouter(
            model=settings.OPENROUTER_MODEL,
            api_key=(
                settings.OPENROUTER_API_KEY.get_secret_value()
            ),
            temperature=settings.TEMPERATURE,
        )

    if settings.LLM_PROVIDER == "GROQ":
        assert settings.GROQ_API_KEY is not None

        return openai.LLM(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY.get_secret_value(),
            base_url=settings.GROQ_BASE_URL,
            temperature=settings.TEMPERATURE,
        )

    raise ValueError(
        f"Unsupported LLM provider: {settings.LLM_PROVIDER}"
    )