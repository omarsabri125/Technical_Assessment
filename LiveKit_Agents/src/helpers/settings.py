from functools import lru_cache
from typing import Literal, Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # LiveKit
    LIVEKIT_URL: str
    LIVEKIT_API_KEY: SecretStr
    LIVEKIT_API_SECRET: SecretStr

    # LLM Provider
    LLM_PROVIDER: Literal[
        "OPENROUTER",
        "GROQ",
    ]

    TEMPERATURE: float = 0.0

    # OpenRouter
    OPENROUTER_API_KEY: SecretStr | None = None
    OPENROUTER_MODEL: str = "openai/gpt-4.1-mini"

    # Groq
    GROQ_API_KEY: SecretStr | None = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    # Web Search
    TAVILY_API_KEY: SecretStr
    TAVILY_MAX_RESULTS: int = 5
    TAVILY_SEARCH_DEPTH: Literal[
        "basic",
        "advanced",
        "fast",
        "ultra-fast",
    ] = "basic"

    # STT Provider
    STT_PROVIDER: Literal[
        "DEEPGRAM",
        "ASSEMBLYAI",
    ] = "DEEPGRAM"

    # Deepgram STT
    DEEPGRAM_STT_MODEL: str = "deepgram/nova-3"
    DEEPGRAM_STT_LANGUAGE: str = "ar-EG"

    # AssemblyAI STT
    ASSEMBLYAI_STT_MODEL: str = (
        "assemblyai/universal-3-5-pro"
    )
    ASSEMBLYAI_STT_LANGUAGE: str = "ar"

    # TTS Provider
    TTS_PROVIDER: Literal[
        "CARTESIA",
        "ELEVENLABS",
    ] = "CARTESIA"

    LIVEKIT_TTS_LANGUAGE: str = "ar"

    # Cartesia TTS
    CARTESIA_TTS_MODEL: str = "cartesia/sonic-3"
    CARTESIA_TTS_VOICE: str = (
        "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
    )

    # ElevenLabs TTS
    ELEVENLABS_API_KEY: SecretStr | None = None
    ELEVENLABS_TTS_MODEL: str = "eleven_multilingual_v2"
    ELEVENLABS_TTS_VOICE: str = ""

    # Agent
    AEC_WARMUP_DURATION: float = 2.0
    FALSE_INTERRUPTION_TIMEOUT: float = 1.0
    PREEMPTIVE_GENERATION_MAX_RETRIES: int = 3

    @model_validator(mode="after")
    def validate_providers(self) -> Self:
        self._validate_llm_provider()
        self._validate_tts_provider()

        return self

    def _validate_llm_provider(self) -> None:
        if (
            self.LLM_PROVIDER == "OPENROUTER"
            and self.OPENROUTER_API_KEY is None
        ):
            raise ValueError(
                "OPENROUTER_API_KEY is required when "
                "LLM_PROVIDER=OPENROUTER."
            )

        if (
            self.LLM_PROVIDER == "GROQ"
            and self.GROQ_API_KEY is None
        ):
            raise ValueError(
                "GROQ_API_KEY is required when "
                "LLM_PROVIDER=GROQ."
            )

    def _validate_tts_provider(self) -> None:
        if self.TTS_PROVIDER == "CARTESIA":
            if not self.CARTESIA_TTS_MODEL.strip():
                raise ValueError(
                    "CARTESIA_TTS_MODEL is required when "
                    "TTS_PROVIDER=CARTESIA."
                )

            if not self.CARTESIA_TTS_VOICE.strip():
                raise ValueError(
                    "CARTESIA_TTS_VOICE is required when "
                    "TTS_PROVIDER=CARTESIA."
                )

        if self.TTS_PROVIDER == "ELEVENLABS":
            if self.ELEVENLABS_API_KEY is None:
                raise ValueError(
                    "ELEVENLABS_API_KEY is required when "
                    "TTS_PROVIDER=ELEVENLABS."
                )

            if not self.ELEVENLABS_TTS_VOICE.strip():
                raise ValueError(
                    "ELEVENLABS_TTS_VOICE is required when "
                    "TTS_PROVIDER=ELEVENLABS."
                )


@lru_cache
def get_settings() -> Settings:
    return Settings()


SETTINGS = get_settings()