from livekit.agents import inference
from livekit.agents.tts import TTS
from livekit.plugins import elevenlabs

from ..helpers.settings import Settings


def create_tts(settings: Settings) -> TTS:
    if settings.TTS_PROVIDER == "CARTESIA":
        return inference.TTS(
            model=settings.CARTESIA_TTS_MODEL,
            voice=settings.CARTESIA_TTS_VOICE,
            language=settings.LIVEKIT_TTS_LANGUAGE,
        )

    if settings.TTS_PROVIDER == "ELEVENLABS":
        if settings.ELEVEN_API_KEY is None:
            raise ValueError(
                "ELEVEN_API_KEY is required when "
                "TTS_PROVIDER=ELEVENLABS."
            )

        return elevenlabs.TTS(
            api_key=settings.ELEVEN_API_KEY.get_secret_value(),
            model=settings.ELEVENLABS_TTS_MODEL,
            voice_id=settings.ELEVENLABS_TTS_VOICE,
            language=settings.LIVEKIT_TTS_LANGUAGE,
        )

    raise ValueError(
        f"Unsupported TTS provider: {settings.TTS_PROVIDER}"
    )