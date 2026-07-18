from livekit.agents import inference

from ..helpers.settings import Settings


def create_stt(settings: Settings):
    """
    Create the configured speech-to-text provider.
    """

    if settings.STT_PROVIDER == "DEEPGRAM":
        return inference.STT(
            model=settings.DEEPGRAM_STT_MODEL,
            language=settings.DEEPGRAM_STT_LANGUAGE,
        )

    if settings.STT_PROVIDER == "ASSEMBLYAI":
        return inference.STT(
            model=settings.ASSEMBLYAI_STT_MODEL,
            language=settings.ASSEMBLYAI_STT_LANGUAGE,
        )

    raise ValueError(
        f"Unsupported STT provider: {settings.STT_PROVIDER}"
    )