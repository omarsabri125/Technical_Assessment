import logging

from livekit.agents import (
    AgentServer,
    JobContext,
    cli,
)

from .agents.research_agent import ResearchAgent
from .helpers.settings import SETTINGS
from .llms.session_factory import create_agent_session
from .tools.web_search_tool import WebSearchService


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            "agent_tool_demo.log",
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger(
    "voice-web-search-agent"
)


server = AgentServer(
    ws_url=SETTINGS.LIVEKIT_URL,
    api_key=SETTINGS.LIVEKIT_API_KEY.get_secret_value(),
    api_secret=SETTINGS.LIVEKIT_API_SECRET.get_secret_value(),
)


web_search_service = WebSearchService(
    api_key=SETTINGS.TAVILY_API_KEY.get_secret_value(),
    max_results=SETTINGS.TAVILY_MAX_RESULTS,
    search_depth=SETTINGS.TAVILY_SEARCH_DEPTH,
)


@server.rtc_session(
    agent_name="voice-web-search-agent",
)
async def entrypoint(
    ctx: JobContext,
) -> None:
    """
    Start a voice web-search session for each LiveKit room.
    """
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    logger.info(
        "Starting voice session for room: %s",
        ctx.room.name,
    )

    agent = ResearchAgent(
        web_search_service=web_search_service,
    )

    session = create_agent_session(
        settings=SETTINGS,
    )

    await session.start(
        room=ctx.room,
        agent=agent,
    )

    logger.info(
        "Voice web-search session started successfully."
    )

    await session.generate_reply(
        instructions=(
            "Greet the user briefly. Tell them that you can "
            "search the web for current and accurate information."
        ),
        allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(server)