import logging

from livekit.agents import (
    Agent,
    RunContext,
    function_tool,
)
from livekit.agents.llm import ToolError

from ..tools.web_search_tool import (
    WebSearchError,
    WebSearchService,
)


logger = logging.getLogger("research-agent")


AGENT_INSTRUCTIONS = """
You are Lina, a concise voice research assistant.

Use the search_web tool whenever the user asks about:
- Current events
- Recent news
- Prices
- Current people or company information
- Recent technology updates
- Any fact that may have changed over time

Never claim that you searched the web unless you actually called the tool.

Safety rules:
- Never invent search results.
- If the search_web tool fails, do not answer from memory.
- Tell the user that live search is temporarily unavailable.
- If no reliable results are found, say that clearly.
- Do not present assumptions as facts.

After receiving search results:
- Answer the user's question directly.
- Keep spoken answers concise.
- Mention the source title when useful.
- Do not read long URLs aloud.
"""


class ResearchAgent(Agent):
    def __init__(
        self,
        web_search_service: WebSearchService,
    ) -> None:
        super().__init__(
            instructions=AGENT_INSTRUCTIONS,
            allow_interruptions=True,
        )

        self._web_search_service = web_search_service

    @function_tool()
    async def search_web(
        self,
        context: RunContext,
        query: str,
    ) -> str:
        """Search the live web for current information.

        Use this tool for recent news, prices, technology updates,
        current people, companies, products, events, or any
        information that may have changed.

        Args:
            query: A concise web search query describing the
                information that should be found.
        """
        del context

        normalized_query = query.strip()

        logger.info(
            "[TOOL CALL] search_web(query=%s)",
            normalized_query,
        )

        try:
            results = await self._web_search_service.search(
                normalized_query
            )

        except ValueError as error:
            logger.warning(
                "[TOOL ERROR] Invalid query: %s",
                error,
            )

            raise ToolError(
                "The search query is invalid. "
                "Ask the user to provide a clearer search request."
            ) from error

        except WebSearchError as error:
            logger.exception(
                "[TOOL ERROR] Web search failed for query: %s",
                normalized_query,
            )

            raise ToolError(
                "The live web search service is temporarily unavailable. "
                "Tell the user that the search could not be completed "
                "and ask them to try again later. "
                "Do not answer the original question from memory. "
                "Do not guess or invent search results."
            ) from error

        if not results:
            raise ToolError(
                "No reliable web search results were found. "
                "Tell the user that no relevant results were available "
                "and ask them to rephrase the question. "
                "Do not guess or invent an answer."
            )

        formatted_results: list[str] = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            title = result.get("title", "Untitled result")
            content = result.get("content", "").strip()
            url = result.get("url", "")

            if len(content) > 700:
                content = f"{content[:700]}..."

            formatted_results.append(
                "\n".join(
                    [
                        f"Result {index}",
                        f"Title: {title}",
                        f"Content: {content}",
                        f"URL: {url}",
                    ]
                )
            )

        tool_result = "\n\n".join(
            formatted_results
        )

        logger.info(
            "[TOOL RESULT] Returned %d search results.",
            len(results),
        )

        return tool_result