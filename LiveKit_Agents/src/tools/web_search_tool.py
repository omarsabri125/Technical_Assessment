from typing import Any

from tavily import AsyncTavilyClient


class WebSearchError(RuntimeError):
    """Raised when the external web search request fails."""


class WebSearchService:
    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> None:
        self._client = AsyncTavilyClient(
            api_key=api_key,
        )

        self._max_results = max_results
        self._search_depth = search_depth

    async def search(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Search query cannot be empty."
            )

        if len(normalized_query) > 400:
            raise ValueError(
                "Search query must be 400 characters or fewer."
            )

        try:
            response = await self._client.search(
                query=normalized_query,
                search_depth=self._search_depth,
                max_results=self._max_results,
                include_answer=False,
                include_raw_content=False,
            )
        except Exception as error:
            raise WebSearchError(
                "The web search service is currently unavailable."
            ) from error

        results = response.get("results", [])

        return [
            {
                "title": result.get(
                    "title",
                    "Untitled result",
                ),
                "url": result.get(
                    "url",
                    "",
                ),
                "content": result.get(
                    "content",
                    "",
                ),
                "score": result.get(
                    "score",
                    0,
                ),
            }
            for result in results
        ]