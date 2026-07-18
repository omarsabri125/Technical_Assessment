import logging
from pathlib import Path
from typing import Sequence

from ..chunking.image_aware_splitter import OneImagePerChunkSplitter
from ..ingestion.docling_parser import DoclingPdfParser
from ..ingestion.markdown_loader import MarkdownDocumentLoader
from ..infrastructure.chunk_repository import JsonlChunkRepository
from ..infrastructure.vector_store import ChromaStoreManager
from ..models import IndexingResult

logger = logging.getLogger(__name__)


class IndexingService:
    """Coordinate parsing, loading, chunking, and index persistence."""

    def __init__(
        self,
        parser: DoclingPdfParser,
        loader: MarkdownDocumentLoader,
        splitter: OneImagePerChunkSplitter,
        chunk_repository: JsonlChunkRepository,
        vector_store: ChromaStoreManager,
    ) -> None:
        self._parser = parser
        self._loader = loader
        self._splitter = splitter
        self._chunk_repository = chunk_repository
        self._vector_store = vector_store

    def index(
        self,
        pdf_paths: Sequence[str | Path] = (),
        markdown_paths: Sequence[str | Path] = (),
    ) -> IndexingResult:
        generated_markdown = self._parser.parse_many(pdf_paths)
        all_markdown_paths = self._deduplicate_paths(
            [*generated_markdown, *[Path(path) for path in markdown_paths]]
        )
        if not all_markdown_paths:
            raise ValueError("Provide at least one PDF or Markdown file to index.")

        documents = self._loader.load(all_markdown_paths)
        chunks = self._splitter.split_documents(documents)
        if not chunks:
            raise ValueError("No chunks were generated from the supplied documents.")

        self._chunk_repository.save(chunks)
        self._vector_store.rebuild(chunks)

        logger.info(
            "Index completed: %s document(s), %s chunk(s)",
            len(documents),
            len(chunks),
        )
        return IndexingResult(
            markdown_files=tuple(all_markdown_paths),
            document_count=len(documents),
            chunk_count=len(chunks),
        )

    @staticmethod
    def _deduplicate_paths(paths: Sequence[Path]) -> list[Path]:
        unique: dict[str, Path] = {}
        for path in paths:
            resolved = path.expanduser().resolve()
            unique[str(resolved)] = resolved
        return list(unique.values())
