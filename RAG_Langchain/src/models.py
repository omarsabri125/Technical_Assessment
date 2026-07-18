from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document


@dataclass(frozen=True, slots=True)
class IndexingResult:
    markdown_files: tuple[Path, ...]
    document_count: int
    chunk_count: int


@dataclass(frozen=True, slots=True)
class RagAnswer:
    question: str
    answer: str
    retrieved_documents: tuple[Document, ...]
    image_paths: tuple[Path, ...]
