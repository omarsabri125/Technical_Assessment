from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


class MarkdownDocumentLoader:
    """Load one or more UTF-8 Markdown files into LangChain documents."""

    def load(self, paths: Iterable[str | Path]) -> list[Document]:
        documents: list[Document] = []

        for raw_path in paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Markdown file does not exist: {path}")
            if path.suffix.lower() not in {".md", ".markdown"}:
                raise ValueError(f"Expected a Markdown file, received: {path.name}")

            loaded = TextLoader(str(path), encoding="utf-8").load()
            for document in loaded:
                document.metadata["source"] = str(path)
            documents.extend(loaded)

        return documents
