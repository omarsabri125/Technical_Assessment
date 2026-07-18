import json
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document


class JsonlChunkRepository:
    """Persist retrieval chunks independently from the vector database."""

    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path

    def save(self, documents: Sequence[Document]) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_path.open("w", encoding="utf-8") as file:
            for document in documents:
                record = {
                    "page_content": document.page_content,
                    "metadata": document.metadata,
                }
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def load(self) -> list[Document]:
        if not self._file_path.exists():
            raise FileNotFoundError(
                f"Chunk index does not exist: {self._file_path}. Run the index command first."
            )

        documents: list[Document] = []
        with self._file_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    documents.append(
                        Document(
                            page_content=record["page_content"],
                            metadata=record.get("metadata", {}),
                        )
                    )
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(
                        f"Invalid chunk data at line {line_number} in {self._file_path}"
                    ) from exc

        return documents
