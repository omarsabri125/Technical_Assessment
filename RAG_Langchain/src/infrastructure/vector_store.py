import shutil
from pathlib import Path
from typing import Sequence

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class ChromaStoreManager:
    """Create and load the persistent Chroma collection."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embedding_model_name: str,
    ) -> None:
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    def rebuild(self, documents: Sequence[Document]) -> Chroma:
        if not documents:
            raise ValueError("Cannot build a vector store without documents.")

        if self._persist_directory.exists():
            shutil.rmtree(self._persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)

        return Chroma.from_documents(
            documents=list(documents),
            embedding=self._embedding_model,
            persist_directory=str(self._persist_directory),
            collection_name=self._collection_name,
        )

    def load(self) -> Chroma:
        if not self._persist_directory.exists():
            raise FileNotFoundError(
                f"Chroma directory does not exist: {self._persist_directory}. "
                "Run the index command first."
            )

        return Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embedding_model,
            persist_directory=str(self._persist_directory),
        )
