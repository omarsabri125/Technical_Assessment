from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..models import RagAnswer


class RagService:
    """Retrieve evidence and generate a grounded answer."""

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        prompt_template: str,
        max_context_documents: int = 3,
    ) -> None:
        self._retriever = retriever
        self._max_context_documents = max_context_documents
        self._answer_chain = (
            ChatPromptTemplate.from_template(prompt_template)
            | llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> RagAnswer:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty.")

        retrieved = list(self._retriever.invoke(normalized_question))
        context_documents = retrieved[: self._max_context_documents]
        context = self._format_documents(context_documents)
        answer = self._answer_chain.invoke(
            {
                "context": context,
                "question": normalized_question,
            }
        )

        return RagAnswer(
            question=normalized_question,
            answer=answer,
            retrieved_documents=tuple(retrieved),
            image_paths=self._extract_image_paths(retrieved),
        )

    @staticmethod
    def _format_documents(documents: Sequence[Document]) -> str:
        return "\n\n".join(
            f"[{index}]\n{document.page_content}"
            for index, document in enumerate(documents, start=1)
        )

    @staticmethod
    def _extract_image_paths(documents: Sequence[Document]) -> tuple[Path, ...]:
        unique_paths: dict[str, Path] = {}
        for document in documents:
            if not document.metadata.get("has_image"):
                continue
            raw_path = document.metadata.get("image_path")
            if not raw_path:
                continue
            path = Path(str(raw_path))
            unique_paths[str(path)] = path
        return tuple(unique_paths.values())
