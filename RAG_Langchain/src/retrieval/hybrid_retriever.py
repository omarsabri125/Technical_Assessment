from typing import Any, Sequence

from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever


class HybridRetrieverFactory:
    """Create dense + sparse retrieval with optional Cohere reranking."""

    def __init__(
        self,
        dense_top_k: int,
        sparse_top_k: int,
        rerank_top_n: int,
        rerank_model: str,
        cohere_api_key: str | None,
    ) -> None:
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._rerank_top_n = rerank_top_n
        self._rerank_model = rerank_model
        self._cohere_api_key = cohere_api_key

    def create(self, documents: Sequence[Document], vector_store: Any) -> Any:
        if not documents:
            raise ValueError("Cannot build a retriever without indexed chunks.")

        sparse_retriever = BM25Retriever.from_documents(list(documents))
        sparse_retriever.k = self._sparse_top_k

        dense_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self._dense_top_k},
        )

        ensemble = EnsembleRetriever(
            retrievers=[sparse_retriever, dense_retriever],
            weights=[0.5, 0.5],
        )

        if not self._cohere_api_key:
            return ensemble

        reranker = CohereRerank(
            model=self._rerank_model,
            top_n=self._rerank_top_n,
            cohere_api_key=self._cohere_api_key,
        )
        return ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=ensemble,
        )
