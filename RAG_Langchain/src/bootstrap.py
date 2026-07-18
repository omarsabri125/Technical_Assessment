from langchain_groq import ChatGroq

from .services.indexing_service import IndexingService
from .services.rag_service import RagService
from .chunking.image_aware_splitter import ImageAwareMarkdownSplitter
from .config import Settings
from .ingestion.docling_parser import DoclingPdfParser
from .ingestion.markdown_loader import MarkdownDocumentLoader
from .infrastructure.chunk_repository import JsonlChunkRepository
from .infrastructure.vector_store import ChromaStoreManager
from .prompts import FINANCIAL_QA_PROMPT
from .retrieval.hybrid_retriever import HybridRetrieverFactory


def build_vector_store(settings: Settings) -> ChromaStoreManager:
    return ChromaStoreManager(
        persist_directory=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        embedding_model_name=settings.embedding_model,
    )


def build_indexing_service(settings: Settings) -> IndexingService:
    return IndexingService(
        parser=DoclingPdfParser(settings.processed_dir),
        loader=MarkdownDocumentLoader(),
        splitter=ImageAwareMarkdownSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        chunk_repository=JsonlChunkRepository(settings.chunks_file),
        vector_store=build_vector_store(settings),
    )


def build_rag_service(settings: Settings) -> RagService:
    chunk_repository = JsonlChunkRepository(settings.chunks_file)
    chunks = chunk_repository.load()

    vector_store = build_vector_store(settings).load()
    retriever = HybridRetrieverFactory(
        dense_top_k=settings.dense_top_k,
        sparse_top_k=settings.sparse_top_k,
        rerank_top_n=settings.rerank_top_n,
        rerank_model=settings.rerank_model,
        cohere_api_key=settings.cohere_api_key,
    ).create(chunks, vector_store)

    llm = ChatGroq(
        groq_api_key=settings.require_groq_key(),
        model=settings.llm_model,
        temperature=0.0,
    )

    return RagService(
        retriever=retriever,
        llm=llm,
        prompt_template=FINANCIAL_QA_PROMPT,
        max_context_documents=settings.rerank_top_n,
    )
