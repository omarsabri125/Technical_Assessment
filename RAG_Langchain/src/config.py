import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Settings:
    """Application configuration loaded from environment variables."""

    project_root: Path
    input_dir: Path
    processed_dir: Path
    chroma_dir: Path
    chunks_file: Path

    embedding_model: str
    llm_model: str
    rerank_model: str
    chroma_collection: str

    dense_top_k: int
    sparse_top_k: int
    rerank_top_n: int
    chunk_size: int
    chunk_overlap: int

    groq_api_key: str | None
    cohere_api_key: str | None
    hugging_face_token: str | None

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Settings":
        env_path = Path(env_file) if env_file else PROJECT_ROOT / ".env"
        load_dotenv(env_path)

        data_dir = PROJECT_ROOT / "data"
        settings = cls(
            project_root=PROJECT_ROOT,
            input_dir=data_dir / "input",
            processed_dir=data_dir / "processed",
            chroma_dir=data_dir / "chroma",
            chunks_file=data_dir / "index" / "chunks.jsonl",
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            rerank_model=os.getenv("RERANK_MODEL", "rerank-english-v3.0"),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "multimodal_rag"),
            dense_top_k=int(os.getenv("DENSE_TOP_K", "4")),
            sparse_top_k=int(os.getenv("SPARSE_TOP_K", "4")),
            rerank_top_n=int(os.getenv("RERANK_TOP_N", "3")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            hugging_face_token=os.getenv("HUGGING_FACE_TOKEN"),
        )
        settings.ensure_directories()
        return settings

    def ensure_directories(self) -> None:
        for path in (
            self.input_dir,
            self.processed_dir,
            self.chroma_dir,
            self.chunks_file.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def require_groq_key(self) -> str:
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing. Add it to the .env file.")
        return self.groq_api_key
