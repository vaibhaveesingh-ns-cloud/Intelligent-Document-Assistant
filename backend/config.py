from dataclasses import dataclass, field
from typing import List


@dataclass
class AppConfig:
    """Runtime configuration values for the document assistant service."""

    chroma_persist_directory: str = "chroma_db"
    collection_name: str = "docuassist_documents"
    conversation_window: int = 6
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    default_embedding_model: str = "text-embedding-3-small"
    default_local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_llm_model: str = "gpt-3.5-turbo"
    allowed_origins: List[str] = field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ]
    )


config = AppConfig()
