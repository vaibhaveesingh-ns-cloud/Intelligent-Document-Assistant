"""Document normalization utilities for the assistant backend."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass
class RawDocument:
    """Represents a raw document payload prior to chunking."""

    content: str
    metadata: Dict[str, str]

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class DocumentProcessor:
    """Convert raw document payloads into LangChain `Document` chunks."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def build_documents(self, raw_documents: Iterable[RawDocument]) -> List[Document]:
        documents: List[Document] = []
        for raw in raw_documents:
            if not raw.content.strip():
                continue

            base_metadata = {
                **raw.metadata,
                "content_hash": raw.content_hash,
                "processed_at": datetime.utcnow().isoformat(),
            }
            source_document = Document(page_content=_clean_text(raw.content), metadata=base_metadata)
            chunks = self.text_splitter.split_documents([source_document])

            for index, chunk in enumerate(chunks):
                chunk.metadata.update(
                    {
                        "chunk_id": f"{chunk.metadata['content_hash']}_chunk_{index}",
                        "chunk_index": index,
                        "total_chunks": len(chunks),
                    }
                )

            documents.extend(chunks)
        return documents


def build_raw_documents(
    texts_and_metadata: Iterable[Tuple[str, Dict[str, str]]]
) -> List[RawDocument]:
    """Helper for creating `RawDocument` instances."""

    return [RawDocument(content=text, metadata=metadata) for text, metadata in texts_and_metadata]
