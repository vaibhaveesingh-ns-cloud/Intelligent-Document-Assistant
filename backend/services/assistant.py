"""Service layer that orchestrates ingestion, embeddings, and conversation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend.config import config
from backend.services.loaders import extract_text_from_bytes
from backend.services.processor import DocumentProcessor, RawDocument, build_raw_documents


@dataclass
class SourceCitation:
    """Represents a source snippet returned to the client."""

    source: str
    preview: str
    metadata: Dict[str, str]


@dataclass
class SessionState:
    """Tracks per-session conversation context."""

    memory: ConversationBufferWindowMemory
    last_activity: float = field(default_factory=lambda: time.time())


class DocumentAssistant:
    """Facade that coordinates document ingestion and conversational retrieval."""

    def __init__(self) -> None:
        self._vector_store: Optional[Chroma] = None
        self._embedding_descriptor: Optional[str] = None
        self._known_hashes: set[str] = set()
        self._sessions: Dict[str, SessionState] = {}

    # ------------------------------------------------------------------
    # Embeddings + Vector Store
    # ------------------------------------------------------------------
    def _resolve_embeddings(
        self,
        *,
        use_local_embeddings: bool,
        embedding_model: Optional[str],
        local_embedding_model: Optional[str],
        openai_api_key: Optional[str],
    ):
        if use_local_embeddings:
            model_name = local_embedding_model or config.default_local_embedding_model
            descriptor = f"local:{model_name}"
            embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        else:
            model_name = embedding_model or config.default_embedding_model
            descriptor = f"openai:{model_name}"
            embeddings = OpenAIEmbeddings(model=model_name, api_key=openai_api_key)

        return descriptor, embeddings

    def _ensure_vector_store(self, descriptor: str, embeddings) -> Chroma:
        if self._vector_store is None:
            import chromadb
            from chromadb.config import Settings
            
            # Create persistent Chroma client
            chroma_client = chromadb.PersistentClient(
                path=config.chroma_persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self._vector_store = Chroma(
                client=chroma_client,
                collection_name=config.collection_name,
                embedding_function=embeddings,
            )
            self._embedding_descriptor = descriptor
            self._hydrate_hash_cache()
        elif descriptor != self._embedding_descriptor:
            raise ValueError(
                "Existing index was built with a different embedding model. "
                "Please reset the storage directory before switching embeddings."
            )
        return self._vector_store

    def _hydrate_hash_cache(self) -> None:
        if not self._vector_store:
            return
        try:
            existing = self._vector_store.get(include=["metadatas"])
        except Exception:
            existing = None
        if not existing:
            return
        for metadata in existing.get("metadatas", []):
            if metadata and metadata.get("content_hash"):
                self._known_hashes.add(metadata["content_hash"])

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_documents(
        self,
        uploads: Iterable[Tuple[str, bytes]],
        *,
        chunk_size: int,
        chunk_overlap: int,
        use_local_embeddings: bool,
        embedding_model: Optional[str],
        local_embedding_model: Optional[str],
        openai_api_key: Optional[str],
    ) -> Dict[str, int]:
        descriptor, embeddings = self._resolve_embeddings(
            use_local_embeddings=use_local_embeddings,
            embedding_model=embedding_model,
            local_embedding_model=local_embedding_model,
            openai_api_key=openai_api_key,
        )
        vector_store = self._ensure_vector_store(descriptor, embeddings)

        texts_and_metadata: List[Tuple[str, Dict[str, str]]] = []
        skipped = 0

        for filename, file_bytes in uploads:
            text = extract_text_from_bytes(filename, file_bytes)
            if not text:
                skipped += 1
                continue

            raw_doc = RawDocument(
                content=text,
                metadata={
                    "source": filename,
                    "filename": filename,
                },
            )
            if raw_doc.content_hash in self._known_hashes:
                skipped += 1
                continue

            self._known_hashes.add(raw_doc.content_hash)
            texts_and_metadata.append((raw_doc.content, raw_doc.metadata))

        if not texts_and_metadata:
            return {"ingested": 0, "skipped": skipped}

        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = processor.build_documents(build_raw_documents(texts_and_metadata))

        if documents:
            vector_store.add_documents(documents)
            # ChromaDB persistence is handled automatically with PersistentClient

        return {"ingested": len(documents), "skipped": skipped}

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------
    def _session(self, session_id: str) -> SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            state = SessionState(
                memory=ConversationBufferWindowMemory(
                    k=config.conversation_window,
                    memory_key="chat_history",
                    return_messages=True,
                )
            )
            self._sessions[session_id] = state
        state.last_activity = time.time()
        return state

    def ask_question(
        self,
        *,
        session_id: str,
        question: str,
        top_k: int,
        llm_model: str,
        openai_api_key: Optional[str],
    ) -> Dict[str, object]:
        if not self._vector_store:
            raise ValueError("No documents have been indexed yet.")

        llm = ChatOpenAI(model=llm_model, temperature=0, api_key=openai_api_key)
        retriever = self._vector_store.as_retriever(search_kwargs={"k": top_k})
        session = self._session(session_id)

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=session.memory,
            return_source_documents=True,
        )
        result = chain.invoke({"question": question})
        answer = result.get("answer", "")
        source_documents = result.get("source_documents", [])

        citations: List[SourceCitation] = []
        for document in source_documents:
            metadata = document.metadata or {}
            preview = document.page_content[:300]
            citations.append(
                SourceCitation(
                    source=metadata.get("source", "Unknown source"),
                    preview=preview,
                    metadata={k: str(v) for k, v in metadata.items()},
                )
            )

        return {
            "session_id": session_id,
            "answer": answer,
            "sources": [citation.__dict__ for citation in citations],
        }

    def reset(self) -> None:
        """Clear cached conversation state and hashes (used for tests)."""

        self._sessions.clear()
        self._known_hashes.clear()
        if self._vector_store is not None:
            self._vector_store.delete_collection()
            self._vector_store = None
            self._embedding_descriptor = None
