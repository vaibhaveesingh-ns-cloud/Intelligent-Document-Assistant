"""Service layer that orchestrates ingestion, embeddings, and conversation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import numpy as np

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
        # In-memory index structures
        self._embedding_descriptor: Optional[str] = None
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._known_hashes: set[str] = set()
        self._sessions: Dict[str, SessionState] = {}
        self._doc_texts: list[str] = []
        self._doc_metas: list[Dict[str, str]] = []
        self._vectors: Optional[np.ndarray] = None  # shape: (N, D)

    # ------------------------------------------------------------------
    # Embeddings + Vector Store
    # ------------------------------------------------------------------
    def _resolve_embeddings(
        self,
        *,
        use_local_embeddings: bool,  # kept for API compatibility, ignored
        embedding_model: Optional[str],
        local_embedding_model: Optional[str],  # kept for API compatibility, ignored
        openai_api_key: Optional[str],
    ):
        # Always use OpenAI embeddings to keep the image lightweight and avoid PyTorch
        model_name = embedding_model or config.default_embedding_model
        descriptor = f"openai:{model_name}"
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        embeddings = OpenAIEmbeddings(model=model_name, api_key=api_key)

        return descriptor, embeddings

    def _ensure_embedding_model(self, descriptor: str, embeddings: OpenAIEmbeddings) -> None:
        if self._embedding_descriptor is None:
            self._embedding_descriptor = descriptor
            self._embeddings = embeddings
            return
        if descriptor != self._embedding_descriptor:
            raise ValueError(
                "Existing index was built with a different embedding model. "
                "Please reset the index before switching embeddings."
            )

    def _normalize_vectors(self) -> None:
        if self._vectors is None or len(self._vectors) == 0:
            return
        # Normalize to unit vectors for cosine similarity via dot product
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = self._vectors / norms

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
        self._ensure_embedding_model(descriptor, embeddings)

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
            # Compute embeddings for all new chunks
            chunk_texts = [d.page_content for d in documents]
            vectors = np.array(self._embeddings.embed_documents(chunk_texts))  # type: ignore[union-attr]

            # Append to in-memory index
            self._doc_texts.extend(chunk_texts)
            self._doc_metas.extend([d.metadata for d in documents])
            if self._vectors is None or self._vectors.size == 0:
                self._vectors = vectors
            else:
                self._vectors = np.vstack([self._vectors, vectors])

            self._normalize_vectors()

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
        if self._vectors is None or len(self._doc_texts) == 0:
            raise ValueError("No documents have been indexed yet.")

        # Use environment variable if no API key provided
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Compute query embedding and cosine similarity
        query_vec = np.array(self._embeddings.embed_query(question))  # type: ignore[union-attr]
        qn = np.linalg.norm(query_vec)
        if qn == 0:
            qn = 1.0
        query_vec = query_vec / qn
        sims = np.dot(self._vectors, query_vec)  # type: ignore[arg-type]
        top_idx = np.argsort(-sims)[:top_k]
        retrieved = [(self._doc_texts[i], self._doc_metas[i], float(sims[i])) for i in top_idx]

        if not api_key:
            # Return retrieved context without LLM answer
            citations: List[SourceCitation] = []
            for text, meta, _score in retrieved:
                preview = text[:300]
                citations.append(
                    SourceCitation(
                        source=meta.get("source", "Unknown source"),
                        preview=preview,
                        metadata={k: str(v) for k, v in meta.items()},
                    )
                )
            return {
                "session_id": session_id,
                "answer": "Please provide an OpenAI API key to enable answer generation.",
                "sources": [citation.__dict__ for citation in citations],
            }
        
        try:
            # Build a simple prompt with retrieved context
            llm = ChatOpenAI(model=llm_model, temperature=0, api_key=api_key)
            context_blocks = []
            for i, (text, meta, score) in enumerate(retrieved, start=1):
                source = meta.get("source", f"doc_{i}")
                context_blocks.append(f"[Source: {source}]\n{text}")
            context = "\n\n".join(context_blocks)

            prompt = (
                "You are a helpful assistant. Answer the question using ONLY the context. "
                "If the answer is not present in the context, say you don't know.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )

            session = self._session(session_id)
            # Store minimal memory (question only)
            session.memory.save_context({"input": question}, {"output": ""})

            answer = llm.invoke(prompt).content  # type: ignore[attr-defined]

            citations: List[SourceCitation] = []
            for text, meta, _score in retrieved:
                preview = text[:300]
                citations.append(
                    SourceCitation(
                        source=meta.get("source", "Unknown source"),
                        preview=preview,
                        metadata={k: str(v) for k, v in meta.items()},
                    )
                )

            return {
                "session_id": session_id,
                "answer": answer,
                "sources": [citation.__dict__ for citation in citations],
            }
        except Exception as e:
            # If OpenAI fails (quota exceeded, etc.), fall back to document search
            if "quota" in str(e).lower() or "429" in str(e):
                retriever = self._vector_store.as_retriever(search_kwargs={"k": top_k})
                docs = retriever.get_relevant_documents(question)
                
                citations: List[SourceCitation] = []
                for document in docs:
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
                    "answer": "OpenAI quota exceeded. I found relevant documents but cannot generate an AI-powered answer. Please check the source documents below for information related to your question.",
                    "sources": [citation.__dict__ for citation in citations],
                }
            else:
                raise e

    def reset(self) -> None:
        """Clear cached conversation state and hashes (used for tests)."""

        self._sessions.clear()
        self._known_hashes.clear()
        self._embedding_descriptor = None
        self._embeddings = None
        self._doc_texts = []
        self._doc_metas = []
        self._vectors = None
