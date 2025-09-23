"""FastAPI application powering the DocuAssist experience."""

from __future__ import annotations

import logging
import traceback
from typing import List, Optional
from uuid import uuid4

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import config
from backend.services.assistant import DocumentAssistant
from backend.services.loaders import MissingDependencyError, UnsupportedFileTypeError, SUPPORTED_EXTENSIONS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DocuAssist API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = DocumentAssistant()


class UploadResponse(BaseModel):
    ingested: int
    skipped: int


class SourceSchema(BaseModel):
    source: str
    preview: str
    metadata: dict


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    top_k: int = Field(default=4, ge=1, le=10)
    llm_model: Optional[str] = None
    openai_api_key: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceSchema]


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok"}

@app.get("/api/test-assistant")
def test_assistant() -> dict:
    try:
        # Test if assistant can be initialized
        test_assistant = DocumentAssistant()
        return {"status": "ok", "message": "Assistant initialized successfully"}
    except Exception as exc:
        logger.error(f"Assistant initialization failed: {exc}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"status": "error", "message": str(exc)}


@app.get("/api/config")
def configuration() -> dict:
    return {
        "supported_extensions": SUPPORTED_EXTENSIONS,
        "defaults": {
            "chunk_size": config.default_chunk_size,
            "chunk_overlap": config.default_chunk_overlap,
            "embedding_model": config.default_embedding_model,
            "local_embedding_model": config.default_local_embedding_model,
            "llm_model": config.default_llm_model,
        },
    }


@app.post("/api/documents", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(config.default_chunk_size),
    chunk_overlap: int = Form(config.default_chunk_overlap),
    embedding_model: Optional[str] = Form(None),
    local_embedding_model: Optional[str] = Form(None),
    use_local_embeddings: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    uploads = []
    for file in files:
        uploads.append((file.filename, await file.read()))

    try:
        logger.info(f"Processing {len(uploads)} files for upload")
        result = assistant.ingest_documents(
            uploads,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_local_embeddings=use_local_embeddings,
            embedding_model=embedding_model,
            local_embedding_model=local_embedding_model,
            openai_api_key=openai_api_key,
        )
        logger.info(f"Successfully processed upload: {result}")
    except UnsupportedFileTypeError as exc:  # pragma: no cover - runtime validation
        logger.error(f"Unsupported file type: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MissingDependencyError as exc:  # pragma: no cover - runtime validation
        logger.error(f"Missing dependency: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        logger.error(f"Value error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Unexpected error during upload: {exc}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}") from exc

    return UploadResponse(**result)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid4())
    llm_model = request.llm_model or config.default_llm_model
    try:
        payload = assistant.ask_question(
            session_id=session_id,
            question=request.question,
            top_k=request.top_k,
            llm_model=llm_model,
            openai_api_key=request.openai_api_key,
        )
    except ValueError as exc:
        logger.error(f"ValueError in chat endpoint: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Unexpected error in chat endpoint: {exc}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}") from exc

    sources = [SourceSchema(**source) for source in payload["sources"]]
    return ChatResponse(session_id=session_id, answer=payload["answer"], sources=sources)
