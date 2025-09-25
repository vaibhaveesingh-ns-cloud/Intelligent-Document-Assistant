"""Utilities for converting uploaded files into plain text (PDF, TXT, MD)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Dict

try:
    import PyPDF2
except Exception:  # pragma: no cover - optional dependency fallback
    PyPDF2 = None

docx = None  # Removed to keep dependencies minimal
Presentation = None  # Removed to keep dependencies minimal


class UnsupportedFileTypeError(RuntimeError):
    """Raised when the requested file type cannot be processed."""


class MissingDependencyError(RuntimeError):
    """Raised when a loader dependency is not available."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MissingDependencyError(message)


def _load_pdf(file_bytes: bytes) -> str:
    _require(PyPDF2 is not None, "PyPDF2 is required for PDF ingestion.")
    text_parts = []
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        try:
            text_parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(text_parts)


def _load_docx(file_bytes: bytes) -> str:  # pragma: no cover - disabled
    raise MissingDependencyError("DOCX ingestion is disabled in this build.")


def _load_pptx(file_bytes: bytes) -> str:  # pragma: no cover - disabled
    raise MissingDependencyError("PPTX ingestion is disabled in this build.")


def _load_image(file_bytes: bytes) -> str:  # pragma: no cover - disabled
    raise MissingDependencyError("Image OCR ingestion is disabled in this build.")


def _load_txt(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except Exception:  # pragma: no cover - fallback path
            continue
    return file_bytes.decode("utf-8", errors="ignore")


def _load_md(file_bytes: bytes) -> str:
    return _load_txt(file_bytes)


def _load_csv(file_bytes: bytes) -> str:  # pragma: no cover - disabled
    raise MissingDependencyError("CSV/XLSX ingestion is disabled in this build.")


_LOADER_MAP: Dict[str, Callable[[bytes], str]] = {
    ".pdf": _load_pdf,
    ".txt": _load_txt,
    ".md": _load_md,
}


def extract_text_from_bytes(filename: str, file_bytes: bytes) -> str:
    """Return normalized text extracted from the uploaded file."""

    extension = Path(filename).suffix.lower()
    if extension not in _LOADER_MAP:
        raise UnsupportedFileTypeError(
            f"Unsupported file extension '{extension}'. "
            "Supported types: PDF, DOCX, PPTX, TXT, MD, CSV, PNG, JPG, JPEG."
        )

    loader = _LOADER_MAP[extension]
    text = loader(file_bytes)
    return text.strip()


SUPPORTED_EXTENSIONS = tuple(sorted(_LOADER_MAP.keys()))
