"""
config.py
---------
Central configuration loader for the entire RAG chatbot project.

All modules — ingestion, retrieval, API, LLM, UI — import their
settings from this single file.  Values are read exclusively from
the .env file so there are no hard-coded constants anywhere.

Usage:
    from config import cfg
    print(cfg.OLLAMA_MODEL)
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Locate and load .env
# ---------------------------------------------------------------------------

_HERE     = Path(__file__).resolve().parent
_ENV_FILE = _HERE / ".env"

if not _ENV_FILE.exists():
    _ENV_FILE = _HERE.parent / ".env"

if not _ENV_FILE.exists():
    raise FileNotFoundError(
        "No .env file found. "
        "Copy .env.example to .env and fill in your values."
    )

load_dotenv(_ENV_FILE)

if os.getenv("HF_TOKEN") is not None:
    os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))


# ---------------------------------------------------------------------------
# Configuration class
# ---------------------------------------------------------------------------

class Config:
    """
    Typed configuration sourced entirely from environment variables.

    Sections
    --------
    Paths          — directories for PDFs, processed data, vector DB
    Chunking       — how pages are split into chunks
    Embedding      — model name and batch size
    ChromaDB       — collection name and distance metric
    Retrieval      — top-k, score threshold, page exclusions
    Hybrid Search  — vector + BM25 fusion settings
    Reranking      — cross-encoder model and candidate pool
    API Server     — host, port, prefix, CORS
    LLM (Ollama)   — model, temperature, streaming
    Streamlit UI   — API URL, page title, history limit
    Verification   — ingestion tester settings
    """

    # ── Paths ────────────────────────────────────────────────────────────────

    PDF_DIR:    Path = Path(os.getenv("PDF_DIR",    "data/raw"))
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "data/processed"))
    CHROMA_DIR: Path = Path(os.getenv("CHROMA_DIR", "data/chroma_db"))

    EXTRACTED_PAGES_FILE: Path = OUTPUT_DIR / "extracted_pages.json"
    CHUNKS_FILE:          Path = OUTPUT_DIR / "chunks.json"
    BM25_INDEX_FILE:      Path = OUTPUT_DIR / "bm25_index.pkl"

    # ── Chunking ─────────────────────────────────────────────────────────────

    CHUNK_SIZE:      int = int(os.getenv("CHUNK_SIZE",      400))
    CHUNK_OVERLAP:   int = int(os.getenv("CHUNK_OVERLAP",   80))
    MIN_CHUNK_WORDS: int = int(os.getenv("MIN_CHUNK_WORDS", 30))
    MIN_PAGE_CHARS:  int = int(os.getenv("MIN_PAGE_CHARS",  100))

    # Chunking strategy: static | sentence | structure
    # static    — fixed sliding window (original)
    # sentence  — always cuts at sentence boundaries
    # structure — detects headings/sections, chunks by document structure
    CHUNK_STRATEGY: str = os.getenv("CHUNK_STRATEGY", "structure")

    # Table extraction — requires pdfplumber (pip install pdfplumber)
    EXTRACT_TABLES: bool = os.getenv("EXTRACT_TABLES", "true").lower() == "true"

    # Image understanding — requires LLaVA in Ollama (ollama pull llava)
    EXTRACT_IMAGES: bool = os.getenv("EXTRACT_IMAGES", "false").lower() == "true"

    # ── Embedding ────────────────────────────────────────────────────────────

    EMBED_MODEL:      str = os.getenv("EMBED_MODEL",      "all-MiniLM-L6-v2")
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", 50))

    # ── ChromaDB ─────────────────────────────────────────────────────────────

    CHROMA_COLLECTION:      str = os.getenv("CHROMA_COLLECTION",      "unece_documents")
    CHROMA_DISTANCE_METRIC: str = os.getenv("CHROMA_DISTANCE_METRIC", "cosine")

    # ── Retrieval Engine ─────────────────────────────────────────────────────

    RETRIEVAL_TOP_K:         int   = int(os.getenv("RETRIEVAL_TOP_K",   5))
    RETRIEVAL_MIN_SCORE:     float = float(os.getenv("RETRIEVAL_MIN_SCORE", 0.45))
    RETRIEVAL_EXCLUDE_PAGES: dict  = json.loads(
        os.getenv("RETRIEVAL_EXCLUDE_PAGES", "{}")
    )

    # ── Hybrid Search ────────────────────────────────────────────────────────

    HYBRID_SEARCH_ENABLED: bool  = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
    HYBRID_FETCH_K:        int   = int(os.getenv("HYBRID_FETCH_K",       20))
    HYBRID_RRF_K:          int   = int(os.getenv("HYBRID_RRF_K",         60))
    HYBRID_VECTOR_WEIGHT:  float = float(os.getenv("HYBRID_VECTOR_WEIGHT", 0.7))

    # ── Reranking ────────────────────────────────────────────────────────────

    RERANKER_ENABLED:    bool = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
    RERANKER_MODEL:      str  = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_CANDIDATES: int  = int(os.getenv("RERANKER_CANDIDATES", 20))

    # ── API Server ───────────────────────────────────────────────────────────

    API_HOST:   str = os.getenv("API_HOST",   "0.0.0.0")
    API_PORT:   int = int(os.getenv("API_PORT", 8080))
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")
    API_MAX_TOP_K: int = int(os.getenv("API_MAX_TOP_K", 20))
    API_CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.getenv(
            "API_CORS_ORIGINS",
            "http://localhost:8501,http://127.0.0.1:8501"
        ).split(",")
        if o.strip()
    ]

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE:  str = os.getenv("LOG_FILE", "")

    # ── LLM — Ollama ─────────────────────────────────────────────────────────

    # Ollama server URL — default for local installation
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Model name — must match a model pulled with: ollama pull <model>
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")

    # Maximum tokens the LLM generates per response
    OLLAMA_MAX_TOKENS: int = int(os.getenv("OLLAMA_MAX_TOKENS", 1024))

    # Low temperature = more factual, less creative — ideal for document Q&A
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", 0.1))

    # Timeout in seconds — increase for slower machines or larger models
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", 120))

    # Stream tokens to UI as they generate — better user experience
    OLLAMA_STREAM: bool = os.getenv("OLLAMA_STREAM", "true").lower() == "true"

    # ── Streamlit UI ─────────────────────────────────────────────────────────

    # URL of the FastAPI backend — must match API_HOST and API_PORT
    UI_API_BASE_URL: str = os.getenv("UI_API_BASE_URL", "http://localhost:8080/api/v1")

    # Browser tab title
    UI_PAGE_TITLE: str = os.getenv("UI_PAGE_TITLE", "UNECE Policy Chatbot")

    # Max messages kept in session chat history
    UI_MAX_HISTORY: int = int(os.getenv("UI_MAX_HISTORY", 50))

    # ── Verification ─────────────────────────────────────────────────────────

    VERIFY_TOP_K: int = int(os.getenv("VERIFY_TOP_K", 3))


# Single shared instance
cfg = Config()


# ---------------------------------------------------------------------------
# Self-test:  python config.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Active configuration")
    print("─" * 50)
    print(f"  PDF_DIR                  : {cfg.PDF_DIR}")
    print(f"  OUTPUT_DIR               : {cfg.OUTPUT_DIR}")
    print(f"  CHROMA_DIR               : {cfg.CHROMA_DIR}")
    print("─" * 50)
    print(f"  EMBED_MODEL              : {cfg.EMBED_MODEL}")
    print(f"  CHROMA_COLLECTION        : {cfg.CHROMA_COLLECTION}")
    print("─" * 50)
    print(f"  RETRIEVAL_TOP_K          : {cfg.RETRIEVAL_TOP_K}")
    print(f"  RETRIEVAL_MIN_SCORE      : {cfg.RETRIEVAL_MIN_SCORE}")
    print("─" * 50)
    print(f"  HYBRID_SEARCH_ENABLED    : {cfg.HYBRID_SEARCH_ENABLED}")
    print(f"  RERANKER_ENABLED         : {cfg.RERANKER_ENABLED}")
    print(f"  RERANKER_MODEL           : {cfg.RERANKER_MODEL}")
    print("─" * 50)
    print(f"  API_HOST                 : {cfg.API_HOST}")
    print(f"  API_PORT                 : {cfg.API_PORT}")
    print(f"  API_PREFIX               : {cfg.API_PREFIX}")
    print(f"  API_MAX_TOP_K            : {cfg.API_MAX_TOP_K}")
    print("─" * 50)
    print(f"  OLLAMA_BASE_URL          : {cfg.OLLAMA_BASE_URL}")
    print(f"  OLLAMA_MODEL             : {cfg.OLLAMA_MODEL}")
    print(f"  OLLAMA_TEMPERATURE       : {cfg.OLLAMA_TEMPERATURE}")
    print(f"  OLLAMA_MAX_TOKENS        : {cfg.OLLAMA_MAX_TOKENS}")
    print(f"  OLLAMA_STREAM            : {cfg.OLLAMA_STREAM}")
    print("─" * 50)
    print(f"  UI_API_BASE_URL          : {cfg.UI_API_BASE_URL}")
    print(f"  UI_PAGE_TITLE            : {cfg.UI_PAGE_TITLE}")
    print(f"  UI_MAX_HISTORY           : {cfg.UI_MAX_HISTORY}")