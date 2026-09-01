"""
main.py
-------
FastAPI application for the RAG chatbot — retrieval layer.

Endpoints
---------
GET  /api/v1/health      — liveness check + collection stats
GET  /api/v1/documents   — list all ingested documents with metadata
POST /api/v1/search      — semantic search, returns ranked chunks as JSON
POST /api/v1/context     — semantic search, returns LLM-ready context string
POST /api/v1/feedback    — save thumbs-up / thumbs-down rating

Integration guide for the LLM team
------------------------------------
1. Call POST /api/v1/search  with {"query": "..."}
   → receive ranked chunks with text + citations

2. Build your LLM prompt using the chunks:
      context = "\\n\\n---\\n\\n".join(r["text"] for r in response["results"])
      prompt  = f"Answer using only this context:\\n{context}\\n\\nQuestion: {query}"

   OR call POST /api/v1/context to get the pre-formatted context string directly.

3. After the user rates the answer, call POST /api/v1/feedback
   → pass the query, helpful=true/false, llm_answer, and citations

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
    or:
    python api/main.py
"""

import json
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add ingestion/ and retrieval/ to path so imports resolve from any cwd
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "ingestion"))
sys.path.insert(0, str(_ROOT / "retrieval"))

from config import cfg
from feedback_store import get_feedback_summary, save_feedback
from logger import log
from models import (
    ChunkResult,
    ContextResponse,
    DocumentInfo,
    DocumentsResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from retrieval_engine import RetrieverClient

# Manifest file path — written by ingestion_manager.py
MANIFEST_FILE = cfg.OUTPUT_DIR / "manifest.json"


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    """Holds expensive shared objects — initialised once at startup."""
    retriever: Optional[RetrieverClient] = None


app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    log.info("=" * 60)
    log.info("RAG Chatbot API starting up")
    log.info("  Host          : %s", cfg.API_HOST)
    log.info("  Port          : %d", cfg.API_PORT)
    log.info("  Prefix        : %s", cfg.API_PREFIX)
    log.info("  Hybrid search : %s", cfg.HYBRID_SEARCH_ENABLED)
    log.info("  Reranking     : %s", cfg.RERANKER_ENABLED)
    log.info("=" * 60)

    try:
        app_state.retriever = RetrieverClient()
        log.info("RetrieverClient ready.")
    except Exception as exc:
        log.error("Startup failed: %s", exc)
        raise RuntimeError("Could not initialise retriever.") from exc

    yield

    log.info("RAG Chatbot API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Chatbot — Retrieval API",
    description="""
## RAG Chatbot Retrieval API

Built by the retrieval team. Use these endpoints to integrate
LLM-generated answers with semantically retrieved document context.

### Quick integration guide

```python
import requests

# 1. Get ranked chunks
resp = requests.post(
    "http://localhost:8080/api/v1/search",
    json={"query": "What policies help older workers?"}
)
chunks = resp.json()["results"]

# 2. Build your LLM prompt
context = "\\n\\n---\\n\\n".join(c["text"] for c in chunks)
citations = [c["citation"] for c in chunks]

# 3. Call your LLM with the context (your team's responsibility)
answer = your_llm(context, query)

# 4. Save user feedback
requests.post(
    "http://localhost:8080/api/v1/feedback",
    json={
        "query": query,
        "helpful": True,
        "llm_answer": answer,
        "citations": citations
    }
)
```
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware — CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware — Request logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log every request with method, path, status code, and duration."""
    request_id = str(uuid.uuid4())[:8]
    start      = time.perf_counter()

    log.info("[%s] → %s %s", request_id, request.method, request.url.path)

    response  = await call_next(request)
    duration  = (time.perf_counter() - start) * 1000

    log.info(
        "[%s] ← %s %s  %d  %.1f ms",
        request_id, request.method, request.url.path,
        response.status_code, duration,
    )

    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    log.warning("HTTP %d — %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred. Please try again.",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _pipeline_info() -> dict:
    return {
        "hybrid_search" : cfg.HYBRID_SEARCH_ENABLED,
        "reranking"     : cfg.RERANKER_ENABLED,
        "embed_model"   : cfg.EMBED_MODEL,
        "reranker_model": cfg.RERANKER_MODEL if cfg.RERANKER_ENABLED else None,
    }


def _require_retriever():
    """Raise 503 if the retriever is not ready."""
    if app_state.retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retriever not initialised. Check server logs.",
        )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get(
    f"{cfg.API_PREFIX}/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
    responses={503: {"model": ErrorResponse}},
)
async def health_check():
    """
    Returns API status and vector store statistics.

    Call this endpoint to confirm the API is running and the
    retrieval pipeline loaded successfully before sending queries.
    """
    _require_retriever()
    stats = app_state.retriever.get_collection_stats()

    return HealthResponse(
        status         = "ok",
        api_version    = cfg.API_PREFIX,
        collection     = stats["collection"],
        total_vectors  = stats["total_vectors"],
        embed_model    = stats["embed_model"],
        hybrid_search  = stats["hybrid_search"],
        reranking      = stats["reranking"],
        reranker_model = stats.get("reranker_model"),
    )


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------

@app.get(
    f"{cfg.API_PREFIX}/documents",
    response_model=DocumentsResponse,
    summary="List ingested documents",
    tags=["Documents"],
    responses={404: {"model": ErrorResponse}},
)
async def list_documents():
    """
    Returns metadata for every document currently in the vector store.

    Use this to show the user which documents the chatbot can answer
    questions about, or to verify that a newly added PDF was ingested.
    """
    if not MANIFEST_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Manifest file not found. "
                "Run ingestion_manager.py to ingest documents first."
            ),
        )

    with open(MANIFEST_FILE, encoding="utf-8") as fh:
        manifest: dict = json.load(fh)

    documents = []
    total_chunks = 0

    for filename, info in manifest.items():
        chunk_count = info.get("chunk_count", 0)
        total_chunks += chunk_count

        documents.append(DocumentInfo(
            filename    = filename,
            doc_name    = Path(filename).stem,
            chunk_count = chunk_count,
            ingested_at = info.get("ingested_at", "unknown"),
            sha256      = info.get("sha256", "unknown"),
        ))

    # Sort alphabetically by filename for consistent ordering
    documents.sort(key=lambda d: d.filename)

    return DocumentsResponse(
        total_documents = len(documents),
        total_chunks    = total_chunks,
        documents       = documents,
    )


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------

@app.post(
    f"{cfg.API_PREFIX}/search",
    response_model=SearchResponse,
    summary="Semantic search — returns ranked chunks",
    tags=["Retrieval"],
    responses={
        422: {"model": ErrorResponse, "description": "Validation error — check request body."},
        503: {"model": ErrorResponse, "description": "Retriever not ready."},
    },
)
async def search(request: SearchRequest):
    """
    Search the document collection and return ranked chunks as JSON.

    **Pipeline:** Vector search → BM25 keyword search → RRF fusion → Cross-encoder reranking

    Each result includes:
    - `text` — the chunk text to pass to your LLM
    - `citation` — source document and page number
    - `rerank_score` — final relevance score (higher = more relevant)

    **Minimal integration example:**
    ```python
    import requests
    resp = requests.post(
        "http://localhost:8080/api/v1/search",
        json={"query": "What policies help older workers?", "top_k": 5}
    )
    results = resp.json()["results"]
    context = "\\n\\n".join(r["text"] for r in results)
    ```
    """
    _require_retriever()

    log.info("Search: '%s'  top_k=%s  min_score=%s",
             request.query, request.top_k, request.min_score)

    try:
        chunks = app_state.retriever.search(
            query     = request.query,
            top_k     = request.top_k,
            min_score = request.min_score,
        )
    except Exception as exc:
        log.error("Search error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed. Please try again.",
        ) from exc

    results = [ChunkResult(**chunk.to_dict()) for chunk in chunks]

    log.info("Search returned %d results for: '%s'", len(results), request.query)

    return SearchResponse(
        query         = request.query,
        total_results = len(results),
        results       = results,
        pipeline      = _pipeline_info(),
    )


# ---------------------------------------------------------------------------
# POST /context
# ---------------------------------------------------------------------------

@app.post(
    f"{cfg.API_PREFIX}/context",
    response_model=ContextResponse,
    summary="Retrieve formatted context string for LLM",
    tags=["Retrieval"],
    responses={
        422: {"model": ErrorResponse, "description": "Validation error."},
        503: {"model": ErrorResponse, "description": "Retriever not ready."},
    },
)
async def get_context(request: SearchRequest):
    """
    Search the document collection and return a pre-formatted context block.

    The `context` field is ready to inject directly into your LLM prompt —
    no additional formatting needed.

    **Minimal integration example:**
    ```python
    import requests
    resp = requests.post(
        "http://localhost:8080/api/v1/context",
        json={"query": "What policies help older workers?"}
    )
    data     = resp.json()
    context  = data["context"]    # inject this into your LLM prompt
    citations = data["citations"]  # show these to the user

    prompt = f\"\"\"
    Answer the question using ONLY the context below.
    Cite the source document and page for every claim.

    CONTEXT:
    {context}

    QUESTION: {data['query']}

    ANSWER:
    \"\"\"
    ```
    """
    _require_retriever()

    log.info("Context: '%s'  top_k=%s", request.query, request.top_k)

    try:
        chunks  = app_state.retriever.search(
            query     = request.query,
            top_k     = request.top_k,
            min_score = request.min_score,
        )
        context     = app_state.retriever.format_context(chunks)
        confidences = app_state.retriever.get_chunk_confidences(chunks)
    except Exception as exc:
        log.error("Context error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Context retrieval failed. Please try again.",
        ) from exc

    citations = [chunk.citation for chunk in chunks]

    log.info("Context built from %d chunks for: '%s'", len(chunks), request.query)

    return ContextResponse(
        query         = request.query,
        total_results = len(chunks),
        context       = context,
        citations     = citations,
        confidences   = confidences,
    )


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------

@app.post(
    f"{cfg.API_PREFIX}/feedback",
    response_model=FeedbackResponse,
    summary="Save user feedback (thumbs up / down)",
    tags=["Feedback"],
    responses={422: {"model": ErrorResponse, "description": "Validation error."}},
)
async def submit_feedback(request: FeedbackRequest):
    """
    Save a thumbs-up or thumbs-down rating for a query/answer pair.

    Call this after the user rates the LLM's answer so the team can
    analyse which queries perform well and which need improvement.

    **Minimal integration example:**
    ```python
    import requests
    requests.post(
        "http://localhost:8080/api/v1/feedback",
        json={
            "query"      : "What policies help older workers?",
            "helpful"    : True,
            "llm_answer" : "According to the UNECE report...",
            "citations"  : ["PB_30_EN_ECE_WG.1_45.pdf — page 8"],
            "comment"    : "Very clear answer"
        }
    )
    ```

    Feedback is saved to `data/processed/feedback.jsonl`
    (one JSON object per line — easy to load with pandas).

    **Load all feedback for analysis:**
    ```python
    import pandas as pd
    df = pd.read_json("data/processed/feedback.jsonl", lines=True)
    print(df.groupby("helpful").size())
    ```
    """
    try:
        feedback_id = save_feedback(
            query      = request.query,
            helpful    = request.helpful,
            llm_answer = request.llm_answer,
            citations  = request.citations,
            comment    = request.comment,
        )
    except Exception as exc:
        log.error("Feedback save error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save feedback. Please try again.",
        ) from exc

    return FeedbackResponse(
        received    = True,
        message     = "Feedback saved. Thank you!",
        feedback_id = feedback_id,
    )


# ---------------------------------------------------------------------------
# Run directly:  python api/main.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host    = cfg.API_HOST,
        port    = cfg.API_PORT,
        reload  = False,   # set True for auto-reload during development
        workers = 1,       # single worker — ML models are not process-safe
    )