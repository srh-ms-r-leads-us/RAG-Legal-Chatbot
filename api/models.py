"""
models.py
---------
Pydantic request and response models for the RAG chatbot API.

Every endpoint's input and output is defined here.
FastAPI validates all incoming data automatically before the
handler function is called — invalid requests get a 422 response
with a clear description of what was wrong.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator

from config import cfg


# ---------------------------------------------------------------------------
# Shared / reusable models
# ---------------------------------------------------------------------------

class ChunkResult(BaseModel):
    """
    A single retrieved chunk returned in search results.
    Your team passes this directly into the LLM prompt as context.
    """
    text:         str   = Field(description="Raw chunk text — pass this to the LLM.")
    source:       str   = Field(description="Source PDF filename.")
    doc_name:     str   = Field(description="Document name without .pdf extension.")
    page_num:     int   = Field(description="Page number within the source PDF (1-indexed).")
    chunk_index:  int   = Field(description="Chunk position within the page (0-indexed).")
    similarity:   float = Field(description="Vector similarity score (0–1, higher = better).")
    bm25_score:   float = Field(description="BM25 keyword match score.")
    rrf_score:    float = Field(description="Reciprocal Rank Fusion combined score.")
    rerank_score: float = Field(description="Cross-encoder rerank score (final ranking signal).")
    citation:     str   = Field(description="Ready-made citation string e.g. 'doc.pdf — page 3'.")


# ---------------------------------------------------------------------------
# /search
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Request body for POST /api/v1/search"""

    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language question.",
        examples=["What policies help retain older workers?"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=cfg.API_MAX_TOP_K,
        description=f"Results to return. Default {cfg.RETRIEVAL_TOP_K}, max {cfg.API_MAX_TOP_K}.",
    )
    min_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=f"Minimum similarity threshold. Default {cfg.RETRIEVAL_MIN_SCORE}.",
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query must not be blank.")
        return v.strip()


class SearchResponse(BaseModel):
    """Response body for POST /api/v1/search"""

    query:         str             = Field(description="The original query string.")
    total_results: int             = Field(description="Number of chunks returned.")
    results:       list[ChunkResult] = Field(description="Ranked list of retrieved chunks.")
    pipeline:      dict            = Field(description="Active pipeline flags (hybrid, reranking).")


# ---------------------------------------------------------------------------
# /context
# ---------------------------------------------------------------------------

class ContextResponse(BaseModel):
    """Response body for POST /api/v1/context"""

    query:         str        = Field(description="The original query string.")
    total_results: int        = Field(description="Number of chunks in the context block.")
    context:       str        = Field(description="Formatted context string — inject directly into LLM prompt.")
    citations:     list[str]  = Field(description="List of citation strings for all sources used.")
    confidences:   list[dict] = Field(description="Per-source confidence scores for UI display.")


# ---------------------------------------------------------------------------
# /documents
# ---------------------------------------------------------------------------

class DocumentInfo(BaseModel):
    """Metadata about one ingested document."""

    filename:     str = Field(description="PDF filename e.g. 'ECE-WG.1-42-PB28.pdf'.")
    doc_name:     str = Field(description="Document name without extension.")
    chunk_count:  int = Field(description="Number of chunks stored in ChromaDB for this document.")
    ingested_at:  str = Field(description="ISO-8601 timestamp of when this document was ingested.")
    sha256:       str = Field(description="SHA-256 hash of the PDF — changes if the file is updated.")


class DocumentsResponse(BaseModel):
    """Response body for GET /api/v1/documents"""

    total_documents: int               = Field(description="Total number of ingested documents.")
    total_chunks:    int               = Field(description="Total chunks across all documents.")
    documents:       list[DocumentInfo] = Field(description="Per-document metadata list.")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response body for GET /api/v1/health"""

    status:         str           = Field(description="'ok' when healthy.")
    api_version:    str           = Field(description="API version prefix.")
    collection:     str           = Field(description="ChromaDB collection name.")
    total_vectors:  int           = Field(description="Total vectors in the collection.")
    embed_model:    str           = Field(description="Embedding model name.")
    hybrid_search:  bool          = Field(description="Whether hybrid search is active.")
    reranking:      bool          = Field(description="Whether reranking is active.")
    reranker_model: Optional[str] = Field(description="Reranker model name (null if disabled).")


# ---------------------------------------------------------------------------
# /feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    """
    Request body for POST /api/v1/feedback.

    Your LLM team calls this after the user rates a response so
    we can log which queries and results were helpful vs not.
    """

    query:      str  = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The original query the user asked.",
    )
    helpful:    bool = Field(
        ...,
        description="True = thumbs up (helpful), False = thumbs down (not helpful).",
    )
    llm_answer: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="The LLM-generated answer shown to the user (optional but useful for analysis).",
    )
    citations:  Optional[list[str]] = Field(
        default=None,
        description="List of citation strings from the /search or /context response.",
    )
    comment:    Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional free-text comment from the user.",
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query must not be blank.")
        return v.strip()


class FeedbackResponse(BaseModel):
    """Response body for POST /api/v1/feedback"""

    received:   bool = Field(description="Always True when feedback was saved successfully.")
    message:    str  = Field(description="Confirmation message.")
    feedback_id: str = Field(description="Unique ID assigned to this feedback entry.")


# ---------------------------------------------------------------------------
# Error response (returned by all error handlers)
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard error envelope returned on all 4xx and 5xx responses."""

    error:   str           = Field(description="Short error code e.g. 'HTTP_422'.")
    message: str           = Field(description="Human-readable description.")
    detail:  Optional[str] = Field(default=None, description="Extra technical detail.")