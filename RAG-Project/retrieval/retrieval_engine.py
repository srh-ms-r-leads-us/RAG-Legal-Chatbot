"""
retrieval_engine.py
-------------------
Core retrieval module for the RAG chatbot.

Full pipeline (all steps configurable via .env):

    User Query
        │
        ├── Vector Search  (ChromaDB semantic similarity)   ──┐
        │                                                      ├── RRF Fusion
        └── BM25 Search    (keyword exact/near-exact match) ──┘       │
                                                                  Reranker
                                                             (cross-encoder)
                                                                       │
                                                               Top K Results

Each stage can be toggled independently in .env:
    HYBRID_SEARCH_ENABLED = true/false
    RERANKER_ENABLED      = true/false

Public API:
    retriever = RetrieverClient()
    results   = retriever.search("What policies help older workers?")
    context   = retriever.format_context(results)

Run (smoke test):
    python retrieval_engine.py
"""

import logging
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder    # pip install sentence-transformers

# Config lives one level up in ingestion/ — add to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingestion"))
from config import cfg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """
    A single search result returned by the retrieval engine.

    Attributes
    ----------
    text          : raw chunk text passed to the LLM as context
    source        : original PDF filename
    doc_name      : filename without extension (used for exclusion filtering)
    page_num      : 1-indexed page number within the source PDF
    chunk_index   : position of this chunk within its page (0-indexed)
    similarity    : cosine similarity from vector search (0–1, higher = better)
    bm25_score    : BM25 keyword score (only set when hybrid search is on)
    rrf_score     : Reciprocal Rank Fusion combined score
    rerank_score  : cross-encoder rerank score (only set when reranking is on)
    citation      : human-readable citation built from source + page
    """
    text:         str
    source:       str
    doc_name:     str
    page_num:     int
    chunk_index:  int
    similarity:   float
    bm25_score:   float = 0.0
    rrf_score:    float = 0.0
    rerank_score: float = 0.0
    citation:     str   = field(init=False)

    def __post_init__(self):
        self.citation = f"{self.source} — page {self.page_num}"

    def to_dict(self) -> dict:
        """Serialise to a plain dict (used by the FastAPI response model)."""
        return {
            "text"        : self.text,
            "source"      : self.source,
            "doc_name"    : self.doc_name,
            "page_num"    : self.page_num,
            "chunk_index" : self.chunk_index,
            "similarity"  : self.similarity,
            "bm25_score"  : self.bm25_score,
            "rrf_score"   : self.rrf_score,
            "rerank_score": self.rerank_score,
            "citation"    : self.citation,
        }


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """
    Lowercase, strip punctuation, split on whitespace.
    Must match the tokenisation used in bm25_index_builder.py exactly.
    """
    text   = text.lower()
    text   = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    vector_hits: list[tuple[str, float]],   # list of (chunk_id, similarity)
    bm25_hits:   list[tuple[str, float]],   # list of (chunk_id, bm25_score)
    rrf_k:       int   = cfg.HYBRID_RRF_K,
    vector_w:    float = cfg.HYBRID_VECTOR_WEIGHT,
) -> list[tuple[str, float]]:
    """
    Combine two ranked lists into one using Reciprocal Rank Fusion.

    RRF score for a document = Σ  weight / (k + rank)
    where rank is 1-indexed position in each list.

    Args:
        vector_hits : ranked results from ChromaDB  [(id, score), ...]
        bm25_hits   : ranked results from BM25      [(id, score), ...]
        rrf_k       : smoothing constant (default 60 from original paper)
        vector_w    : weight for vector results (BM25 gets 1 - vector_w)

    Returns:
        List of (chunk_id, rrf_score) sorted by descending rrf_score.
    """
    bm25_w = 1.0 - vector_w
    scores: dict[str, float] = {}

    # Accumulate RRF contribution from vector search results
    for rank, (chunk_id, _) in enumerate(vector_hits, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + vector_w / (rrf_k + rank)

    # Accumulate RRF contribution from BM25 results
    for rank, (chunk_id, _) in enumerate(bm25_hits, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + bm25_w / (rrf_k + rank)

    # Sort by combined RRF score descending
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Retriever client
# ---------------------------------------------------------------------------

class RetrieverClient:
    """
    Unified retrieval client supporting:
      • Vector-only search        (HYBRID_SEARCH_ENABLED=false)
      • Hybrid search             (HYBRID_SEARCH_ENABLED=true)
      • Reranking on top of either (RERANKER_ENABLED=true)

    Instantiate once at application startup — loading embedding and
    reranker models is expensive; we reuse them across all queries.
    """

    def __init__(
        self,
        top_k:     int   = cfg.RETRIEVAL_TOP_K,
        min_score: float = cfg.RETRIEVAL_MIN_SCORE,
    ):
        self.top_k     = top_k
        self.min_score = min_score
        self._exclude_pages: dict[str, list[int]] = cfg.RETRIEVAL_EXCLUDE_PAGES

        log.info("Initialising RetrieverClient ...")
        log.info("  Hybrid search  : %s", cfg.HYBRID_SEARCH_ENABLED)
        log.info("  Reranking      : %s", cfg.RERANKER_ENABLED)
        log.info("  Embed model    : %s", cfg.EMBED_MODEL)
        log.info("  Reranker model : %s", cfg.RERANKER_MODEL if cfg.RERANKER_ENABLED else "disabled")
        log.info("  Top K          : %d", self.top_k)
        log.info("  Min score      : %.2f", self.min_score)

        # Stage 1 — ChromaDB collection (always required)
        self._collection = self._open_chroma_collection()
        log.info("  Vectors in store : %d", self._collection.count())

        # Stage 2 — BM25 index (only when hybrid search is enabled)
        self._bm25_index    = None
        self._chunk_registry: list[dict] = []

        if cfg.HYBRID_SEARCH_ENABLED:
            self._bm25_index, self._chunk_registry = self._load_bm25_index()
            log.info("  BM25 documents   : %d", len(self._chunk_registry))

        # Stage 3 — Cross-encoder reranker (only when reranking is enabled)
        self._reranker = None

        if cfg.RERANKER_ENABLED:
            log.info("  Loading reranker model (first run may be slow) ...")
            self._reranker = CrossEncoder(cfg.RERANKER_MODEL)
            log.info("  Reranker ready.")

        log.info("RetrieverClient ready.")

    # ── Private: initialisation helpers ─────────────────────────────────────

    def _open_chroma_collection(self):
        """Open the persisted ChromaDB collection."""
        if not cfg.CHROMA_DIR.exists():
            raise RuntimeError(
                f"ChromaDB directory not found: {cfg.CHROMA_DIR}\n"
                "Run the ingestion pipeline first."
            )

        client = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))

        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=cfg.EMBED_MODEL
        )

        try:
            return client.get_collection(
                name=cfg.CHROMA_COLLECTION,
                embedding_function=embed_fn,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Collection '{cfg.CHROMA_COLLECTION}' not found.\n"
                "Run vector_store.py first."
            ) from exc

    def _load_bm25_index(self) -> tuple:
        """Load the persisted BM25 index and chunk registry from disk."""
        if not cfg.BM25_INDEX_FILE.exists():
            raise RuntimeError(
                f"BM25 index not found: {cfg.BM25_INDEX_FILE}\n"
                "Run bm25_index_builder.py first."
            )

        with open(cfg.BM25_INDEX_FILE, "rb") as fh:
            payload = pickle.load(fh)

        return payload["bm25_index"], payload["chunk_registry"]

    # ── Private: exclusion filter ────────────────────────────────────────────

    def _is_excluded(self, doc_name: str, page_num: int) -> bool:
        """Return True if this page is on the exclusion list."""
        return page_num in self._exclude_pages.get(doc_name, [])

    # ── Private: vector search ───────────────────────────────────────────────

    def _vector_search(self, query: str, fetch_k: int) -> list[tuple[str, float, dict, str]]:
        """
        Query ChromaDB for the top fetch_k semantic matches.
        Returns list of (chunk_id, similarity, metadata, text).
        """
        raw = self._collection.query(
            query_texts=[query],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            # Reconstruct chunk_id using the same format as ingestion_manager.py
            # Format: {doc_name}_p{page:03d}_c{chunk_index:02d}
            doc_name    = meta.get("doc_name",    "unknown")
            page_num    = int(meta.get("page_num",    0))
            chunk_index = int(meta.get("chunk_index", 0))
            chunk_id    = f"{doc_name}_p{page_num:03d}_c{chunk_index:02d}"

            similarity = round(1 - dist, 4)
            results.append((chunk_id, similarity, meta, doc))

        return results

    # ── Private: BM25 search ─────────────────────────────────────────────────

    def _bm25_search(self, query: str, fetch_k: int) -> list[tuple[str, float]]:
        """
        Score all chunks with BM25 and return the top fetch_k.

        Returns list of (chunk_id, bm25_score).
        """
        query_tokens = _tokenise(query)

        # BM25 scores every document in the corpus in one call
        scores = self._bm25_index.get_scores(query_tokens)

        # Pair each score with its chunk_id from the registry
        scored = [
            (self._chunk_registry[i]["chunk_id"], float(scores[i]))
            for i in range(len(scores))
        ]

        # Sort descending and return top fetch_k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:fetch_k]

    # ── Private: reranking ───────────────────────────────────────────────────

    def _rerank(
        self,
        query:      str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Score each candidate with the cross-encoder and re-sort.

        The cross-encoder reads the query and chunk text together —
        unlike the embedding model which encodes them independently —
        giving much more precise relevance scores.
        """
        if not candidates:
            return candidates

        # Build (query, chunk_text) pairs for the cross-encoder
        pairs = [(query, c.text) for c in candidates]

        # Cross-encoder returns raw logit scores (higher = more relevant)
        rerank_scores = self._reranker.predict(pairs)

        # Attach scores and sort descending
        for chunk, score in zip(candidates, rerank_scores):
            chunk.rerank_score = round(float(score), 4)

        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        return candidates

    # ── Private: build RetrievedChunk from raw data ──────────────────────────

    def _make_chunk(
        self,
        text:       str,
        metadata:   dict,
        similarity: float,
        bm25_score: float = 0.0,
        rrf_score:  float = 0.0,
    ) -> Optional["RetrievedChunk"]:
        """
        Build a RetrievedChunk applying only the exclusion filter.
        Score filtering happens AFTER reranking in the search() method.
        """
        doc_name = metadata.get("doc_name",    "unknown")
        page_num = int(metadata.get("page_num", 0))

        # Only drop explicitly excluded pages (e.g. bibliography pages)
        if self._is_excluded(doc_name, page_num):
            return None

        return RetrievedChunk(
            text        = text,
            source      = metadata.get("source_file", "unknown"),
            doc_name    = doc_name,
            page_num    = page_num,
            chunk_index = int(metadata.get("chunk_index", 0)),
            similarity  = similarity,
            bm25_score  = bm25_score,
            rrf_score   = rrf_score,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def search(
        self,
        query:     str,
        top_k:     Optional[int]   = None,
        min_score: Optional[float] = None,
    ) -> list[RetrievedChunk]:
        """
        Run the full retrieval pipeline for a user query.

        Pipeline:
            1. Vector search  (always)
            2. BM25 search    (if HYBRID_SEARCH_ENABLED)
            3. RRF fusion     (if HYBRID_SEARCH_ENABLED)
            4. Reranking      (if RERANKER_ENABLED)
            5. Score filter + exclusion filter
            6. Return top_k results

        Parameters
        ----------
        query     : natural language question from the user
        top_k     : override default number of results for this call
        min_score : override default minimum score for this call

        Returns
        -------
        List of RetrievedChunk sorted by best score (rerank > rrf > similarity).
        """
        if not query or not query.strip():
            log.warning("Empty query — returning no results.")
            return []

        effective_top_k     = top_k     if top_k     is not None else self.top_k
        effective_min_score = min_score if min_score is not None else self.min_score

        # How many candidates to fetch from each source before filtering
        fetch_k = max(cfg.RERANKER_CANDIDATES, cfg.HYBRID_FETCH_K)

        # ── Step 1: Vector search ─────────────────────────────────────────
        vector_results = self._vector_search(query, fetch_k)

        # Build a lookup dict: chunk_id → (similarity, metadata, text)
        vector_lookup: dict[str, tuple[float, dict, str]] = {
            chunk_id: (sim, meta, text)
            for chunk_id, sim, meta, text in vector_results
        }

        if not cfg.HYBRID_SEARCH_ENABLED:
            # ── Vector-only path ──────────────────────────────────────────
            candidates: list[RetrievedChunk] = []

            for chunk_id, similarity, meta, text in vector_results:
                chunk = self._make_chunk(text, meta, similarity)
                if chunk:
                    candidates.append(chunk)
                if len(candidates) == fetch_k:
                    break

        else:
            # ── Hybrid path: vector + BM25 → RRF fusion ──────────────────

            # Step 2: BM25 keyword search
            bm25_results = self._bm25_search(query, fetch_k)

            # Build BM25 lookup: chunk_id → (bm25_score, chunk_dict)
            bm25_lookup: dict[str, tuple[float, dict]] = {}
            for chunk_id, bm25_score in bm25_results:
                # Find the corresponding chunk dict in the registry
                for entry in self._chunk_registry:
                    if entry["chunk_id"] == chunk_id:
                        bm25_lookup[chunk_id] = (bm25_score, entry)
                        break

            # Step 3: RRF fusion
            fused = _reciprocal_rank_fusion(
                vector_hits=[(cid, sim) for cid, sim, _, _ in vector_results],
                bm25_hits=bm25_results,
            )

            candidates = []
            for chunk_id, rrf_score in fused:
                # Prefer vector metadata/text (more reliable source)
                if chunk_id in vector_lookup:
                    similarity, meta, text = vector_lookup[chunk_id]
                elif chunk_id in bm25_lookup:
                    bm25_score, entry = bm25_lookup[chunk_id]
                    similarity = 0.0   # not in vector results
                    meta       = entry["metadata"]
                    text       = entry["text"]
                else:
                    continue

                bm25_score = bm25_lookup[chunk_id][0] if chunk_id in bm25_lookup else 0.0

                chunk = self._make_chunk(
                    text, meta, similarity,
                    bm25_score=bm25_score,
                    rrf_score=round(rrf_score, 6),
                )
                if chunk:
                    candidates.append(chunk)

                if len(candidates) == cfg.RERANKER_CANDIDATES:
                    break

        # ── Step 4: Reranking ─────────────────────────────────────────────
        if cfg.RERANKER_ENABLED and self._reranker is not None:
            candidates = self._rerank(query, candidates)

        # ── Step 5: Apply min_score filter AFTER reranking ────────────────
        # Filter based on similarity score (most reliable cross-document metric)
        # This is done after reranking so all candidates get a fair chance
        filtered = [
            c for c in candidates
            if c.similarity >= effective_min_score or c.similarity == 0.0
            # similarity == 0.0 means BM25-only result — keep it and let
            # rerank score decide its fate
        ]

        # If reranking is on, also filter by a minimum rerank score
        # to remove genuinely irrelevant results
        if cfg.RERANKER_ENABLED:
            # -5.0 is a very permissive floor — removes only truly irrelevant chunks
            filtered = [c for c in filtered if c.rerank_score > -5.0]

        # ── Step 6: Deduplicate — keep only best chunk per page ───────────
        seen_pages:   set[str]          = set()
        deduplicated: list[RetrievedChunk] = []

        for chunk in filtered:
            page_key = f"{chunk.doc_name}::p{chunk.page_num}"
            if page_key not in seen_pages:
                seen_pages.add(page_key)
                deduplicated.append(chunk)

        # ── Step 7: Return top K ──────────────────────────────────────────
        return deduplicated[:effective_top_k]

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a single LLM-ready context block.

        Each chunk is prefixed with its citation and confidence level.
        URLs and footnote references are stripped from chunk text to
        prevent the LLM from citing them as sources.
        """
        if not chunks:
            return "No relevant context found in the document collection."

        def clean_chunk_text(text: str) -> str:
            """Remove URLs and footnote references from chunk text."""
            import re
            # Remove URLs (http/https)
            text = re.sub(r'https?://\S+', '', text)
            # Remove bare www. URLs
            text = re.sub(r'www\.\S+', '', text)
            # Remove footnote references like [1], [2], 1., 2. at line start
            text = re.sub(r'^\s*\d+\.\s+https?://\S+', '', text, flags=re.MULTILINE)
            # Collapse multiple spaces/newlines left by removal
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            return text.strip()

        sections = []
        for i, chunk in enumerate(chunks, start=1):
            if cfg.RERANKER_ENABLED:
                score      = chunk.rerank_score
                if score >= 6.0:
                    confidence = "HIGH"
                elif score >= 3.0:
                    confidence = "MEDIUM"
                else:
                    confidence = "LOW"
                score_label = f"rerank: {score:.3f} | confidence: {confidence}"
            elif cfg.HYBRID_SEARCH_ENABLED:
                score_label = f"rrf: {chunk.rrf_score:.6f}"
            else:
                score_label = f"similarity: {chunk.similarity:.4f}"

            header      = f"[Source {i} | {chunk.citation} | {score_label}]"
            clean_text  = clean_chunk_text(chunk.text)
            sections.append(f"{header}\n{clean_text}")

        return "\n\n---\n\n".join(sections)

    def get_chunk_confidences(self, chunks: list[RetrievedChunk]) -> list[dict]:
        """
        Return confidence metadata for each chunk.
        Used by the UI to render visual confidence bars.
        """
        results = []
        for chunk in chunks:
            if cfg.RERANKER_ENABLED:
                score = chunk.rerank_score
                # Normalise rerank logit to 0-1 range for UI display
                # Typical range is -5 to +10, we clamp to 0-10 then normalise
                normalised = max(0.0, min(1.0, (score + 5) / 15))
                if score >= 6.0:
                    label = "High"
                elif score >= 3.0:
                    label = "Medium"
                else:
                    label = "Low"
            else:
                normalised = chunk.similarity
                if normalised >= 0.7:
                    label = "High"
                elif normalised >= 0.5:
                    label = "Medium"
                else:
                    label = "Low"

            results.append({
                "citation"    : chunk.citation,
                "score"       : round(normalised, 3),
                "label"       : label,
                # All individual scores for UI score breakdown display
                "similarity"  : round(chunk.similarity,   4),
                "bm25_score"  : round(chunk.bm25_score,   4),
                "rrf_score"   : round(chunk.rrf_score,    6),
                "rerank_score": round(chunk.rerank_score, 4),
            })
        return results

    def get_collection_stats(self) -> dict:
        """Return stats about the vector store — used by API health endpoint."""
        return {
            "collection"          : cfg.CHROMA_COLLECTION,
            "total_vectors"       : self._collection.count(),
            "embed_model"         : cfg.EMBED_MODEL,
            "hybrid_search"       : cfg.HYBRID_SEARCH_ENABLED,
            "reranking"           : cfg.RERANKER_ENABLED,
            "reranker_model"      : cfg.RERANKER_MODEL if cfg.RERANKER_ENABLED else None,
            "bm25_documents"      : len(self._chunk_registry) if cfg.HYBRID_SEARCH_ENABLED else None,
            "chroma_dir"          : str(cfg.CHROMA_DIR),
        }


# ---------------------------------------------------------------------------
# Smoke test:  python retrieval_engine.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("RetrieverClient — smoke test")
    print(f"  Hybrid search : {cfg.HYBRID_SEARCH_ENABLED}")
    print(f"  Reranking     : {cfg.RERANKER_ENABLED}")
    print("=" * 65)

    retriever = RetrieverClient()

    stats = retriever.get_collection_stats()
    print(f"\nCollection stats:")
    for k, v in stats.items():
        print(f"  {k:<22}: {v}")

    TEST_QUERIES = [
        "What policies help retain older workers in the workforce?",
        "How does demographic change affect Europe?",
        "What causes loneliness among older persons?",
    ]

    for query in TEST_QUERIES:
        print("\n" + "─" * 65)
        print(f"QUERY : {query}")
        print("─" * 65)

        results = retriever.search(query)

        if not results:
            print("  No results above the minimum score threshold.")
            continue

        for rank, chunk in enumerate(results, start=1):
            rerank = f" | rerank: {chunk.rerank_score:.3f}" if cfg.RERANKER_ENABLED else ""
            rrf    = f" | rrf: {chunk.rrf_score:.5f}"       if cfg.HYBRID_SEARCH_ENABLED else ""
            print(
                f"\n  #{rank}  {chunk.citation}"
                f"  | sim: {chunk.similarity}{rrf}{rerank}"
            )
            print(f"       {chunk.text[:180].replace(chr(10), ' ')} ...")

        print(f"\n  --- Context block preview ---")
        context = retriever.format_context(results)
        print(f"  {context[:400].replace(chr(10), ' ')} ...")

    print("\n" + "=" * 65)
    print("Smoke test complete.")