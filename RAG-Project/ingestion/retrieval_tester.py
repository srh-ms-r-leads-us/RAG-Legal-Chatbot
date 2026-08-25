import logging
import sys

import chromadb
from chromadb.utils import embedding_functions

# Import the central configuration object (reads from .env)
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
# Test queries — aligned with the UNECE document corpus
# ---------------------------------------------------------------------------

# Add or remove queries here to match the topics in your documents.
# These run automatically when the script is executed.
TEST_QUERIES: list[str] = [
    "What policies help retain older workers in the workforce?",
    "How does demographic change affect labour markets in Europe?",
    "What are the main causes of loneliness among older persons?",
    "How can pension systems be reformed to reduce old age poverty?",
    "What is the demographic dividend and how does it apply to Central Asia?",
    "What strategies exist to combat ageism in the workplace?",
    "How can digital skills of older workers be improved?",
]


# ---------------------------------------------------------------------------
# ChromaDB connection
# ---------------------------------------------------------------------------

def open_collection():
    """
    Open the existing ChromaDB collection (read-only usage).
    Exits if the collection cannot be found.
    """
    client = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg.EMBED_MODEL
    )

    try:
        collection = client.get_collection(
            name=cfg.CHROMA_COLLECTION,
            embedding_function=embed_fn,
        )
    except Exception as exc:
        log.error("Could not open collection '%s': %s", cfg.CHROMA_COLLECTION, exc)
        log.error("Run vector_store.py first to build the vector store.")
        sys.exit(1)

    return collection


# ---------------------------------------------------------------------------
# Single-query retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, collection, top_k: int) -> list[dict]:
    """
    Embed *query* and return the *top_k* most similar chunks.

    ChromaDB returns cosine *distance* (lower = more similar).
    We convert it to similarity (higher = more similar) for readability:
        similarity = 1 - cosine_distance

    Returns:
        List of result dicts sorted by descending similarity.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text"      : doc,
            "source"    : meta.get("source_file", "unknown"),
            "page"      : meta.get("page_num",    "?"),
            "doc_name"  : meta.get("doc_name",    "unknown"),
            # Convert cosine distance → similarity score in [0, 1]
            "similarity": round(1 - dist, 4),
        })

    return hits   # already sorted by ChromaDB (best match first)


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_results(query: str, hits: list[dict]) -> None:
    """Pretty-print retrieval results for one query."""
    print()
    print("─" * 70)
    print(f"  QUERY : {query}")
    print("─" * 70)

    for rank, hit in enumerate(hits, start=1):
        # Truncate the preview to 250 characters and strip newlines
        preview = hit["text"][:250].replace("\n", " ").strip()

        print(f"\n  Result #{rank}")
        print(f"  Source     : {hit['source']}  (page {hit['page']})")
        print(f"  Similarity : {hit['similarity']}")
        print(f"  Preview    : {preview} ...")

    print()


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """Run all test queries and print ranked results."""
    log.info("=== Stage 4 — Retrieval Verification ===")
    log.info("ChromaDB path   : %s", cfg.CHROMA_DIR.resolve())
    log.info("Collection      : %s", cfg.CHROMA_COLLECTION)
    log.info("Embedding model : %s", cfg.EMBED_MODEL)
    log.info("Results per query (TOP_K) : %d", cfg.VERIFY_TOP_K)

    collection = open_collection()

    log.info("Collection size : %d vectors", collection.count())
    log.info("Running %d test queries ...", len(TEST_QUERIES))

    for query in TEST_QUERIES:
        hits = retrieve(query, collection, top_k=cfg.VERIFY_TOP_K)
        print_results(query, hits)

    print("=" * 70)
    log.info("Verification complete.")
    log.info(
        "If results look relevant, the ingestion pipeline is working correctly."
    )
    log.info("Next steps → build retrieval engine, FastAPI layer, Streamlit UI.")


if __name__ == "__main__":
    run()
