import json
import logging
import pickle
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi          # pip install rank-bm25

# Config lives one level up in ingestion/
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
# Tokenisation
# ---------------------------------------------------------------------------

def tokenise(text: str) -> list[str]:
    """
    Convert a chunk of text into a list of lowercase tokens for BM25.

    Steps:
      1. Lowercase everything for case-insensitive matching
      2. Keep only letters, digits, and spaces (strip punctuation)
      3. Split on whitespace
      4. Remove single-character tokens (articles, stray letters)
    """
    text   = text.lower()
    text   = re.sub(r"[^a-z0-9\s]", " ", text)   # strip punctuation
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1]    # drop single chars
    return tokens


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_and_save() -> None:
    """
    Load chunks, tokenise them, build a BM25Okapi index, and save
    both the index and the chunk registry to disk.

    The chunk registry is a parallel list of metadata dicts stored
    alongside the index so the retrieval engine can look up the source
    and page number for any BM25 result by its list position.
    """
    log.info("=== BM25 Index Builder ===")
    log.info("Input  : %s", cfg.CHUNKS_FILE)
    log.info("Output : %s", cfg.BM25_INDEX_FILE)

    # Guard: require Stage 2 output to exist
    if not cfg.CHUNKS_FILE.exists():
        log.error("Chunks file not found: %s", cfg.CHUNKS_FILE)
        log.error("Run text_chunker.py first.")
        sys.exit(1)

    with open(cfg.CHUNKS_FILE, encoding="utf-8") as fh:
        chunks: list[dict] = json.load(fh)

    log.info("Loaded %d chunks — tokenising ...", len(chunks))

    # Build two parallel structures:
    #   tokenised_corpus  — list of token lists fed into BM25
    #   chunk_registry    — list of metadata dicts for result lookup
    tokenised_corpus: list[list[str]] = []
    chunk_registry:   list[dict]      = []

    for chunk in chunks:
        tokens = tokenise(chunk["text"])
        tokenised_corpus.append(tokens)
        chunk_registry.append({
            "chunk_id"   : chunk["chunk_id"],
            "text"       : chunk["text"],
            "metadata"   : chunk["metadata"],
        })

    log.info("Tokenisation complete — building BM25Okapi index ...")

    # BM25Okapi is the standard Okapi BM25 variant
    # k1=1.5 controls term frequency saturation
    # b=0.75  controls document length normalisation
    bm25_index = BM25Okapi(tokenised_corpus, k1=1.5, b=0.75)

    # Persist index + registry together in a single pickle file
    payload = {
        "bm25_index"    : bm25_index,
        "chunk_registry": chunk_registry,
    }

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(cfg.BM25_INDEX_FILE, "wb") as fh:
        pickle.dump(payload, fh)

    log.info("─" * 50)
    log.info("BM25 index built over %d documents", len(chunk_registry))
    log.info("Saved to : %s", cfg.BM25_INDEX_FILE)
    log.info("─" * 50)
    log.info("Done. The retrieval engine will load this index at startup.")


if __name__ == "__main__":
    build_and_save()