"""
ingestion_manager.py
--------------------
Incremental ingestion pipeline for the RAG chatbot.

Features
--------
  • Manifest-based deduplication — only processes new/changed PDFs
  • Dynamic chunking strategies — static | sentence | structure
  • Table extraction — pdfplumber converts tables to markdown
  • Image understanding — LLaVA vision model describes charts/figures
  • Automatic BM25 index rebuild after any change

Chunking strategies (set CHUNK_STRATEGY in .env):
  static    — fixed 400-word sliding window (original)
  sentence  — cuts at sentence boundaries (never mid-sentence)
  structure — detects headings/sections, chunks by document structure

Run:
    python ingestion/ingestion_manager.py
    python ingestion/ingestion_manager.py --force     # re-ingest all
    python ingestion/ingestion_manager.py --strategy structure  # override
"""

import argparse
import base64
import hashlib
import json
import logging
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import fitz                                         # PyMuPDF
import requests
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

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
# Manifest helpers
# ---------------------------------------------------------------------------

MANIFEST_FILE = cfg.OUTPUT_DIR / "manifest.json"


def load_manifest() -> dict:
    """Load manifest from disk. Returns empty dict if not found."""
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_manifest(manifest: dict) -> None:
    """Persist manifest to disk."""
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


def hash_file(path: Path) -> str:
    """SHA-256 hash of a file — detects any change."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """Normalise raw PDF text."""
    text = re.sub(r"-\n(\w)", r"\1", raw)    # fix hyphenated line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)    # collapse excessive newlines
    text = re.sub(r"[ \t]+", " ", text)       # collapse whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# CHUNKING STRATEGIES
# ---------------------------------------------------------------------------

# ── Strategy 1: Static sliding window (original) ────────────────────────────

def static_chunks(text: str) -> list[str]:
    """
    Fixed-size sliding window chunking.
    Always produces chunks of exactly CHUNK_SIZE words.
    Simple and fast but may cut sentences mid-way.
    """
    words  = text.split()
    step   = cfg.CHUNK_SIZE - cfg.CHUNK_OVERLAP
    chunks = []
    start  = 0

    while start < len(words):
        end   = start + cfg.CHUNK_SIZE
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += step

    return chunks


# ── Strategy 2: Sentence-aware chunking ─────────────────────────────────────

def sentence_aware_chunks(text: str) -> list[str]:
    """
    Chunks that always end at sentence boundaries.
    Never cuts a sentence in half — preserves complete thoughts.

    Algorithm:
      1. Split text into sentences using punctuation
      2. Accumulate sentences until chunk reaches target size
      3. At target size, save chunk and start new one with overlap
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text] if text.strip() else []

    chunks        = []
    current_sents = []
    current_words = 0

    for sentence in sentences:
        s_words = len(sentence.split())

        # If adding this sentence would exceed target, save current chunk
        if current_words + s_words > cfg.CHUNK_SIZE and current_sents:
            chunk_text = " ".join(current_sents)
            chunks.append(chunk_text)

            # Build overlap from end of current chunk
            overlap_text  = chunk_text.split()
            overlap_sents = " ".join(overlap_text[-cfg.CHUNK_OVERLAP:])
            current_sents = [overlap_sents] if overlap_sents else []
            current_words = len(overlap_sents.split()) if overlap_sents else 0

        current_sents.append(sentence)
        current_words += s_words

    # Don't forget the last chunk
    if current_sents:
        chunks.append(" ".join(current_sents))

    return [c for c in chunks if len(c.split()) >= cfg.MIN_CHUNK_WORDS]


# ── Strategy 3: Structure-aware chunking ────────────────────────────────────

# UNECE document section patterns — detected and used as chunk boundaries
_SECTION_PATTERNS = [
    r'\n[A-Z]\s*[-—]\s+[A-Z][^\n]{5,80}\n',     # "A - Ensuring adequate income..."
    r'\n\d+\.\d*\s+[A-Z][^\n]{5,80}\n',          # "7.2 Recommendations"
    r'\nI{1,3}V?\s*\.\s+[A-Z][^\n]{5,80}\n',     # "III. Recommendations"
    r'\n[A-Z]{2,}[A-Z\s]{3,50}\n',               # "POLICY CHALLENGE"
    r'\nBox\s+\d+[.\s]',                          # "Box 3."
    r'\nFigure\s+\d+[.\s]',                       # "Figure 1."
    r'\n#{1,3}\s+[A-Z]',                          # Markdown headings if any
]

_SECTION_RE = re.compile('|'.join(_SECTION_PATTERNS), re.MULTILINE)


def structure_aware_chunks(text: str) -> list[str]:
    """
    Chunk by document structure — headings and sections first.

    Algorithm:
      1. Detect section boundaries using UNECE document patterns
      2. Each section becomes a chunk if it fits within max size
      3. Sections too long are split using sentence-aware chunking
      4. Sections too short are merged with the next section

    This produces the highest quality chunks for structured policy
    documents because each chunk is about exactly ONE policy topic.
    """
    # Find all section boundaries
    boundaries = [0]
    for match in _SECTION_RE.finditer(text):
        boundaries.append(match.start())
    boundaries.append(len(text))

    # Extract sections between boundaries
    sections = []
    for i in range(len(boundaries) - 1):
        section = text[boundaries[i]:boundaries[i+1]].strip()
        if section:
            sections.append(section)

    if not sections:
        # No structure detected — fall back to sentence-aware
        log.debug("No structure detected — using sentence-aware chunking")
        return sentence_aware_chunks(text)

    chunks        = []
    pending       = ""   # accumulates short sections for merging

    for section in sections:
        section_words = len(section.split())
        combined      = (pending + " " + section).strip() if pending else section
        combined_words = len(combined.split())

        if section_words < cfg.MIN_CHUNK_WORDS:
            # Section too short — merge with next
            pending = combined
            continue

        if combined_words <= cfg.CHUNK_SIZE:
            # Fits in one chunk — keep whole section together
            pending = combined
        else:
            # Save any pending content
            if pending and len(pending.split()) >= cfg.MIN_CHUNK_WORDS:
                chunks.append(pending)

            if section_words <= cfg.CHUNK_SIZE:
                # This section fits — use as-is
                pending = section
            else:
                # Section too long — split at sentence boundaries
                sub_chunks = sentence_aware_chunks(section)
                chunks.extend(sub_chunks[:-1])
                pending = sub_chunks[-1] if sub_chunks else ""

    # Don't forget the last pending section
    if pending and len(pending.split()) >= cfg.MIN_CHUNK_WORDS:
        chunks.append(pending)

    return chunks if chunks else sentence_aware_chunks(text)


def get_chunker(strategy: str):
    """Return the chunking function for the given strategy name."""
    strategies = {
        "static"   : static_chunks,
        "sentence" : sentence_aware_chunks,
        "structure": structure_aware_chunks,
    }
    if strategy not in strategies:
        log.warning("Unknown strategy '%s' — using 'structure'", strategy)
        return structure_aware_chunks
    return strategies[strategy]


# ---------------------------------------------------------------------------
# TABLE EXTRACTION (pdfplumber)
# ---------------------------------------------------------------------------

def extract_tables_from_page(pdf_path: Path, page_num: int) -> list[str]:
    """
    Extract tables from a specific page using pdfplumber.
    Converts each table to markdown format so the LLM can read it.

    Returns list of markdown table strings.
    """
    try:
        import pdfplumber
    except ImportError:
        log.debug("pdfplumber not installed — skipping table extraction")
        return []

    tables_markdown = []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num - 1 >= len(pdf.pages):
                return []

            page   = pdf.pages[page_num - 1]
            tables = page.extract_tables()

            for table in tables:
                if not table or len(table) < 2:
                    continue

                # Convert to markdown table format
                md_rows = []
                header  = table[0]

                # Clean None values
                header = [str(cell).strip() if cell else "" for cell in header]
                md_rows.append("| " + " | ".join(header) + " |")
                md_rows.append("| " + " | ".join(["---"] * len(header)) + " |")

                for row in table[1:]:
                    row = [str(cell).strip() if cell else "" for cell in row]
                    # Skip empty rows
                    if any(cell for cell in row):
                        md_rows.append("| " + " | ".join(row) + " |")

                if len(md_rows) > 2:   # header + separator + at least 1 data row
                    tables_markdown.append("\n".join(md_rows))

    except Exception as e:
        log.debug("Table extraction failed on page %d: %s", page_num, e)

    return tables_markdown


# ---------------------------------------------------------------------------
# IMAGE / CHART UNDERSTANDING (LLaVA via Ollama)
# ---------------------------------------------------------------------------

def describe_image_with_llava(image_bytes: bytes, context: str = "") -> str:
    """
    Send an image to LLaVA (vision model) via Ollama and get a description.

    LLaVA can read charts, graphs, and figures and describe their content
    in plain text that can be embedded and searched.

    Args:
        image_bytes : raw image bytes extracted from PDF
        context     : surrounding text to help LLaVA understand context

    Returns:
        Text description of the image, or empty string if failed.
    """
    try:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = (
            "This image is from a UNECE policy document about ageing, "
            "demographics, and workforce policy. "
            "Describe what this image shows. "
            "If it is a chart or graph, extract all visible data values, "
            "axis labels, legend items, and key trends. "
            "If it is a table, extract all rows and columns. "
            "Be specific and include all numbers you can see."
        )

        if context:
            prompt += f"\n\nSurrounding document text for context: {context[:300]}"

        response = requests.post(
            f"{cfg.OLLAMA_BASE_URL}/api/generate",
            json={
                "model" : "llava",   # must be pulled: ollama pull llava
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=60,
        )

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            log.debug("LLaVA returned status %d", response.status_code)
            return ""

    except requests.exceptions.ConnectionError:
        log.debug("Ollama not available for image description")
        return ""
    except Exception as e:
        log.debug("Image description failed: %s", e)
        return ""


def check_llava_available() -> bool:
    """Check if LLaVA model is available in Ollama."""
    try:
        r = requests.get(f"{cfg.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return any("llava" in m.lower() for m in models)
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Full document extraction
# ---------------------------------------------------------------------------

def extract_chunks_from_pdf(
    pdf_path:        Path,
    strategy:        str  = None,
    extract_tables:  bool = True,
    extract_images:  bool = True,
) -> list[dict]:
    """
    Extract text, tables, and images from a PDF and return chunks.

    Each chunk has:
      chunk_id  : unique identifier
      text      : content (text, table markdown, or image description)
      metadata  : source, page, type (text/table/image)

    Args:
        pdf_path       : path to PDF file
        strategy       : chunking strategy (static/sentence/structure)
        extract_tables : whether to extract and convert tables
        extract_images : whether to describe images with LLaVA
    """
    strategy  = strategy or cfg.CHUNK_STRATEGY
    chunker   = get_chunker(strategy)
    doc       = fitz.open(str(pdf_path))
    all_chunks: list[dict] = []

    llava_available = extract_images and check_llava_available()
    if extract_images and not llava_available:
        log.info("  LLaVA not available — skipping image extraction")
        log.info("  To enable: ollama pull llava")

    for page_index in range(len(doc)):
        page_num = page_index + 1
        page     = doc[page_index]

        # ── Text extraction ───────────────────────────────────────────────
        raw_text = page.get_text("text")
        text     = clean_text(raw_text)

        if len(text) >= cfg.MIN_PAGE_CHARS:
            text_chunks = chunker(text)

            for idx, chunk_text in enumerate(text_chunks):
                if len(chunk_text.split()) >= cfg.MIN_CHUNK_WORDS:
                    all_chunks.append({
                        "chunk_id": f"{pdf_path.stem}_p{page_num:03d}_c{idx:02d}",
                        "text"    : chunk_text,
                        "metadata": {
                            "doc_name"    : pdf_path.stem,
                            "source_file" : pdf_path.name,
                            "page_num"    : page_num,
                            "total_pages" : len(doc),
                            "chunk_index" : idx,
                            "chunk_type"  : "text",
                            "strategy"    : strategy,
                        },
                    })

        # ── Table extraction ──────────────────────────────────────────────
        if extract_tables:
            tables = extract_tables_from_page(pdf_path, page_num)
            for t_idx, table_md in enumerate(tables):
                if table_md.strip():
                    all_chunks.append({
                        "chunk_id": f"{pdf_path.stem}_p{page_num:03d}_t{t_idx:02d}",
                        "text"    : f"[TABLE from page {page_num}]\n{table_md}",
                        "metadata": {
                            "doc_name"    : pdf_path.stem,
                            "source_file" : pdf_path.name,
                            "page_num"    : page_num,
                            "total_pages" : len(doc),
                            "chunk_index" : t_idx,
                            "chunk_type"  : "table",
                            "strategy"    : strategy,
                        },
                    })

        # ── Image extraction and description ──────────────────────────────
        if llava_available:
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image  = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_ext     = base_image["ext"]

                    # Skip tiny images — likely icons or decorations
                    if len(image_bytes) < 10000:
                        continue

                    log.info(
                        "  Describing image %d on page %d with LLaVA ...",
                        img_idx + 1, page_num
                    )
                    description = describe_image_with_llava(image_bytes, context=text[:300])

                    if description and len(description.split()) >= 20:
                        all_chunks.append({
                            "chunk_id": f"{pdf_path.stem}_p{page_num:03d}_i{img_idx:02d}",
                            "text"    : (
                                f"[IMAGE/CHART from {pdf_path.name} page {page_num}]\n"
                                f"{description}"
                            ),
                            "metadata": {
                                "doc_name"    : pdf_path.stem,
                                "source_file" : pdf_path.name,
                                "page_num"    : page_num,
                                "total_pages" : len(doc),
                                "chunk_index" : img_idx,
                                "chunk_type"  : "image",
                                "img_format"  : img_ext,
                                "strategy"    : strategy,
                            },
                        })

                except Exception as e:
                    log.debug("Image %d on page %d failed: %s", img_idx, page_num, e)

    doc.close()
    return all_chunks


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def get_collection():
    """Open or create the ChromaDB collection."""
    cfg.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client   = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg.EMBED_MODEL
    )
    return client.get_or_create_collection(
        name=cfg.CHROMA_COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": cfg.CHROMA_DISTANCE_METRIC},
    )


def delete_chunks(collection, chunk_ids: list[str]) -> None:
    if chunk_ids:
        collection.delete(ids=chunk_ids)
        log.info("  Deleted %d old chunks from ChromaDB", len(chunk_ids))


def upsert_chunks(collection, chunks: list[dict]) -> None:
    for start in range(0, len(chunks), cfg.EMBED_BATCH_SIZE):
        batch = chunks[start : start + cfg.EMBED_BATCH_SIZE]
        collection.upsert(
            ids       = [c["chunk_id"]  for c in batch],
            documents = [c["text"]      for c in batch],
            metadatas = [c["metadata"]  for c in batch],
        )


# ---------------------------------------------------------------------------
# BM25 index rebuild
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    text   = text.lower()
    text   = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def rebuild_bm25_index(collection) -> None:
    """Rebuild BM25 index from all current ChromaDB chunks."""
    log.info("Rebuilding BM25 index ...")
    total  = collection.count()
    if total == 0:
        return

    all_docs  = []
    all_metas = []
    all_ids   = []
    offset    = 0

    while offset < total:
        result = collection.get(
            limit=500, offset=offset,
            include=["documents", "metadatas"],
        )
        all_ids.extend(result["ids"])
        all_docs.extend(result["documents"])
        all_metas.extend(result["metadatas"])
        offset += 500

    tokenised_corpus = [_tokenise(doc) for doc in all_docs]
    chunk_registry   = [
        {"chunk_id": cid, "text": doc, "metadata": meta}
        for cid, doc, meta in zip(all_ids, all_docs, all_metas)
    ]

    bm25_index = BM25Okapi(tokenised_corpus, k1=1.5, b=0.75)
    payload    = {"bm25_index": bm25_index, "chunk_registry": chunk_registry}

    with open(cfg.BM25_INDEX_FILE, "wb") as fh:
        pickle.dump(payload, fh)

    log.info("BM25 index rebuilt — %d documents indexed", len(chunk_registry))


# ---------------------------------------------------------------------------
# Main incremental ingestion logic
# ---------------------------------------------------------------------------

def run_incremental(
    force:          bool = False,
    strategy:       str  = None,
    extract_tables: bool = True,
    extract_images: bool = True,
) -> None:
    """
    Scan PDF directory and ingest only new or changed files.

    Args:
        force          : re-ingest all PDFs regardless of manifest
        strategy       : chunking strategy override (uses .env default if None)
        extract_tables : whether to extract tables with pdfplumber
        extract_images : whether to describe images with LLaVA
    """
    strategy = strategy or cfg.CHUNK_STRATEGY

    log.info("=== Incremental Ingestion Manager ===")
    log.info("PDF directory  : %s", cfg.PDF_DIR.resolve())
    log.info("Force mode     : %s", force)
    log.info("Chunk strategy : %s", strategy)
    log.info("Table extract  : %s", extract_tables)
    log.info("Image extract  : %s", extract_images)

    pdf_files = sorted(cfg.PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        log.error("No PDFs found in '%s'", cfg.PDF_DIR)
        sys.exit(1)

    log.info("Found %d PDF file(s)", len(pdf_files))

    manifest   = {} if force else load_manifest()
    collection = get_collection()

    stats = {
        "skipped": 0, "new": 0, "updated": 0, "deleted": 0,
        "chunks_added": 0, "chunks_removed": 0,
        "tables_added": 0, "images_added": 0,
    }

    current_pdf_names = {pdf.name for pdf in pdf_files}

    # ── Detect deleted PDFs ───────────────────────────────────────────────
    for pdf_name in list(manifest.keys()):
        if pdf_name not in current_pdf_names:
            log.info("PDF removed: '%s' — deleting chunks ...", pdf_name)
            old_ids = manifest[pdf_name].get("chunk_ids", [])
            delete_chunks(collection, old_ids)
            stats["chunks_removed"] += len(old_ids)
            del manifest[pdf_name]
            stats["deleted"] += 1

    # ── Process each PDF ─────────────────────────────────────────────────
    for pdf_path in pdf_files:
        pdf_name     = pdf_path.name
        current_hash = hash_file(pdf_path)

        if not force and pdf_name in manifest:
            if manifest[pdf_name]["sha256"] == current_hash:
                log.info("%-55s → SKIPPED (unchanged)", pdf_name)
                stats["skipped"] += 1
                continue
            else:
                log.info("%-55s → UPDATED", pdf_name)
                old_ids = manifest[pdf_name].get("chunk_ids", [])
                delete_chunks(collection, old_ids)
                stats["chunks_removed"] += len(old_ids)
                stats["updated"] += 1
        else:
            log.info("%-55s → NEW", pdf_name)
            stats["new"] += 1

        # Extract chunks (text + tables + images)
        chunks = extract_chunks_from_pdf(
            pdf_path,
            strategy       = strategy,
            extract_tables = extract_tables,
            extract_images = extract_images,
        )

        # Count by type for reporting
        text_chunks  = [c for c in chunks if c["metadata"]["chunk_type"] == "text"]
        table_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "table"]
        image_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "image"]

        log.info(
            "  → %d text chunks | %d table chunks | %d image chunks",
            len(text_chunks), len(table_chunks), len(image_chunks)
        )

        upsert_chunks(collection, chunks)

        stats["chunks_added"] += len(chunks)
        stats["tables_added"] += len(table_chunks)
        stats["images_added"] += len(image_chunks)

        manifest[pdf_name] = {
            "sha256"      : current_hash,
            "ingested_at" : datetime.now(timezone.utc).isoformat(),
            "chunk_count" : len(chunks),
            "chunk_ids"   : [c["chunk_id"] for c in chunks],
            "strategy"    : strategy,
            "tables"      : len(table_chunks),
            "images"      : len(image_chunks),
        }

    # ── Save manifest ─────────────────────────────────────────────────────
    save_manifest(manifest)

    # ── Rebuild BM25 if anything changed ─────────────────────────────────
    if stats["new"] > 0 or stats["updated"] > 0 or stats["deleted"] > 0:
        rebuild_bm25_index(collection)
    else:
        log.info("No changes — BM25 index not rebuilt.")

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("─" * 55)
    log.info("Ingestion complete")
    log.info("  Strategy       : %s", strategy)
    log.info("  Skipped        : %d  (unchanged)", stats["skipped"])
    log.info("  New            : %d", stats["new"])
    log.info("  Updated        : %d", stats["updated"])
    log.info("  Deleted        : %d", stats["deleted"])
    log.info("  Text chunks    : %d", stats["chunks_added"] - stats["tables_added"] - stats["images_added"])
    log.info("  Table chunks   : %d", stats["tables_added"])
    log.info("  Image chunks   : %d", stats["images_added"])
    log.info("  Total chunks   : %d", stats["chunks_added"])
    log.info("  Total in DB    : %d", collection.count())
    log.info("  Manifest       : %s", MANIFEST_FILE)
    log.info("─" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Incremental RAG ingestion with dynamic chunking."
    )
    parser.add_argument("--force",    action="store_true",
                        help="Re-ingest all PDFs regardless of manifest.")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["static", "sentence", "structure"],
                        help="Chunking strategy (overrides .env setting).")
    parser.add_argument("--no-tables", action="store_true",
                        help="Skip table extraction.")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip image description with LLaVA.")
    args = parser.parse_args()

    run_incremental(
        force          = args.force,
        strategy       = args.strategy,
        extract_tables = not args.no_tables,
        extract_images = not args.no_images,
    )