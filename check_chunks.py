"""
check_chunks.py
---------------
Check how many chunks of each type are in ChromaDB.
Run: python check_chunks.py
"""

import sys
sys.path.insert(0, "ingestion")
sys.path.insert(0, "retrieval")

from config import cfg
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=cfg.EMBED_MODEL
)
col = client.get_collection(
    name=cfg.CHROMA_COLLECTION,
    embedding_function=embed_fn,
)

print(f"Total chunks in ChromaDB: {col.count()}")
print("-" * 50)

# Count each chunk type
for chunk_type in ["text", "table", "image"]:
    results = col.get(
        where={"chunk_type": chunk_type},
        include=["metadatas"],
    )
    count = len(results["ids"])
    print(f"{chunk_type:<10} chunks: {count}")

print("-" * 50)

# Show sample image chunks
print("\nSample image chunks:")
image_results = col.get(
    where={"chunk_type": "image"},
    limit=10,
    include=["metadatas", "documents"],
)

for i, (doc_id, meta, doc) in enumerate(zip(
    image_results["ids"],
    image_results["metadatas"],
    image_results["documents"],
), 1):
    print(f"\n  #{i}")
    print(f"  Source : {meta.get('source_file', '?')}")
    print(f"  Page   : {meta.get('page_num', '?')}")
    print(f"  Preview: {doc[:150]}")