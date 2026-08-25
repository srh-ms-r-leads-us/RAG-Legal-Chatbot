"""
debug_search.py
---------------
Check what similarity scores the vehicle PDF chunks are getting.
Run: python debug_search.py
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

print(f"Total vectors in ChromaDB: {col.count()}")
print("-" * 60)

# Test query
query = "what policies help retain older workers?"
print(f"Query: {query}")
print("-" * 60)

results = col.query(
    query_texts=[query],
    n_results=10,
    include=["documents", "metadatas", "distances"],
)

for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
):
    sim = round(1 - dist, 4)
    source = meta.get("source_file", "unknown")
    page   = meta.get("page_num", "?")
    print(f"Similarity: {sim} | {source} | page {page}")
    print(f"Preview: {doc[:150]}")
    print()