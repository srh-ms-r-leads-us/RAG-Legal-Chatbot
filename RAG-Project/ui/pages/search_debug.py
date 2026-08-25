"""
2_Search_Debug.py
-----------------
Search Debug page for the UNECE Policy Chatbot.

Purpose:
    Lets the professor (or developer) test any query and see
    exactly how the retrieval pipeline works — all scores visible,
    no LLM needed.

Shows for every result:
    - Vector similarity score  (semantic meaning match)
    - BM25 keyword score       (exact keyword match)
    - RRF fusion score         (hybrid combined score)
    - Rerank score             (cross-encoder final score)
    - Chunk type               (text / table / image)
    - Actual chunk text        (what gets sent to the LLM)

Access:
    Automatically appears as page 3 in the Streamlit sidebar
    when placed in ui/pages/
"""

import sys
from pathlib import Path

import requests
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "ingestion"))

from config import cfg

st.set_page_config(
    page_title="Search Debug — UNECE Chatbot",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Search Debug Panel")
st.caption(
    "Test retrieval queries and inspect all scores live. "
    "No LLM involved — pure retrieval pipeline output."
)

API_BASE = cfg.UI_API_BASE_URL

# ---------------------------------------------------------------------------
# Connection check
# ---------------------------------------------------------------------------

try:
    health = requests.get(f"{API_BASE}/health", timeout=5).json()
    st.success(f"✅ API online — {health['total_vectors']} vectors in collection")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Vectors",  health["total_vectors"])
    c2.metric("Hybrid Search",  "ON" if health["hybrid_search"] else "OFF")
    c3.metric("Reranking",      "ON" if health["reranking"]     else "OFF")
    c4.metric("Embed Model",    health["embed_model"].split("/")[-1])
except Exception:
    st.error("❌ API offline — run: python api/main.py")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

st.subheader("🔎 Run a Test Query")

col_q, col_k, col_s = st.columns([4, 1, 1])

with col_q:
    query = st.text_input(
        "Query",
        placeholder="What policies help retain older workers?",
        label_visibility="collapsed",
    )

with col_k:
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)

with col_s:
    min_score = st.number_input(
        "Min Score", min_value=0.0, max_value=1.0,
        value=cfg.RETRIEVAL_MIN_SCORE, step=0.05
    )

# Pre-built example queries for quick testing
st.caption("Quick test queries:")
example_queries = [
    "What policies help retain older workers?",
    "How does demographic change affect Europe?",
    "What causes loneliness among older persons?",
    "How can pension systems be reformed?",
    "What are the recommendations on road safety?",
]

cols = st.columns(len(example_queries))
for i, eq in enumerate(example_queries):
    if cols[i].button(eq[:35] + "...", key=f"eq_{i}", use_container_width=True):
        query = eq

# ---------------------------------------------------------------------------
# Run search
# ---------------------------------------------------------------------------

if query and query.strip():
    with st.spinner(f"Searching for: '{query}'..."):
        try:
            resp = requests.post(
                f"{API_BASE}/search",
                json={
                    "query"    : query,
                    "top_k"   : top_k,
                    "min_score": min_score,
                },
                timeout=30,
            )
            data = resp.json()
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    if resp.status_code != 200:
        st.error(f"API error {resp.status_code}: {data}")
        st.stop()

    results = data.get("results", [])
    pipeline = data.get("pipeline", {})

    # ── Summary ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Results for: *{query}*")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Results returned", len(results))
    m2.metric("Hybrid search",    "ON" if pipeline.get("hybrid_search") else "OFF")
    m3.metric("Reranking",        "ON" if pipeline.get("reranking")     else "OFF")
    m4.metric("Reranker model",   (pipeline.get("reranker_model") or "N/A").split("/")[-1])

    if not results:
        st.warning(
            "No results found above the minimum score threshold. "
            "Try lowering Min Score or rephrasing the query."
        )
        st.stop()

    # ── Score comparison table ─────────────────────────────────────────────
    st.subheader("📊 Score Comparison Table")
    st.caption(
        "All four scores shown for every result. "
        "Final ranking is determined by Rerank Score (rightmost column)."
    )

    table_rows = []
    for i, r in enumerate(results, 1):
        chunk_type = "📄" if r.get("chunk_type") == "text" else (
                     "📊" if r.get("chunk_type") == "table" else
                     "🖼️" if r.get("chunk_type") == "image" else "📄"
        )
        table_rows.append({
            "Rank"            : f"#{i}",
            "Source"          : r["source"],
            "Page"            : r["page_num"],
            "Type"            : chunk_type,
            "🔵 Vector Sim"   : round(r["similarity"],   4),
            "🟡 BM25"         : round(r["bm25_score"],   4),
            "🟣 RRF"          : round(r["rrf_score"],    6),
            "🟢 Rerank"       : round(r["rerank_score"], 4),
        })

    st.dataframe(
        table_rows,
        use_container_width=True,
        hide_index=True,
    )

    # ── Visual score bars per result ─────────────────────────────────────
    st.subheader("📈 Detailed Score Breakdown")

    for i, result in enumerate(results, 1):
        similarity   = result.get("similarity",   0.0)
        bm25_score   = result.get("bm25_score",   0.0)
        rrf_score    = result.get("rrf_score",    0.0)
        rerank_score = result.get("rerank_score", 0.0)
        citation     = result.get("citation",     "unknown")
        chunk_text   = result.get("text",         "")
        chunk_type   = result.get("chunk_type",   "text")

        # Determine confidence label
        if rerank_score >= 6.0:
            label      = "High"
            label_icon = "🟢"
        elif rerank_score >= 3.0:
            label      = "Medium"
            label_icon = "🟡"
        else:
            label      = "Low"
            label_icon = "🔴"

        with st.expander(
            f"#{i}  {citation}  |  {label_icon} {label} confidence  |  rerank: {rerank_score:.4f}",
            expanded=(i == 1),   # expand first result by default
        ):
            # Score bars
            c1, c2 = st.columns(2)

            with c1:
                st.caption("🔵 Vector Similarity (semantic meaning)")
                st.progress(float(min(max(similarity, 0.0), 1.0)))
                st.caption(
                    f"`{similarity:.4f}` — "
                    f"{'High' if similarity >= 0.7 else 'Medium' if similarity >= 0.5 else 'Low'} "
                    f"semantic match"
                )

                st.caption("🟣 RRF Score (hybrid fusion)")
                rrf_norm = float(min(rrf_score / 0.02, 1.0)) if rrf_score > 0 else 0.0
                st.progress(rrf_norm)
                st.caption(f"`{rrf_score:.6f}` — combined rank from vector + BM25")

            with c2:
                st.caption("🟡 BM25 Score (keyword match)")
                bm25_norm = float(min(bm25_score / 50.0, 1.0)) if bm25_score > 0 else 0.0
                st.progress(bm25_norm)
                st.caption(
                    f"`{bm25_score:.4f}` — "
                    f"{'Strong' if bm25_score > 10 else 'Moderate' if bm25_score > 3 else 'Weak'} "
                    f"keyword match"
                )

                st.caption("🟢 Rerank Score (cross-encoder)")
                rerank_norm = float(min(max((rerank_score + 5) / 15, 0.0), 1.0))
                st.progress(rerank_norm)
                st.caption(f"`{rerank_score:.4f}` — final relevance score (determines ranking)")

            st.divider()

            # Chunk metadata
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Page",       result.get("page_num",    "?"))
            mc2.metric("Chunk Type", chunk_type.upper())
            mc3.metric("Chunk Index", result.get("chunk_index", "?"))

            # Actual chunk text
            st.caption("📝 Chunk text sent to LLM:")
            st.text_area(
                label       = f"chunk_text_{i}",
                value       = chunk_text,
                height      = 200,
                disabled    = True,
                label_visibility = "collapsed",
            )

    # ── Hybrid search explanation ──────────────────────────────────────────
    st.divider()
    with st.expander("ℹ️ How the scores are calculated", expanded=False):
        st.markdown("""
### Score Explanation

| Score | What it measures | Range |
|-------|-----------------|-------|
| 🔵 **Vector Similarity** | Semantic meaning match between query and chunk | 0.0 – 1.0 |
| 🟡 **BM25 Score** | Exact/near-exact keyword match frequency | 0.0 – 50+ |
| 🟣 **RRF Score** | Combined rank from both vector and BM25 searches | 0.0 – 0.02 |
| 🟢 **Rerank Score** | Cross-encoder reads query + chunk together for precise relevance | -5 to +10 |

### How ranking works
```
Step 1: Vector search  → top 20 semantic matches
Step 2: BM25 search    → top 20 keyword matches  
Step 3: RRF fusion     → merge both into top 20 (0.7 × vector + 0.3 × BM25)
Step 4: Cross-encoder  → rerank top 20 → final top 5
Step 5: Deduplication  → one chunk per page
```

### RRF Formula
```
RRF score = 0.7 / (60 + vector_rank) + 0.3 / (60 + bm25_rank)
```
A chunk ranked highly in BOTH searches gets the highest RRF score.
        """)