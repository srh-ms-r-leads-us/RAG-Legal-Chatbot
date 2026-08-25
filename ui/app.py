"""
app.py
------
UNECE Policy Chatbot — Advanced Streamlit UI

Features
--------
  ✅ Chat history with conversation memory
  ✅ Streaming LLM responses with blinking cursor
  ✅ Source viewer with confidence bars per citation
  ✅ Query validation — greetings handled without retrieval
  ✅ Answer caching — identical queries return instantly
  ✅ Document upload — drop a PDF and ingest it directly
  ✅ Better error messages — specific guidance per error type
  ✅ Feedback buttons (thumbs up/down) per answer
  ✅ Settings sidebar with all retrieval controls
  ✅ Pipeline status indicators

Run:
    streamlit run ui/app.py
"""

import hashlib
import subprocess
import sys
import time
from pathlib import Path

import requests
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "ingestion"))
sys.path.insert(0, str(_ROOT / "llm"))

from config import cfg
from ollama_client import OllamaClient
from prompt_builder import (
    build_error_response,
    build_no_context_response,
    build_rag_prompt,
    get_greeting_response,
    is_greeting_or_offtopic,
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=cfg.UI_PAGE_TITLE,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Confidence bar colours */
.conf-high   { color: #22c55e; font-weight: 600; }
.conf-medium { color: #f59e0b; font-weight: 600; }
.conf-low    { color: #ef4444; font-weight: 600; }

/* Source citation card */
.source-card {
    background: #1e293b;
    border-left: 3px solid #6366f1;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.85rem;
}

/* Tighter chat spacing */
.stChatMessage { margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session():
    defaults = {
        "messages"      : [],          # [{role, content, citations, confidences}]
        "cache"         : {},          # {query_hash: {content, citations, confidences}}
        "feedback_sent" : set(),       # message indices already rated
        "ollama"        : None,        # shared OllamaClient
        "upload_status" : None,        # last upload result message
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.ollama is None:
        st.session_state.ollama = OllamaClient()


init_session()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(query: str, top_k: int, min_score: float, model: str) -> str:
    """Generate a cache key from query + settings."""
    raw = f"{query.strip().lower()}|{top_k}|{min_score}|{model}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_cached(key: str) -> dict | None:
    return st.session_state.cache.get(key)


def set_cached(key: str, value: dict) -> None:
    # Keep cache size bounded — drop oldest if too large
    if len(st.session_state.cache) > 50:
        oldest = next(iter(st.session_state.cache))
        del st.session_state.cache[oldest]
    st.session_state.cache[key] = value


# ---------------------------------------------------------------------------
# API helpers — cached to prevent re-running on every Streamlit rerun
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def api_health() -> dict | None:
    """
    Cached health check — re-runs at most once every 30 seconds.
    Without caching this fires on every single UI interaction.
    """
    try:
        r = requests.get(f"{cfg.UI_API_BASE_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def api_documents() -> list[dict]:
    """
    Cached document list — re-runs at most once every 60 seconds.
    Documents rarely change so a longer TTL is fine.
    """
    try:
        r = requests.get(f"{cfg.UI_API_BASE_URL}/documents", timeout=5)
        return r.json().get("documents", []) if r.status_code == 200 else []
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def get_available_models() -> list[str]:
    """
    Cached Ollama model list — re-runs at most once every 60 seconds.
    Pulling the model list from Ollama on every keystroke is expensive.
    """
    try:
        r = requests.get(f"{cfg.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            models = r.json().get("models", [])
            return [m["name"] for m in models]
    except Exception:
        pass
    return [cfg.OLLAMA_MODEL]


@st.cache_data(ttl=10, show_spinner=False)
def check_ollama_available() -> bool:
    """
    Cached Ollama availability check — re-runs at most once every 10 seconds.
    """
    try:
        r = requests.get(f"{cfg.OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def api_context(query: str, top_k: int, min_score: float) -> dict | None:
    """
    NOT cached — every query must hit the retrieval engine fresh.
    Query-level caching is handled separately in session_state.cache.
    """
    try:
        r = requests.post(
            f"{cfg.UI_API_BASE_URL}/context",
            json={"query": query, "top_k": top_k, "min_score": min_score},
            timeout=30,
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_feedback(query: str, helpful: bool, answer: str, citations: list[str]) -> None:
    try:
        requests.post(
            f"{cfg.UI_API_BASE_URL}/feedback",
            json={"query": query, "helpful": helpful,
                  "llm_answer": answer, "citations": citations},
            timeout=10,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Confidence bar renderer
# ---------------------------------------------------------------------------

def render_score_breakdown(conf: dict) -> None:
    """
    Render full retrieval score breakdown for one source.
    Shows all four scores so the retrieval pipeline is fully transparent:
      - Vector similarity  → semantic meaning match
      - BM25 score         → keyword match
      - RRF score          → hybrid fusion score
      - Rerank score       → final cross-encoder relevance
    """
    citation     = conf.get("citation",     "unknown")
    label        = conf.get("label",        "Low")
    similarity   = conf.get("similarity",   0.0)
    bm25_score   = conf.get("bm25_score",   0.0)
    rrf_score    = conf.get("rrf_score",    0.0)
    rerank_score = conf.get("rerank_score", 0.0)

    col_class = (
        "conf-high"   if label == "High"   else
        "conf-medium" if label == "Medium" else
        "conf-low"
    )

    # Source header with confidence label
    st.markdown(
        f'<div class="source-card" style="background:#1e293b; border-left:3px solid #6366f1; '
        f'border-radius:6px; padding:8px 12px; margin:4px 0; font-size:0.85rem; color:#e2e8f0;">'
        f'📄 <b style="color:#e2e8f0;">{citation}</b> &nbsp;|&nbsp; '
        f'<span class="{col_class}">{label} confidence</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Four score bars in 2 columns
    c1, c2 = st.columns(2)

    with c1:
        # Vector similarity — already 0-1
        st.caption("🔵 Vector Similarity (semantic)")
        st.progress(float(min(max(similarity, 0.0), 1.0)))
        st.caption(f"`{similarity:.4f}`")

        # RRF score — typical range 0 to 0.02, normalise for display
        st.caption("🟣 RRF Score (hybrid fusion)")
        rrf_norm = float(min(rrf_score / 0.02, 1.0)) if rrf_score > 0 else 0.0
        st.progress(rrf_norm)
        st.caption(f"`{rrf_score:.6f}`")

    with c2:
        # BM25 — typical range 0 to 50, normalise for display
        st.caption("🟡 BM25 Score (keyword match)")
        bm25_norm = float(min(bm25_score / 50.0, 1.0)) if bm25_score > 0 else 0.0
        st.progress(bm25_norm)
        st.caption(f"`{bm25_score:.4f}`")

        # Rerank — logit range approx -5 to +10, normalise to 0-1
        st.caption("🟢 Rerank Score (final relevance)")
        rerank_norm = float(min(max((rerank_score + 5) / 15, 0.0), 1.0))
        st.progress(rerank_norm)
        st.caption(f"`{rerank_score:.4f}`")

    st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    with st.sidebar:
        st.title("⚙️ Settings")

        # ── Status ────────────────────────────────────────────────────────
        st.subheader("Connection")
        health   = api_health()
        ollama_ok = check_ollama_available()

        if health:
            st.success(f"✅ API — {health['total_vectors']} vectors")
        else:
            st.error("❌ API offline\n`python api/main.py`")

        if ollama_ok:
            st.success("✅ Ollama online")
        else:
            st.error("❌ Ollama offline\n`ollama serve`")

        st.divider()

        # ── Model ─────────────────────────────────────────────────────────
        st.subheader("🤖 LLM Model")
        models  = get_available_models()
        def_idx = models.index(cfg.OLLAMA_MODEL) if cfg.OLLAMA_MODEL in models else 0
        model   = st.selectbox("Model", models, index=def_idx,
                               help="Pull new models with: ollama pull <name>")

        st.divider()

        # ── Retrieval ─────────────────────────────────────────────────────
        st.subheader("🔍 Retrieval Settings")
        top_k     = st.slider("Results (top_k)", 1, cfg.API_MAX_TOP_K,
                               cfg.RETRIEVAL_TOP_K,
                               help="Number of document chunks retrieved per query.")
        min_score = st.slider("Min similarity", 0.0, 1.0,
                               cfg.RETRIEVAL_MIN_SCORE, step=0.05,
                               help="Chunks below this score are discarded.")

        st.divider()

        # ── Pipeline info ─────────────────────────────────────────────────
        st.subheader("⚡ Pipeline")
        c1, c2 = st.columns(2)
        c1.metric("Hybrid", "ON" if cfg.HYBRID_SEARCH_ENABLED else "OFF")
        c2.metric("Rerank", "ON" if cfg.RERANKER_ENABLED      else "OFF")
        st.caption(f"Embed: `{cfg.EMBED_MODEL}`")

        st.divider()

        # ── Document upload ───────────────────────────────────────────────
        st.subheader("📤 Add Document")
        uploaded = st.file_uploader(
            "Upload a PDF to ingest",
            type=["pdf"],
            help="Drop a new PDF here — it will be ingested automatically.",
        )
        if uploaded:
            save_path = cfg.PDF_DIR / uploaded.name
            if save_path.exists():
                st.warning(f"'{uploaded.name}' already exists. It will be updated.")

            if st.button("Ingest PDF", use_container_width=True):
                with st.spinner(f"Ingesting {uploaded.name} ..."):
                    try:
                        # Save uploaded file to raw/ folder
                        cfg.PDF_DIR.mkdir(parents=True, exist_ok=True)
                        with open(save_path, "wb") as fh:
                            fh.write(uploaded.read())

                        # Run ingestion_manager.py as subprocess
                        result = subprocess.run(
                            [sys.executable,
                             str(_ROOT / "ingestion" / "ingestion_manager.py")],
                            capture_output=True, text=True, timeout=300,
                        )
                        if result.returncode == 0:
                            st.success(f"✅ '{uploaded.name}' ingested successfully!")
                            # Clear cache so new doc is searchable immediately
                            st.session_state.cache = {}
                        else:
                            st.error(f"Ingestion failed:\n{result.stderr[-500:]}")
                    except subprocess.TimeoutExpired:
                        st.error("Ingestion timed out. Try again.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.divider()

        # ── Document browser ──────────────────────────────────────────────
        st.subheader("📚 Documents")
        docs = api_documents()
        if docs:
            for doc in docs:
                with st.expander(f"📄 {doc['filename']}", expanded=False):
                    st.caption(f"Chunks: {doc['chunk_count']}")
                    st.caption(f"Added: {doc['ingested_at'][:10]}")
        else:
            st.caption("No documents found.")

        st.divider()

        # ── Clear cache + history ─────────────────────────────────────────
        c1, c2 = st.columns(2)
        if c1.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages     = []
            st.session_state.feedback_sent = set()
            st.rerun()
        if c2.button("♻️ Clear cache", use_container_width=True):
            st.session_state.cache = {}
            st.toast("Cache cleared!")

    return {"model": model, "top_k": top_k, "min_score": min_score}


# ---------------------------------------------------------------------------
# Message renderer
# ---------------------------------------------------------------------------

def render_message(msg: dict, idx: int) -> None:
    """Render a single message with sources, confidence bars, and feedback."""
    role = msg["role"]
    with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
        st.markdown(msg["content"])

        # Source viewer with confidence bars
        if role == "assistant":
            citations   = msg.get("citations",   [])
            confidences = msg.get("confidences", [])

            if citations:
                with st.expander(f"📄 Sources ({len(citations)}) — click to see retrieval scores", expanded=False):
                    if confidences:
                        for conf in confidences:
                            render_score_breakdown(conf)
                    else:
                        for i, cit in enumerate(citations, 1):
                            st.markdown(f"**{i}.** {cit}")

            # Feedback buttons
            if idx not in st.session_state.feedback_sent:
                c1, c2, c3 = st.columns([1, 1, 8])
                prev_query  = ""
                if idx > 0 and st.session_state.messages[idx - 1]["role"] == "user":
                    prev_query = st.session_state.messages[idx - 1]["content"]

                with c1:
                    if st.button("👍", key=f"up_{idx}"):
                        api_feedback(prev_query, True,
                                     msg["content"], citations)
                        st.session_state.feedback_sent.add(idx)
                        st.toast("Thanks! 👍")
                        st.rerun()
                with c2:
                    if st.button("👎", key=f"dn_{idx}"):
                        api_feedback(prev_query, False,
                                     msg["content"], citations)
                        st.session_state.feedback_sent.add(idx)
                        st.toast("Thanks for the feedback — we'll improve!")
                        st.rerun()
            else:
                st.caption("✅ Feedback received")


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def rewrite_query_with_history(query: str, history: list[dict]) -> str:
    """
    Rewrite a follow-up query to be self-contained using conversation history.

    Problem:
        User asks "What causes loneliness?" → answer given
        User asks "What did Finland do about it?" ← "it" is ambiguous
        Retrieval searches "Finland do about it" → finds nothing

    Solution:
        Rewrite to: "What did Finland do about loneliness among older persons?"
        Retrieval now finds the right chunks.

    Only rewrites if the query contains pronouns or references that
    depend on conversation history (it, this, that, they, them, those).
    """
    # Only rewrite if there is history AND query contains ambiguous references
    ambiguous_words = ["it", "this", "that", "they", "them", "those",
                       "these", "there", "such", "the same", "similar",
                       "about it", "do about", "what about"]

    query_lower = query.lower()
    needs_rewrite = (
        len(history) >= 2 and
        any(word in query_lower for word in ambiguous_words) and
        len(query.split()) < 15   # short follow-up queries need rewriting
    )

    if not needs_rewrite:
        return query   # return original — no rewriting needed

    # Build a compact conversation summary for the rewriter
    recent = history[-4:]   # last 2 exchanges
    conversation = ""
    for msg in recent:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:300]
        conversation += f"{role}: {content}\n"

    # Ask the LLM to rewrite the query
    rewrite_prompt = f"""Given this conversation:
{conversation}

Rewrite this follow-up question to be completely self-contained
(replace pronouns like 'it', 'this', 'that' with the actual topic):

Follow-up: {query}

Rewritten question (one sentence only, no explanation):"""

    try:
        ollama = st.session_state.ollama
        rewritten = ollama.generate(rewrite_prompt, model=cfg.OLLAMA_MODEL)
        rewritten = rewritten.strip().strip('"').strip("'")

        # Sanity check — if rewrite is too long or empty, use original
        if rewritten and 5 < len(rewritten.split()) < 30:
            return rewritten
    except Exception:
        pass

    return query   # fallback to original if rewrite fails


def generate_answer(
    query:     str,
    model:     str,
    top_k:     int,
    min_score: float,
) -> tuple[str, list[str], list[dict]]:
    """
    Full RAG pipeline with caching and improved error handling.

    Returns: (answer, citations, confidences)
    """
    ollama = st.session_state.ollama

    # ── Cache check ───────────────────────────────────────────────────────
    cache_key = _cache_key(query, top_k, min_score, model)
    cached    = get_cached(cache_key)
    if cached:
        st.toast("⚡ Cached response", icon="⚡")
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(cached["content"])
            if cached["citations"]:
                with st.expander(
                    f"📄 Sources ({len(cached['citations'])}) — click to see retrieval scores",
                    expanded=False
                ):
                    if cached.get("confidences"):
                        for conf in cached["confidences"]:
                            render_score_breakdown(conf)
                    else:
                        for i, cit in enumerate(cached["citations"], 1):
                            st.markdown(f"**{i}.** {cit}")
        return cached["content"], cached["citations"], cached.get("confidences", [])

    # ── Greeting / off-topic check ────────────────────────────────────────
    if is_greeting_or_offtopic(query):
        response = get_greeting_response(query)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response)
        return response, [], []

    # ── Query rewriting for follow-up questions ───────────────────────────
    # Rewrite ambiguous follow-up queries using conversation history
    # so retrieval can find the right chunks
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-8:]
    ]
    retrieval_query = rewrite_query_with_history(query, history)

    # Show rewritten query if it changed
    if retrieval_query != query:
        st.caption(f"🔄 Searching for: *{retrieval_query}*")

    # ── Step 1: Retrieve context ──────────────────────────────────────────
    with st.spinner("🔍 Searching documents ..."):
        context_data = api_context(retrieval_query, top_k, min_score)

    if not context_data or context_data.get("total_results", 0) == 0:
        response = build_no_context_response()
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response)
        return response, [], []

    context     = context_data["context"]
    citations   = context_data["citations"]
    confidences = context_data.get("confidences", [])

    # ── Step 2: Build prompt with conversation history ────────────────────
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-8:]
    ]
    prompt = build_rag_prompt(query=query, context=context, history=history)

    # ── Step 3: Stream LLM response ───────────────────────────────────────
    if not check_ollama_available():
        response = build_error_response("connection")
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response)
        return response, citations, confidences

    answer_parts: list[str] = []

    try:
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            full_text   = ""

            for token in ollama.stream(prompt, model=model):
                full_text += token
                placeholder.markdown(full_text + "▌")
                answer_parts.append(token)

            placeholder.markdown(full_text)

            # Sources with full score breakdown
            if citations:
                with st.expander(f"📄 Sources ({len(citations)}) — click to see retrieval scores", expanded=False):
                    if confidences:
                        for conf in confidences:
                            render_score_breakdown(conf)
                    else:
                        for i, c in enumerate(citations, 1):
                            st.markdown(f"**{i}.** {c}")

    except ConnectionError:
        response = build_error_response("connection")
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response)
        return response, citations, confidences

    except requests.exceptions.Timeout:
        response = build_error_response("timeout")
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response)
        return response, citations, confidences

    except Exception:
        response = build_error_response("general")
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response)
        return response, citations, confidences

    answer = "".join(answer_parts)

    # ── Cache the successful response ─────────────────────────────────────
    set_cached(cache_key, {
        "content"    : answer,
        "citations"  : citations,
        "confidences": confidences,
    })

    return answer, citations, confidences


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title(f"📄 {cfg.UI_PAGE_TITLE}")
    st.caption(
        "Ask questions about UNECE policy documents on ageing, "
        "demographics, and workforce policy."
    )

    settings = render_sidebar()

    # Render existing chat history
    for i, msg in enumerate(st.session_state.messages):
        render_message(msg, i)

    # Chat input
    query = st.chat_input("Ask a question about the UNECE policy documents ...")

    if query:
        # Trim history if too long
        if len(st.session_state.messages) >= cfg.UI_MAX_HISTORY:
            st.session_state.messages = st.session_state.messages[-cfg.UI_MAX_HISTORY:]

        # Render user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        # Generate answer
        answer, citations, confidences = generate_answer(
            query     = query,
            model     = settings["model"],
            top_k     = settings["top_k"],
            min_score = settings["min_score"],
        )

        # Save to history
        st.session_state.messages.append({
            "role"       : "assistant",
            "content"    : answer,
            "citations"  : citations,
            "confidences": confidences,
        })


if __name__ == "__main__":
    main()