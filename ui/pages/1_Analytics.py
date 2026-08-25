"""
1_Analytics.py
--------------
Analytics page for the UNECE Policy Chatbot.

Shows:
  • Feedback summary (helpful vs unhelpful)
  • Most common queries
  • Worst performing queries (thumbs down)
  • Evaluation results (if evaluator has been run)
  • Collection statistics

Access:
    Streamlit automatically adds this as a second page in the sidebar
    when placed in ui/pages/
"""

import json
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "ingestion"))

from config import cfg

st.set_page_config(
    page_title="Analytics — UNECE Chatbot",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Analytics Dashboard")
st.caption("Feedback analysis and evaluation results for the UNECE Policy Chatbot.")


# ---------------------------------------------------------------------------
# Feedback analysis
# ---------------------------------------------------------------------------

FEEDBACK_FILE = cfg.OUTPUT_DIR / "feedback.jsonl"

st.header("💬 User Feedback")

if not FEEDBACK_FILE.exists():
    st.info("No feedback collected yet. Ask questions in the chatbot and rate the answers.")
else:
    # Load feedback
    entries = []
    with open(FEEDBACK_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass

    if not entries:
        st.info("Feedback file exists but is empty.")
    else:
        total     = len(entries)
        helpful   = sum(1 for e in entries if e.get("helpful") is True)
        unhelpful = total - helpful
        pct       = round(helpful / total * 100, 1) if total else 0

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Ratings", total)
        c2.metric("👍 Helpful",    helpful)
        c3.metric("👎 Not Helpful", unhelpful)
        c4.metric("Satisfaction",  f"{pct}%")

        st.divider()

        # Helpful vs unhelpful bar
        st.subheader("Satisfaction Breakdown")
        st.progress(helpful / total if total else 0)
        st.caption(f"{helpful} helpful / {unhelpful} not helpful out of {total} ratings")

        st.divider()

        # Thumbs down queries — most important for improvement
        bad = [e for e in entries if e.get("helpful") is False]
        if bad:
            st.subheader("⚠️ Queries That Got Thumbs Down")
            st.caption("These are the questions the chatbot answered poorly — fix these first.")
            for e in bad[-10:]:    # show last 10
                with st.expander(f"❌ {e['query'][:80]}", expanded=False):
                    st.markdown(f"**Query:** {e['query']}")
                    if e.get("llm_answer"):
                        st.markdown(f"**Answer given:** {e['llm_answer'][:500]}...")
                    if e.get("citations"):
                        st.markdown(f"**Sources used:** {', '.join(e['citations'])}")
                    if e.get("comment"):
                        st.markdown(f"**User comment:** {e['comment']}")
                    st.caption(f"Timestamp: {e.get('timestamp', 'unknown')}")

        st.divider()

        # All feedback table
        st.subheader("All Feedback Entries")
        table_data = [
            {
                "Timestamp": e.get("timestamp", "")[:19],
                "Query"    : e["query"][:60] + ("..." if len(e["query"]) > 60 else ""),
                "Helpful"  : "👍" if e.get("helpful") else "👎",
                "Comment"  : (e.get("comment") or "")[:40],
            }
            for e in reversed(entries)   # newest first
        ]
        st.dataframe(table_data, use_container_width=True)


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------

st.header("🧪 Evaluation Results")

RESULTS_DIR = _ROOT / "evaluation" / "results"

if not RESULTS_DIR.exists() or not list(RESULTS_DIR.glob("eval_results_*.json")):
    st.info(
        "No evaluation results yet.\n\n"
        "Run the evaluator with:\n"
        "```\npython evaluation/evaluator.py --skip-llm\n```"
    )
else:
    # Load the most recent evaluation
    result_files = sorted(RESULTS_DIR.glob("eval_results_*.json"), reverse=True)
    latest       = result_files[0]

    with open(latest, encoding="utf-8") as fh:
        eval_results: list[dict] = json.load(fh)

    st.caption(f"Showing results from: `{latest.name}`")

    # Summary metrics
    valid = [r for r in eval_results if not r.get("error")]
    if valid:
        avg_recall    = sum(r["source_recall"]    for r in valid) / len(valid)
        avg_precision = sum(r["source_precision"] for r in valid) / len(valid)
        avg_coverage  = sum(r["keyword_coverage"] for r in valid) / len(valid)
        avg_rerank    = sum(r["avg_rerank_score"] for r in valid) / len(valid)
        avg_ret_ms    = sum(r["retrieval_latency_ms"] for r in valid) / len(valid)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Source Recall",    f"{avg_recall:.1%}")
        c2.metric("Source Precision", f"{avg_precision:.1%}")
        c3.metric("Keyword Coverage", f"{avg_coverage:.1%}")
        c4.metric("Avg Rerank Score", f"{avg_rerank:.2f}")
        c5.metric("Avg Retrieval",    f"{avg_ret_ms:.0f}ms")

        st.divider()

        # Results by category
        st.subheader("Results by Category")
        categories = sorted(set(r["category"] for r in valid))
        cat_data   = []
        for cat in categories:
            cat_r = [r for r in valid if r["category"] == cat]
            cat_data.append({
                "Category"        : cat,
                "Questions"       : len(cat_r),
                "Avg Recall"      : f"{sum(r['source_recall'] for r in cat_r)/len(cat_r):.1%}",
                "Avg Precision"   : f"{sum(r['source_precision'] for r in cat_r)/len(cat_r):.1%}",
                "Avg Keywords"    : f"{sum(r['keyword_coverage'] for r in cat_r)/len(cat_r):.1%}",
                "Avg Latency(ms)" : f"{sum(r['retrieval_latency_ms'] for r in cat_r)/len(cat_r):.0f}",
            })
        st.dataframe(cat_data, use_container_width=True)

        st.divider()

        # Per-question details
        st.subheader("Per-Question Results")
        q_data = [
            {
                "ID"       : r["id"],
                "Category" : r["category"],
                "Question" : r["question"][:60] + "...",
                "Recall"   : f"{r['source_recall']:.0%}",
                "Precision": f"{r['source_precision']:.0%}",
                "Keywords" : f"{r['keyword_coverage']:.0%}",
                "Ret(ms)"  : r["retrieval_latency_ms"],
                "Error"    : r.get("error") or "",
            }
            for r in eval_results
        ]
        st.dataframe(q_data, use_container_width=True)

        # Download report
        report_files = sorted(RESULTS_DIR.glob("eval_report_*.md"), reverse=True)
        if report_files:
            with open(report_files[0], encoding="utf-8") as fh:
                report_md = fh.read()
            st.download_button(
                label    = "📥 Download Evaluation Report (Markdown)",
                data     = report_md,
                file_name= report_files[0].name,
                mime     = "text/markdown",
            )


# ---------------------------------------------------------------------------
# Collection stats
# ---------------------------------------------------------------------------

st.header("🗄️ Vector Store Statistics")

try:
    import requests
    health = requests.get(f"{cfg.UI_API_BASE_URL}/health", timeout=5).json()
    docs   = requests.get(f"{cfg.UI_API_BASE_URL}/documents", timeout=5).json()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Vectors",   health.get("total_vectors", "N/A"))
    c2.metric("Total Documents", docs.get("total_documents", "N/A"))
    c3.metric("Total Chunks",    docs.get("total_chunks",    "N/A"))

    st.caption(f"Collection: `{health.get('collection', 'N/A')}`")
    st.caption(f"Embed model: `{health.get('embed_model', 'N/A')}`")
    st.caption(f"Hybrid search: {health.get('hybrid_search', 'N/A')} | "
               f"Reranking: {health.get('reranking', 'N/A')}")

except Exception:
    st.warning("Could not load collection stats — is the API running?")