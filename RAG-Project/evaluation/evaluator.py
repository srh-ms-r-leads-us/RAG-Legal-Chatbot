"""
evaluator.py
------------
Evaluation pipeline for the RAG chatbot.

Measures:
  1. Retrieval accuracy   — did the right source documents come back?
  2. Keyword coverage     — does the answer contain expected keywords?
  3. Response latency     — how long did the full pipeline take?
  4. Source precision     — what % of retrieved sources were relevant?
  5. Confidence scores    — average rerank score across results

Outputs:
  • Console summary table
  • evaluation/results/eval_results_<timestamp>.json
  • evaluation/results/eval_report_<timestamp>.md

Run:
    python evaluation/evaluator.py

Run against a specific number of questions:
    python evaluation/evaluator.py --limit 5
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "ingestion"))
sys.path.insert(0, str(_ROOT / "llm"))

from config import cfg
from ollama_client import OllamaClient
from prompt_builder import build_rag_prompt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BENCHMARK_FILE  = Path(__file__).parent / "benchmark_questions.json"
RESULTS_DIR     = Path(__file__).parent / "results"
API_BASE        = cfg.UI_API_BASE_URL


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_search(query: str, top_k: int = 5) -> dict | None:
    """Call /search and return results."""
    try:
        resp = requests.post(
            f"{API_BASE}/search",
            json={"query": query, "top_k": top_k},
            timeout=30,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        print(f"  API error: {e}")
        return None


def api_context(query: str, top_k: int = 5) -> dict | None:
    """Call /context and return formatted context."""
    try:
        resp = requests.post(
            f"{API_BASE}/context",
            json={"query": query, "top_k": top_k},
            timeout=30,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        print(f"  API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_source_recall(
    retrieved_sources: list[str],
    expected_sources:  list[str],
) -> float:
    """
    Source recall — what fraction of expected sources were retrieved?

    Score = |retrieved ∩ expected| / |expected|
    1.0 = all expected sources found
    0.0 = none of the expected sources found
    """
    if not expected_sources:
        return 1.0

    retrieved_set = {s.lower() for s in retrieved_sources}
    expected_set  = {s.lower() for s in expected_sources}

    hits = sum(
        1 for exp in expected_set
        if any(exp in ret for ret in retrieved_set)
    )
    return round(hits / len(expected_set), 3)


def score_keyword_coverage(
    answer:            str,
    expected_keywords: list[str],
) -> float:
    """
    Keyword coverage — what fraction of expected keywords appear in the answer?

    Score = |found keywords| / |expected keywords|
    """
    if not expected_keywords or not answer:
        return 0.0

    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(found / len(expected_keywords), 3)


def score_source_precision(
    retrieved_sources: list[str],
    expected_sources:  list[str],
) -> float:
    """
    Source precision — what fraction of retrieved sources were relevant?

    Score = |retrieved ∩ expected| / |retrieved|
    """
    if not retrieved_sources:
        return 0.0
    if not expected_sources:
        return 1.0

    retrieved_set = {s.lower() for s in retrieved_sources}
    expected_set  = {s.lower() for s in expected_sources}

    hits = sum(
        1 for ret in retrieved_set
        if any(exp in ret for exp in expected_set)
    )
    return round(hits / len(retrieved_set), 3)


# ---------------------------------------------------------------------------
# Single question evaluation
# ---------------------------------------------------------------------------

def evaluate_question(
    question:          dict,
    ollama:            OllamaClient,
    eval_llm:          bool = True,
) -> dict:
    """
    Run the full pipeline for one benchmark question and score it.

    Args:
        question : benchmark question dict from benchmark_questions.json
        ollama   : OllamaClient instance
        eval_llm : if True, also generate and score an LLM answer

    Returns:
        Result dict with all scores and metadata.
    """
    q_id      = question["id"]
    query     = question["question"]
    exp_kw    = question["expected_keywords"]
    exp_src   = question["expected_sources"]
    category  = question["category"]

    print(f"\n  [{q_id}] {query[:70]}...")

    result = {
        "id"              : q_id,
        "question"        : query,
        "category"        : category,
        "expected_sources": exp_src,
        "expected_keywords": exp_kw,
        # scores
        "source_recall"   : 0.0,
        "source_precision": 0.0,
        "keyword_coverage": 0.0,
        "avg_rerank_score": 0.0,
        "retrieval_latency_ms": 0,
        "llm_latency_ms"  : 0,
        "total_latency_ms": 0,
        "retrieved_sources": [],
        "llm_answer"      : "",
        "error"           : None,
    }

    # ── Step 1: Retrieval ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    search_data = api_search(query)
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    if not search_data or not search_data.get("results"):
        result["error"] = "No retrieval results"
        print(f"    ❌ No results returned")
        return result

    chunks = search_data["results"]
    retrieved_sources = [c["doc_name"] for c in chunks]
    avg_rerank        = sum(c.get("rerank_score", 0) for c in chunks) / len(chunks)

    result["retrieved_sources"]    = retrieved_sources
    result["retrieval_latency_ms"] = retrieval_ms
    result["avg_rerank_score"]     = round(avg_rerank, 3)
    result["source_recall"]        = score_source_recall(retrieved_sources, exp_src)
    result["source_precision"]     = score_source_precision(retrieved_sources, exp_src)

    print(f"    ✅ Retrieved {len(chunks)} chunks in {retrieval_ms}ms  "
          f"| recall: {result['source_recall']}  "
          f"| precision: {result['source_precision']}")

    # ── Step 2: LLM answer (optional) ────────────────────────────────────
    if eval_llm and ollama.is_available():
        context_data = api_context(query)
        if context_data:
            prompt = build_rag_prompt(
                query   = query,
                context = context_data["context"],
            )
            t1 = time.perf_counter()
            try:
                answer = ollama.generate(prompt)
                llm_ms = int((time.perf_counter() - t1) * 1000)
                result["llm_answer"]    = answer
                result["llm_latency_ms"] = llm_ms
                result["keyword_coverage"] = score_keyword_coverage(answer, exp_kw)
                print(f"    ✅ LLM answered in {llm_ms}ms  "
                      f"| keyword coverage: {result['keyword_coverage']}")
            except Exception as e:
                result["error"] = f"LLM error: {e}"
                print(f"    ⚠️  LLM error: {e}")

    result["total_latency_ms"] = result["retrieval_latency_ms"] + result["llm_latency_ms"]
    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list[dict], timestamp: str) -> str:
    """Generate a markdown evaluation report."""
    total     = len(results)
    valid     = [r for r in results if not r.get("error")]
    errors    = [r for r in results if r.get("error")]

    avg_recall    = sum(r["source_recall"]    for r in valid) / max(len(valid), 1)
    avg_precision = sum(r["source_precision"] for r in valid) / max(len(valid), 1)
    avg_coverage  = sum(r["keyword_coverage"] for r in valid) / max(len(valid), 1)
    avg_rerank    = sum(r["avg_rerank_score"] for r in valid) / max(len(valid), 1)
    avg_ret_ms    = sum(r["retrieval_latency_ms"] for r in valid) / max(len(valid), 1)
    avg_llm_ms    = sum(r["llm_latency_ms"] for r in valid) / max(len(valid), 1)

    lines = [
        f"# RAG Chatbot Evaluation Report",
        f"**Generated:** {timestamp}",
        f"**Questions evaluated:** {total}  |  **Errors:** {len(errors)}",
        f"",
        f"## Summary Metrics",
        f"",
        f"| Metric | Score |",
        f"|--------|-------|",
        f"| Source Recall | {avg_recall:.1%} |",
        f"| Source Precision | {avg_precision:.1%} |",
        f"| Keyword Coverage | {avg_coverage:.1%} |",
        f"| Avg Rerank Score | {avg_rerank:.3f} |",
        f"| Avg Retrieval Latency | {avg_ret_ms:.0f} ms |",
        f"| Avg LLM Latency | {avg_llm_ms:.0f} ms |",
        f"",
        f"## Per-Question Results",
        f"",
        f"| ID | Category | Recall | Precision | Keywords | Ret(ms) | LLM(ms) | Error |",
        f"|----|----------|--------|-----------|----------|---------|---------|-------|",
    ]

    for r in results:
        error_str = r.get("error", "") or ""
        lines.append(
            f"| {r['id']} | {r['category']} "
            f"| {r['source_recall']:.0%} "
            f"| {r['source_precision']:.0%} "
            f"| {r['keyword_coverage']:.0%} "
            f"| {r['retrieval_latency_ms']} "
            f"| {r['llm_latency_ms']} "
            f"| {error_str[:30]} |"
        )

    # Category breakdown
    categories = sorted(set(r["category"] for r in valid))
    lines += ["", "## Results by Category", ""]
    for cat in categories:
        cat_results = [r for r in valid if r["category"] == cat]
        cat_recall  = sum(r["source_recall"] for r in cat_results) / len(cat_results)
        lines.append(f"**{cat}** ({len(cat_results)} questions): "
                     f"avg recall {cat_recall:.1%}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(limit: int | None = None, skip_llm: bool = False) -> None:
    """Run the full evaluation pipeline."""
    print("=" * 65)
    print("RAG Chatbot — Evaluation Pipeline")
    print("=" * 65)

    # Load benchmark questions
    with open(BENCHMARK_FILE, encoding="utf-8") as fh:
        questions: list[dict] = json.load(fh)

    if limit:
        questions = questions[:limit]

    print(f"Questions to evaluate : {len(questions)}")
    print(f"LLM evaluation        : {'disabled (--skip-llm)' if skip_llm else 'enabled'}")
    print(f"API base URL          : {API_BASE}")

    # Check API is up
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        if resp.status_code != 200:
            print("❌ API is not responding. Start it with: python api/main.py")
            sys.exit(1)
        health = resp.json()
        print(f"API status            : {health['status']} — {health['total_vectors']} vectors")
    except Exception:
        print("❌ Cannot reach API. Start it with: python api/main.py")
        sys.exit(1)

    ollama = OllamaClient()
    if not skip_llm and not ollama.is_available():
        print("⚠️  Ollama not available — skipping LLM evaluation")
        skip_llm = True

    # Run evaluations
    results = []
    start   = time.perf_counter()

    for i, question in enumerate(questions, start=1):
        print(f"\n[{i}/{len(questions)}]", end="")
        result = evaluate_question(
            question = question,
            ollama   = ollama,
            eval_llm = not skip_llm,
        )
        results.append(result)

    total_time = time.perf_counter() - start

    # Summary
    valid = [r for r in results if not r.get("error")]
    print("\n\n" + "=" * 65)
    print("EVALUATION COMPLETE")
    print("=" * 65)
    print(f"Total time            : {total_time:.1f}s")
    print(f"Questions evaluated   : {len(results)}")
    print(f"Successful            : {len(valid)}")
    print(f"Errors                : {len(results) - len(valid)}")

    if valid:
        print(f"\nAvg Source Recall     : {sum(r['source_recall'] for r in valid)/len(valid):.1%}")
        print(f"Avg Source Precision  : {sum(r['source_precision'] for r in valid)/len(valid):.1%}")
        print(f"Avg Keyword Coverage  : {sum(r['keyword_coverage'] for r in valid)/len(valid):.1%}")
        print(f"Avg Rerank Score      : {sum(r['avg_rerank_score'] for r in valid)/len(valid):.3f}")
        print(f"Avg Retrieval Latency : {sum(r['retrieval_latency_ms'] for r in valid)/len(valid):.0f}ms")
        if not skip_llm:
            print(f"Avg LLM Latency       : {sum(r['llm_latency_ms'] for r in valid)/len(valid):.0f}ms")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = RESULTS_DIR / f"eval_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    md_path = RESULTS_DIR / f"eval_report_{timestamp}.md"
    report  = generate_report(results, timestamp)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"\nResults saved to : {json_path}")
    print(f"Report saved to  : {md_path}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Chatbot Evaluation Pipeline")
    parser.add_argument("--limit",    type=int,  default=None,  help="Evaluate only first N questions")
    parser.add_argument("--skip-llm", action="store_true",      help="Skip LLM generation (retrieval only)")
    args = parser.parse_args()
    run(limit=args.limit, skip_llm=args.skip_llm)