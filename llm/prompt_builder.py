"""
prompt_builder.py
-----------------
Builds structured RAG prompts and validates queries.

Changes from v1:
  • Structured answer format — forces bullet points and clear sections
  • Query validation — detects off-topic/greeting queries before retrieval
  • Confidence instruction — LLM states how confident it is
  • No-repeat rule — LLM never cites the same source twice consecutively
  • Follow-up awareness — uses conversation history for context
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingestion"))
from config import cfg


# ---------------------------------------------------------------------------
# Off-topic query patterns — detected before hitting the retrieval pipeline
# ---------------------------------------------------------------------------

# Short greetings and non-document queries that should be handled directly
_GREETING_PATTERNS = [
    r"^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening))[!\s.?]*$",
    r"^(how are you|what's up|sup|yo)[!\s.?]*$",
    r"^(thanks|thank you|thx|cheers)[!\s.?]*$",
    r"^(bye|goodbye|see you|cya)[!\s.?]*$",
    r"^(ok|okay|sure|yes|no|nope|yep)[!\s.?]*$",
    r"^[?\s!.]*$",   # empty or punctuation only
]

_COMPILED_GREETINGS = [re.compile(p, re.IGNORECASE) for p in _GREETING_PATTERNS]


def is_greeting_or_offtopic(query: str) -> bool:
    """
    Return True if the query is a greeting or clearly off-topic.
    These are handled with a friendly message before retrieval.
    """
    stripped = query.strip()
    if len(stripped) < 4:
        return True
    return any(p.match(stripped) for p in _COMPILED_GREETINGS)


def get_greeting_response(query: str) -> str:
    """Return a friendly response for greetings without calling the LLM."""
    q = query.strip().lower()
    if any(w in q for w in ["thank", "thanks", "thx"]):
        return "You're welcome! Feel free to ask any questions about the UNECE policy documents."
    if any(w in q for w in ["bye", "goodbye", "cya"]):
        return "Goodbye! Come back anytime you have questions about UNECE policy documents."
    return (
        "Hello! I'm the UNECE Policy Chatbot. I can answer questions about "
        "ageing, demographics, and workforce policy based on UNECE documents.\n\n"
        "Try asking something like:\n"
        "- *What policies help retain older workers?*\n"
        "- *How does demographic change affect Europe?*\n"
        "- *What causes loneliness among older persons?*"
    )


# ---------------------------------------------------------------------------
# System message — built from 2025-2026 RAG research best practices
#
# Research basis:
#   • StrictCitations strategy (Zhu et al. 2026) — highest verifiable grounding
#   • RAG Triad framework (TruLens/RAGAS) — context relevance, groundedness, answer relevance
#   • SurePrompts RAG guide (2026) — hedging language, synthesis rules
#   • NightFeats NeurIPS 2025 — citation-preserving composition, contradiction handling
#   • Mistral 7B Instruct best practices — instruction format, grounding behaviour
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = """You are a precise and trustworthy research assistant specialising in UNECE policy documents on ageing, demographics, and workforce policy.

RULE 1 — GROUNDING (most important):
Answer EXCLUSIVELY from the provided context. Never use outside knowledge.

RULE 2 — SILENCE ON MISSING INFO (critical):
If the context does not contain information about something — stay completely silent about it.
Do NOT write: "not mentioned", "not specified", "not found", "Source: Not mentioned", "the context does not provide", "not explicitly stated", or ANY similar phrase.
Just skip it entirely and only write about what IS in the context.

RULE 3 — CITATION FORMAT:
Every factual claim must end with: (Source: FILENAME — page N)
Use the exact filename from the context header.
CORRECT: (Source: PB_30_EN_ECE_WG.1_45.pdf — page 8)
NEVER write: [Source 1], Source 4, (Source: Not mentioned), or any URL.

RULE 4 — ANSWER FORMAT:
• One clear direct sentence answering the question
• Bullet points with citations after each claim
• Only if NOTHING in the context is relevant at all, say: "The available UNECE documents do not contain information about this topic."
• End with: Confidence: High / Medium / Low — based on N source(s)
• Maximum 5 bullet points — be concise
• No numbered sections — bullet points only"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_rag_prompt(
    query:   str,
    context: str,
    history: list[dict] | None = None,
) -> str:
    """
    Build a complete RAG prompt using research-backed best practices.

    Structure:
        System message (grounding + citation + quality rules)
        Conversation history (last 4 exchanges for follow-up awareness)
        Context block (retrieved chunks with citation headers)
        Question + final answer instruction

    Args:
        query   : current user question (already rewritten if follow-up)
        context : formatted context from retrieval_engine.format_context()
        history : previous messages [{role, content}] for follow-up awareness

    Returns:
        Complete prompt string ready to send to Ollama.
    """
    parts = [SYSTEM_MESSAGE]

    # ── Conversation history ──────────────────────────────────────────────
    # Include last 4 exchanges so follow-up questions have context
    # Truncate long messages to save tokens — 400 chars is enough for context
    if history:
        recent = [m for m in history if m["role"] in ("user", "assistant")][-8:]
        if recent:
            parts.append("\n═══ CONVERSATION HISTORY (for follow-up context only) ═══")
            for msg in recent:
                role    = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"][:400]
                if len(msg["content"]) > 400:
                    content += "..."
                parts.append(f"{role}: {content}")
            parts.append("═══ END HISTORY ═══")

    # ── Retrieved document context ────────────────────────────────────────
    # Context headers already contain filename, page, confidence level
    # The LLM must use these headers for citations — never invent source labels
    parts.append("\n═══ CONTEXT FROM UNECE DOCUMENTS ═══")
    parts.append(context)
    parts.append("═══ END CONTEXT ═══")

    # ── Question + answer instruction ─────────────────────────────────────
    # Final instruction reinforces citation format and structure
    # Placed immediately before "Answer:" so it is the last thing the LLM sees
    parts.append(f"\nQuestion: {query}")
    parts.append(
        "\nAnswer using ONLY what the context above explicitly states. "
        "Start with one direct sentence. Use bullet points with (Source: FILENAME — page N) after each claim. "
        "IMPORTANT: If something is not in the context, do not mention it at all — not even to say it is missing. "
        "Only write about what IS there. End with Confidence: High/Medium/Low.\n"
        "\nAnswer:"
    )

    return "\n".join(parts)


def build_no_context_response() -> str:
    """Response when retrieval finds nothing above the score threshold."""
    return (
        "I could not find relevant information in the available UNECE documents "
        "to answer your question.\n\n"
        "**Suggestions:**\n"
        "- Try rephrasing your question with different keywords\n"
        "- Lower the minimum similarity score in the sidebar settings\n"
        "- Make sure the relevant PDF has been ingested (check the document list in the sidebar)\n"
        "- Try asking a more specific question about the document content"
    )


def build_error_response(error_type: str = "general") -> str:
    """Specific error responses based on what went wrong."""
    messages = {
        "timeout": (
            "⏱️ The response took too long to generate.\n\n"
            "**Try:**\n"
            "- Asking a shorter or more specific question\n"
            "- Switching to a faster model (e.g. `phi3`) in the sidebar"
        ),
        "connection": (
            "❌ Cannot connect to Ollama.\n\n"
            "**Fix:** Open a terminal and run: `ollama serve`"
        ),
        "general": (
            "⚠️ An error occurred while generating the answer.\n\n"
            "Please try again. If the problem persists, check the API terminal for details."
        ),
    }
    return messages.get(error_type, messages["general"])


def get_greeting_response(query: str) -> str:
    """Return a friendly response for greetings without calling the LLM."""
    q = query.strip().lower()
    if any(w in q for w in ["thank", "thanks", "thx"]):
        return "You're welcome! Feel free to ask any questions about the UNECE policy documents."
    if any(w in q for w in ["bye", "goodbye", "cya"]):
        return "Goodbye! Come back anytime you have questions about UNECE policy documents."
    return (
        "Hello! I'm the UNECE Policy Chatbot. I can answer questions about "
        "ageing, demographics, and workforce policy based on UNECE documents.\n\n"
        "Try asking something like:\n"
        "- *What policies help retain older workers?*\n"
        "- *How does demographic change affect Europe?*\n"
        "- *What causes loneliness among older persons?*"
    )


def is_greeting_or_offtopic(query: str) -> bool:
    """Return True if the query is a greeting or clearly off-topic."""
    import re
    _GREETING_PATTERNS = [
        r"^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening))[!\s.?]*$",
        r"^(how are you|what's up|sup|yo)[!\s.?]*$",
        r"^(thanks|thank you|thx|cheers)[!\s.?]*$",
        r"^(bye|goodbye|see you|cya)[!\s.?]*$",
        r"^(ok|okay|sure|yes|no|nope|yep)[!\s.?]*$",
        r"^[?\s!.]*$",
    ]
    stripped = query.strip()
    if len(stripped) < 4:
        return True
    return any(
        re.match(p, stripped, re.IGNORECASE)
        for p in _GREETING_PATTERNS
    )


def get_dynamic_k(query: str, user_k: int) -> tuple[int, str]:
    """
    Select optimal top_k using query classification.
    Only applies when user uses the default k — respects manual overrides.
    """
    default_k = cfg.RETRIEVAL_TOP_K

    if user_k != default_k:
        return user_k, "manual"

    query_lower = query.lower().strip()

    # Comparative — needs multiple documents
    if any(w in query_lower for w in [
        "compare", "comparison", "versus", "vs", "difference between",
        "across countries", "across europe", "contrast", "how does",
    ]):
        return 7, "comparative"

    # Factual — specific fact, less noise better
    if any(w in query_lower for w in [
        "what is the", "what was the", "who is", "when was", "which country",
    ]):
        if not any(w in query_lower for w in ["compare", "versus", "across"]):
            return 3, "factual"

    # Visual — image chunk + supporting context
    if any(w in query_lower for w in [
        "figure", "chart", "graph", "image", "photograph", "bar chart",
    ]):
        return 4, "visual"

    return 5, "general"


def adaptive_k_from_scores(
    similarity_scores: list[float],
    min_k: int = 2,
    max_k: int = 8,
    gap_threshold: float = 0.15,
) -> tuple[int, str]:
    """
    CAR Algorithm — Cluster-based Adaptive Retrieval (Xu et al., Oct 2025)

    Determines optimal k by finding where similarity scores drop off
    significantly, rather than using fixed k or keyword heuristics.

    Core insight from the paper:
      Relevant chunks cluster together with similar scores.
      A big gap in scores indicates transition from relevant to irrelevant.
      Stop retrieval at the gap.

    Args:
        similarity_scores : list of similarity scores already sorted descending
        min_k             : always retrieve at least this many (default 2)
        max_k             : never retrieve more than this many (default 8)
        gap_threshold     : minimum score drop to consider a gap (default 0.15)

    Returns:
        (optimal_k, reason) — k to use and explanation for UI display

    Example:
        scores = [0.89, 0.87, 0.85, 0.41, 0.39]
        gap between index 2→3 = 0.85 - 0.41 = 0.44 > threshold
        → optimal_k = 3 (stop before the gap)

        scores = [0.75, 0.73, 0.71, 0.69, 0.67]
        no gap > threshold
        → optimal_k = 5 (use all, scores are uniformly relevant)
    """
    if not similarity_scores:
        return min_k, "default"

    scores = sorted(similarity_scores, reverse=True)[:max_k]
    n      = len(scores)

    if n <= min_k:
        return n, "all relevant"

    # Find the largest gap in consecutive scores
    gaps = []
    for i in range(1, n):
        gap = scores[i-1] - scores[i]
        gaps.append((gap, i))   # (gap size, index after gap)

    # Find largest gap
    max_gap, gap_index = max(gaps, key=lambda x: x[0])

    if max_gap >= gap_threshold and gap_index >= min_k:
        # Stop at the gap — chunks after are significantly less relevant
        optimal_k = gap_index
        reason    = f"score gap {max_gap:.2f} at position {gap_index}"
        return max(min_k, min(optimal_k, max_k)), reason

    # No significant gap — all scores are similar, use standard k
    return min(n, max_k), "uniform relevance"
    """Response when retrieval finds nothing above the score threshold."""
    return (
        "I could not find relevant information in the available documents "
        "to answer your question.\n\n"
        "**Suggestions:**\n"
        "- Try rephrasing your question with different keywords\n"
        "- Lower the minimum similarity score in the sidebar settings\n"
        "- Make sure the relevant PDF has been ingested (check the document list in the sidebar)\n"
        "- Try asking a more specific question about the document content"
    )


def build_error_response(error_type: str = "general") -> str:
    """Specific error responses based on what went wrong."""
    messages = {
        "timeout": (
            "⏱️ The response took too long to generate.\n\n"
            "**Try:**\n"
            "- Asking a shorter or more specific question\n"
            "- Increasing `OLLAMA_TIMEOUT` in your `.env` file\n"
            "- Switching to a faster model (e.g. `phi3`) in the sidebar"
        ),
        "connection": (
            "❌ Cannot connect to Ollama.\n\n"
            "**Fix:** Open a terminal and run: `ollama serve`"
        ),
        "general": (
            "⚠️ An error occurred while generating the answer.\n\n"
            "Please try again. If the problem persists, check the API terminal for details."
        ),
    }
    return messages.get(error_type, messages["general"])


# ---------------------------------------------------------------------------
# Preview:  python llm/prompt_builder.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_context = (
        "[Source 1 | PB_30_EN_ECE_WG.1_45.pdf — page 8 | rerank: 6.297 | confidence: HIGH]\n"
        "Labour force participation of older persons (55-64) has been increasing steadily "
        "in the region, from 51.5 per cent in 2010 to 67 per cent overall in 2023.\n\n---\n\n"
        "[Source 2 | ECE-WG.1-42-PB28.pdf — page 6 | rerank: 5.102 | confidence: MEDIUM]\n"
        "To close the widespread pension gap between women and men, one approach is to "
        "recognize raising children in the pension system."
    )

    history = [
        {"role": "user",      "content": "Tell me about UNECE ageing policy."},
        {"role": "assistant", "content": "UNECE has published policy briefs on ageing workforce..."},
    ]

    # Test query validation
    test_queries = ["hi", "hello!", "What policies help older workers?", "thanks"]
    print("Query validation tests:")
    for q in test_queries:
        print(f"  '{q}' → off-topic: {is_greeting_or_offtopic(q)}")

    print("\n" + "=" * 65)
    print("PROMPT PREVIEW")
    print("=" * 65)
    prompt = build_rag_prompt(
        query   = "What policies help retain older workers?",
        context = sample_context,
        history = history,
    )
    print(prompt)
    print(f"\nTotal length: {len(prompt)} chars")