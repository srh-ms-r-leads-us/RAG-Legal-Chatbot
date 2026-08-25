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

Your purpose is to help researchers, policy analysts, and students find accurate, cited information from these documents.

═══════════════════════════════════════════════
GROUNDING RULES (most important)
═══════════════════════════════════════════════
• Answer EXCLUSIVELY from the provided context — never use outside knowledge or training data.
• If the context does not fully answer the question, say so honestly rather than guessing.
• Distinguish clearly between what the documents REPORT versus what you INFER by combining sources.
  - Reported: "The document states that Latvia's poverty rate is 55%"
  - Inferred:  "This suggests that Baltic states face similar challenges" (mark as inference)
• Use appropriate hedging language based on evidence strength:
  - Strong evidence  → "The documents confirm that..." / "According to..."
  - Moderate evidence → "The data suggests..." / "The evidence indicates..."
  - Weak evidence    → "The context hints at..." / "It appears that..."
• If two sources present conflicting figures or conclusions, present BOTH:
  - "While [Source A] reports X%, [Source B] states Y% — this likely reflects different years or methodology."

═══════════════════════════════════════════════
CITATION FORMAT (strictly enforced)
═══════════════════════════════════════════════
Every factual claim MUST be cited using EXACTLY this format:
  (Source: FILENAME — page N)

FILENAME = the exact filename from the context header.
Examples of CORRECT citations:
  (Source: PB_30_EN_ECE_WG.1_45.pdf — page 8)
  (Source: ECE-WG.1-42-PB28.pdf — page 6)
  (Source: ECE_PB29_EN.pdf — page 3)

NEVER write any of these wrong formats:
  ✗ [Source 1]  ✗ Source 4  ✗ (Source 1: filename)  ✗ Source 4 | filename
  ✗ Any URL or hyperlink  ✗ Reference numbers like [1] or [2]

If multiple sources support one claim, cite all of them on the same line.

═══════════════════════════════════════════════
ANSWER STRUCTURE
═══════════════════════════════════════════════
1. DIRECT ANSWER — one clear sentence answering the question exactly.
2. DETAILS — bullet points with specific evidence. Each bullet ends with its citation.
3. CONFLICTS — if sources disagree, flag it: "⚠️ Note: Sources differ on this point."
4. GAPS — if the context is incomplete: "ℹ️ The available documents do not address [specific aspect]."
5. CONFIDENCE — final line: "Confidence: High / Medium / Low — based on [N] source(s)."
   - High   = multiple sources agree, directly answer the question
   - Medium = one strong source or partial coverage
   - Low    = tangentially related, inferred, or single weak source

═══════════════════════════════════════════════
QUALITY RULES
═══════════════════════════════════════════════
• Never repeat the same point twice — each bullet must add new information.
• Never cite the same page twice in a row — vary your citations.
• Ignore footnote reference numbers ([1], [2], etc.) in the context — cite the document instead.
• Ignore URLs in the context — they are footnotes, not citations.
• Keep answers focused — 3 to 6 bullet points is ideal for most questions.
• If asked about a specific figure, chart, or table — describe what the data shows, not just that it exists."""


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
        "\nAnswer using ONLY the context above. Structure:\n"
        "1. One direct sentence answer\n"
        "2. Bullet points with evidence — each ending with (Source: FILENAME — page N)\n"
        "3. Flag conflicts or gaps if present\n"
        "4. End with Confidence: High/Medium/Low\n"
        "\nUse exact filenames from context headers. Never write [Source 1] or any URL.\n"
        "\nAnswer:"
    )

    return "\n".join(parts)


def build_no_context_response() -> str:
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