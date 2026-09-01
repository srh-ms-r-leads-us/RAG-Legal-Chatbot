import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import cfg

log = logging.getLogger("rag_api")

# Feedback is appended to this file — one JSON object per line
FEEDBACK_FILE = cfg.OUTPUT_DIR / "feedback.jsonl"


def save_feedback(
    query:      str,
    helpful:    bool,
    llm_answer: str  | None = None,
    citations:  list | None = None,
    comment:    str  | None = None,
) -> str:
    """
    Append one feedback entry to the feedback file.

    Args:
        query      : the original user query
        helpful    : True = thumbs up, False = thumbs down
        llm_answer : the answer shown to the user (optional)
        citations  : list of citation strings from the retrieval response
        comment    : free-text comment from the user (optional)

    Returns:
        feedback_id — unique ID assigned to this entry (UUID4 short form)
    """
    feedback_id = str(uuid.uuid4())

    entry = {
        "feedback_id": feedback_id,
        "timestamp"  : datetime.now(timezone.utc).isoformat(),
        "query"      : query,
        "helpful"    : helpful,
        "llm_answer" : llm_answer,
        "citations"  : citations or [],
        "comment"    : comment,
    }

    # Ensure the output directory exists
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Append one line — safe for concurrent writes in single-worker mode
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log.info(
        "Feedback saved [%s] query='%s...' helpful=%s",
        feedback_id, query[:40], helpful,
    )

    return feedback_id


def load_all_feedback() -> list[dict]:
    """
    Load all feedback entries from disk.
    Returns an empty list if the file does not exist yet.
    """
    if not FEEDBACK_FILE.exists():
        return []

    entries = []
    with open(FEEDBACK_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("Skipping malformed feedback line: %s", line[:80])

    return entries


def get_feedback_summary() -> dict:
    """
    Return a quick summary of all collected feedback.
    Useful for a future /feedback/summary endpoint.
    """
    entries   = load_all_feedback()
    total     = len(entries)
    helpful   = sum(1 for e in entries if e.get("helpful") is True)
    unhelpful = sum(1 for e in entries if e.get("helpful") is False)

    return {
        "total_feedback": total,
        "helpful"       : helpful,
        "unhelpful"     : unhelpful,
        "helpful_pct"   : round(helpful / total * 100, 1) if total else 0.0,
    }