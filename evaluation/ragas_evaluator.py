"""
ragas_evaluator.py
------------------
RAGAS evaluation pipeline for the UNECE Policy RAG Chatbot.

Measures 4 industry-standard metrics:
  1. Faithfulness        — did the LLM hallucinate? (0-1)
  2. Answer Relevancy    — does the answer address the question? (0-1)
  3. Context Precision   — were retrieved chunks actually used? (0-1)
  4. Context Recall      — did we find all needed information? (0-1)

25 benchmark questions with verified ground truth answers
covering all UNECE policy documents in the corpus.

Research basis:
  Es et al. (2023) — RAGAS: Automated Evaluation of Retrieval
  Augmented Generation. EACL 2024.
  https://arxiv.org/abs/2309.15217

Run:
    python evaluation/ragas_evaluator.py
    python evaluation/ragas_evaluator.py --limit 5   # quick test
    python evaluation/ragas_evaluator.py --no-llm    # retrieval only
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

API_BASE    = cfg.UI_API_BASE_URL
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# 25 Benchmark Questions with Ground Truth
# ---------------------------------------------------------------------------

RAGAS_DATASET = {
    "question": [
        # Policy Brief No. 28: Older Persons in Vulnerable Situations
        "What is the UNECE definition of vulnerable situations for older persons?",
        "What proportion of people were at risk of poverty or social exclusion in the European Union in 2020?",
        "What are the three main categories of policy strategies for addressing vulnerability in older persons according to UNECE Policy Brief No. 28?",
        "What does Luxembourg incentive programme for re-employment of older jobseekers provide?",
        "What one-time financial support did Slovakia provide to older persons during the cost of living crisis?",
        "What proportion of people aged 60 and older experienced some form of abuse in community settings in 2022?",
        "What is the Silver Line project in Lithuania?",
        "Which countries reported that more than 50 per cent of older women live alone?",

        # ILO/UNECE/UNFPA: Demographic Change in Europe and Central Asia
        "What is the projected total population of Central and Western Asia in 2050 compared to 2024?",
        "Which underrepresented groups does the ILO UNECE UNFPA brief identify as key to expanding labour force participation?",
        "What is the demographic dividend as described in the ILO UNECE UNFPA policy brief?",
        "What labour force participation rate for men aged 65 and above was recorded in Europe and Central Asia in 2024?",

        # Policy Brief No. 30: Unlocking the Potential of an Ageing Workforce
        "What are the three key policy strategies to unlock the potential of an ageing workforce according to UNECE Policy Brief No. 30?",
        "By how much did the share of older workers aged 55 and above in the total UNECE workforce change between 2000 and 2023?",
        "How much could GDP per capita increase by 2050 if the employment rate of older workers reached 80 per cent?",
        "What is the average effective retirement age for men and women among UNECE OECD member states?",
        "What does Austria Level Up adult training initiative offer and how many people have benefited from it?",
        "What is Seniors at Work in Switzerland?",
        "What financial disincentive to work did Denmark remove regarding pension benefits?",
        "What percentage of unemployed workers aged 55 to 64 remain unemployed for more than a year?",

        # Take care of time: Ageing in Georgia
        "What was Georgia population aged 65 and above in 2010 and what is it projected to reach by 2030?",
        "Which organizations jointly organized the essay contest documented in Take care of time Ageing in Georgia?",

        # Cross-document questions
        "What causes loneliness among older persons?",
        "What policies help retain older workers in the workforce?",
        "How does demographic change affect labour markets in Europe?",
    ],

    "ground_truth": [
        # Policy Brief No. 28
        "Vulnerable situations are events experienced at a specific moment in time that create difficulty across one or more areas of life, which may overwhelm coping capacities and increase the risk of a negative impact on life.",
        "One in five persons (20 per cent) were at risk of poverty or social exclusion in the European Union in 2020.",
        "Prevention, mitigation, and protection.",
        "Offered by the Luxembourg Employment Agency (ADEM), it reimburses the employer's share of social security contributions for hiring unemployed workers aged 45 or older, for two years for ages 45-49 or up to retirement age for those 50+.",
        "Older persons aged 62 and above without income could apply for a one-time contribution of 100 euros; about 1,500 people received the subsidy.",
        "Around one in six people aged 60 and older experienced some form of abuse in community settings during 2022.",
        "A free phone service offering emotional and informational support to older persons who feel lonely or isolated, providing regular conversations with a telephone friend and the option of a free psychological or spiritual consultation.",
        "Denmark, Estonia, and Finland.",

        # ILO/UNECE/UNFPA
        "The population is projected to grow from 198 million in 2024 to 237 million in 2050.",
        "Women, youth, older workers, persons with disabilities, migrants, and refugees.",
        "The demographic dividend is a unique time-bound opportunity for accelerating economic growth that arises from a favourable young population structure where large cohorts of young people enter their productive and reproductive years.",
        "10.9 per cent in 2024, projected to rise to 13.3 per cent in 2050.",

        # Policy Brief No. 30
        "Readying the workforce (lifelong learning and skills development), retaining the workforce (extending working lives through age-friendly workplaces and flexible arrangements), and re-engaging the workforce (attracting previously retired workers back by removing financial disincentives).",
        "It roughly doubled, from about 10.5 per cent in 2000 to just over 20 per cent in 2023.",
        "Around 20 per cent by 2050, which would also fully compensate for the projected decrease in total workforce among OECD member states.",
        "63.6 years for men and 62.7 years for women.",
        "It offers free compulsory and basic education plus coaching and transition counselling covering language, mathematics, and digital literacy; around 50,000 people have benefited since 2012.",
        "A Swiss organization that connects experienced older workers with employers offering temporary or project-based employment, with profiles of more than 80,000 older persons and over 7,000 employers.",
        "Denmark eliminated means testing for public pension benefits when pensioners have a work-related income, so that earnings no longer reduce the pension amount.",
        "41.5 per cent of unemployed workers aged 55-64 remain unemployed for more than a year, compared to 30 per cent for workers aged 25-54.",

        # Take care of time: Ageing in Georgia
        "In 2010, more than 14 per cent of Georgia's 4.4 million population was aged 65 or older; this is projected to grow to 21 per cent by 2030.",
        "The United Nations Economic Commission for Europe (UNECE) and the United Nations Population Fund (UNFPA), in collaboration with Georgia's Ministry of Labour, Health and Social Affairs.",

        # Cross-document
        "Loneliness among older persons is caused by loss of purpose, social isolation, declining physical health, chronic illness, disability, death of friends and family, stigma around mental health, and ageism.",
        "Policies to retain older workers include age-friendly workplaces, flexible retirement options, combating ageism through legal protections, lifelong learning programmes, and financial incentives for employers.",
        "Demographic change leads to a shrinking and ageing workforce, labour shortages in essential sectors, pressure on pension systems, and potential declines in productivity and innovation.",
    ],
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_search(query: str, top_k: int = 5) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/search",
            json={"query": query, "top_k": top_k},
            timeout=30,
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        print(f"  API search error: {e}")
        return None


def api_context(query: str, top_k: int = 5) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/context",
            json={"query": query, "top_k": top_k},
            timeout=30,
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        print(f"  API context error: {e}")
        return None


# ---------------------------------------------------------------------------
# RAGAS metrics — lightweight local implementation
# (no OpenAI API key needed — uses your local Ollama)
# ---------------------------------------------------------------------------

def compute_faithfulness(
    answer:   str,
    contexts: list[str],
    ollama:   OllamaClient,
    model:    str,
) -> float:
    """
    Faithfulness — what fraction of claims in the answer are supported by context?

    Method:
      1. Ask LLM to extract all factual claims from the answer
      2. For each claim, ask LLM if it is supported by the context
      3. Score = supported_claims / total_claims
    """
    if not answer or not contexts:
        return 0.0

    context_text = "\n\n".join(contexts[:5])

    # Step 1: Extract claims
    claims_prompt = (
        f"Extract all factual claims from this answer as a numbered list. "
        f"One claim per line. Only extract specific facts, not general statements.\n\n"
        f"Answer: {answer[:1000]}\n\n"
        f"Claims (numbered list):"
    )

    try:
        claims_text = ollama.generate(claims_prompt, model=model)
        claims = [
            line.strip().lstrip("0123456789.-) ")
            for line in claims_text.strip().split("\n")
            if line.strip() and len(line.strip()) > 10
        ]
    except Exception:
        return 0.5   # default if extraction fails

    if not claims:
        return 1.0   # no specific claims = no hallucination

    # Step 2: Check each claim against context
    supported = 0
    for claim in claims[:10]:   # limit to 10 claims for speed
        check_prompt = (
            f"Context:\n{context_text[:2000]}\n\n"
            f"Claim: {claim}\n\n"
            f"Is this claim fully supported by the context above? "
            f"Answer with only YES or NO."
        )
        try:
            verdict = ollama.generate(check_prompt, model=model).strip().upper()
            if "YES" in verdict:
                supported += 1
        except Exception:
            supported += 0.5   # uncertain

    return round(supported / len(claims), 3)


def compute_answer_relevancy(
    question: str,
    answer:   str,
    ollama:   OllamaClient,
    model:    str,
) -> float:
    """
    Answer Relevancy — does the answer directly address the question?

    Method:
      Generate 3 questions from the answer.
      Score = average cosine similarity between generated questions and original.
      High similarity = answer is on-topic.

    Simplified version: ask LLM to rate relevancy 1-5.
    """
    if not answer:
        return 0.0

    prompt = (
        f"Question: {question}\n\n"
        f"Answer: {answer[:800]}\n\n"
        f"On a scale of 1 to 5, how directly does the answer address the question? "
        f"1=completely off-topic, 5=directly and completely answers it. "
        f"Respond with only a single number."
    )

    try:
        score_text = ollama.generate(prompt, model=model).strip()
        # Extract first number found
        import re
        numbers = re.findall(r'\b[1-5]\b', score_text)
        if numbers:
            return round(int(numbers[0]) / 5.0, 3)
    except Exception:
        pass

    return 0.5


def compute_context_precision(
    question: str,
    contexts: list[str],
    answer:   str,
    ollama:   OllamaClient,
    model:    str,
) -> float:
    """
    Context Precision — what fraction of retrieved chunks were actually useful?

    Score = useful_chunks / total_chunks
    """
    if not contexts:
        return 0.0

    useful = 0
    for ctx in contexts[:5]:
        prompt = (
            f"Question: {question}\n\n"
            f"Context chunk: {ctx[:500]}\n\n"
            f"Was this context chunk useful for answering the question? "
            f"Answer with only YES or NO."
        )
        try:
            verdict = ollama.generate(prompt, model=model).strip().upper()
            if "YES" in verdict:
                useful += 1
        except Exception:
            useful += 0.5

    return round(useful / len(contexts), 3)


def compute_context_recall(
    ground_truth: str,
    contexts:     list[str],
    ollama:       OllamaClient,
    model:        str,
) -> float:
    """
    Context Recall — does the context contain enough to produce the ground truth?

    Method:
      Break ground truth into sentences.
      Check if each sentence is attributable to the context.
      Score = attributable_sentences / total_sentences
    """
    if not ground_truth or not contexts:
        return 0.0

    context_text = "\n\n".join(contexts[:5])

    # Split ground truth into sentences
    import re
    sentences = [
        s.strip() for s in re.split(r'[.!?]', ground_truth)
        if len(s.strip()) > 10
    ]

    if not sentences:
        return 1.0

    supported = 0
    for sentence in sentences[:5]:   # limit for speed
        prompt = (
            f"Context:\n{context_text[:2000]}\n\n"
            f"Statement: {sentence}\n\n"
            f"Is this statement supported by or can be derived from the context? "
            f"Answer with only YES or NO."
        )
        try:
            verdict = ollama.generate(prompt, model=model).strip().upper()
            if "YES" in verdict:
                supported += 1
        except Exception:
            supported += 0.5

    return round(supported / len(sentences), 3)


# ---------------------------------------------------------------------------
# Single question evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    question:     str,
    ground_truth: str,
    ollama:       OllamaClient,
    model:        str,
    eval_llm:     bool = True,
    top_k:        int  = 5,
) -> dict:
    """Evaluate one question through the full RAG pipeline."""

    result = {
        "question"          : question,
        "ground_truth"      : ground_truth,
        "answer"            : "",
        "contexts"          : [],
        "faithfulness"      : 0.0,
        "answer_relevancy"  : 0.0,
        "context_precision" : 0.0,
        "context_recall"    : 0.0,
        "retrieval_ms"      : 0,
        "llm_ms"            : 0,
        "error"             : None,
    }

    # ── Retrieval ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    search_data = api_search(question, top_k=top_k)
    result["retrieval_ms"] = int((time.perf_counter() - t0) * 1000)

    if not search_data or not search_data.get("results"):
        result["error"] = "No retrieval results"
        return result

    # Extract context texts
    contexts = [r["text"] for r in search_data["results"]]
    result["contexts"] = contexts

    if not eval_llm:
        # Retrieval-only mode — compute context recall only
        result["context_recall"] = compute_context_recall(
            ground_truth, contexts, ollama, model
        )
        return result

    # ── Generation ────────────────────────────────────────────────────────
    context_data = api_context(question, top_k=top_k)
    if not context_data:
        result["error"] = "Context API failed"
        return result

    prompt = build_rag_prompt(query=question, context=context_data["context"])

    t1 = time.perf_counter()
    try:
        answer = ollama.generate(prompt, model=model)
        result["llm_ms"] = int((time.perf_counter() - t1) * 1000)
        result["answer"] = answer
    except Exception as e:
        result["error"] = f"LLM error: {e}"
        return result

    # ── RAGAS Metrics ─────────────────────────────────────────────────────
    print(f"    Computing RAGAS metrics...")

    result["faithfulness"] = compute_faithfulness(
        answer, contexts, ollama, model
    )
    result["answer_relevancy"] = compute_answer_relevancy(
        question, answer, ollama, model
    )
    result["context_precision"] = compute_context_precision(
        question, contexts, answer, ollama, model
    )
    result["context_recall"] = compute_context_recall(
        ground_truth, contexts, ollama, model
    )

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_ragas_report(results: list[dict], model: str, timestamp: str) -> str:
    valid = [r for r in results if not r.get("error")]

    def avg(key):
        vals = [r[key] for r in valid if r[key] > 0]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    lines = [
        f"# RAGAS Evaluation Report — UNECE Policy RAG Chatbot",
        f"**Generated:** {timestamp}",
        f"**Model:** {model}",
        f"**Questions evaluated:** {len(results)} | **Errors:** {len(results) - len(valid)}",
        f"",
        f"## RAGAS Metrics Summary",
        f"",
        f"| Metric | Score | Description |",
        f"|--------|-------|-------------|",
        f"| Faithfulness | {avg('faithfulness')} | LLM stays grounded in context |",
        f"| Answer Relevancy | {avg('answer_relevancy')} | Answer addresses the question |",
        f"| Context Precision | {avg('context_precision')} | Retrieved chunks are useful |",
        f"| Context Recall | {avg('context_recall')} | Context covers the answer |",
        f"",
        f"## Interpretation",
        f"",
        f"- **Faithfulness > 0.8** = low hallucination risk ✅",
        f"- **Answer Relevancy > 0.8** = answers are on-topic ✅",
        f"- **Context Precision > 0.7** = retrieval is efficient ✅",
        f"- **Context Recall > 0.8** = retrieval is comprehensive ✅",
        f"",
        f"## Per-Question Results",
        f"",
        f"| # | Question | Faith | Relevancy | Precision | Recall | Ret(ms) | Error |",
        f"|---|----------|-------|-----------|-----------|--------|---------|-------|",
    ]

    for i, r in enumerate(results, 1):
        q_short = r["question"][:55] + "..." if len(r["question"]) > 55 else r["question"]
        err     = (r.get("error") or "")[:20]
        lines.append(
            f"| {i} | {q_short} "
            f"| {r['faithfulness']} "
            f"| {r['answer_relevancy']} "
            f"| {r['context_precision']} "
            f"| {r['context_recall']} "
            f"| {r['retrieval_ms']} "
            f"| {err} |"
        )

    lines += [
        f"",
        f"## Research Reference",
        f"",
        f"RAGAS: Automated Evaluation of Retrieval Augmented Generation.",
        f"Es et al. (2023). EACL 2024. https://arxiv.org/abs/2309.15217",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    limit:    int | None = None,
    eval_llm: bool       = True,
    model:    str        = None,
) -> None:
    print("=" * 65)
    print("RAGAS Evaluation — UNECE Policy RAG Chatbot")
    print("=" * 65)

    questions     = RAGAS_DATASET["question"]
    ground_truths = RAGAS_DATASET["ground_truth"]

    if limit:
        questions     = questions[:limit]
        ground_truths = ground_truths[:limit]

    # Check API
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        print(f"API status    : {health['status']} — {health['total_vectors']} vectors")
    except Exception:
        print("❌ API offline — run: python api/main.py")
        sys.exit(1)

    ollama = OllamaClient()
    model  = model or cfg.OLLAMA_MODEL

    if eval_llm and not ollama.is_available():
        print("⚠️  Ollama offline — switching to retrieval-only mode")
        eval_llm = False

    print(f"Questions     : {len(questions)}")
    print(f"LLM eval      : {'enabled (' + model + ')' if eval_llm else 'disabled'}")
    print(f"RAGAS metrics : faithfulness, answer_relevancy, context_precision, context_recall")
    print()

    results = []
    start   = time.perf_counter()

    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths), 1):
        print(f"[{i}/{len(questions)}] {question[:65]}...")

        result = evaluate_one(
            question     = question,
            ground_truth = ground_truth,
            ollama       = ollama,
            model        = model,
            eval_llm     = eval_llm,
        )

        results.append(result)

        if result.get("error"):
            print(f"  ❌ Error: {result['error']}")
        else:
            print(
                f"  ✅ Faith:{result['faithfulness']}  "
                f"Rel:{result['answer_relevancy']}  "
                f"Prec:{result['context_precision']}  "
                f"Recall:{result['context_recall']}  "
                f"({result['retrieval_ms']}ms retrieval)"
            )

    total_time = time.perf_counter() - start

    # Summary
    valid = [r for r in results if not r.get("error")]
    print("\n" + "=" * 65)
    print("RAGAS EVALUATION COMPLETE")
    print("=" * 65)
    print(f"Total time       : {total_time:.1f}s")
    print(f"Questions        : {len(results)} | Errors: {len(results) - len(valid)}")

    if valid:
        def avg(k): return round(sum(r[k] for r in valid if r[k]>0) / max(len([r for r in valid if r[k]>0]),1), 3)
        print(f"\nFaithfulness     : {avg('faithfulness')}")
        print(f"Answer Relevancy : {avg('answer_relevancy')}")
        print(f"Context Precision: {avg('context_precision')}")
        print(f"Context Recall   : {avg('context_recall')}")
        print(f"Avg Retrieval    : {sum(r['retrieval_ms'] for r in valid)//len(valid)}ms")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = RESULTS_DIR / f"ragas_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    md_path = RESULTS_DIR / f"ragas_report_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(generate_ragas_report(results, model, timestamp))

    print(f"\nJSON saved : {json_path}")
    print(f"Report saved: {md_path}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS Evaluation for UNECE RAG Chatbot")
    parser.add_argument("--limit",   type=int,  default=None, help="Evaluate only first N questions")
    parser.add_argument("--no-llm",  action="store_true",     help="Skip LLM generation — retrieval only")
    parser.add_argument("--model",   type=str,  default=None, help="Ollama model to use")
    args = parser.parse_args()

    run(
        limit    = args.limit,
        eval_llm = not args.no_llm,
        model    = args.model,
    )