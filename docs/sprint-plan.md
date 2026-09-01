# Sprint Plan

## Sprint 1 — Problem Framing & System Design (Weeks 1–4)

### Goal
Define problem, architecture, and project plan

### Tasks
- Finalize use case and user scenarios
- Select dataset (UNECE + scope definition)
- Design full RAG architecture (diagram + module interfaces)
- Decide tech stack (LLM, embedding, DB, tools)
- Define evaluation strategy (metrics + benchmark plan)
- Assign team roles
- Set up GitHub + documentation structure

### Deliverables
- Architecture diagram
- Project plan (sprints + roles)
- Evaluation design draft

---

## Sprint 2 — Data Pipeline & Baseline Retrieval (Weeks 5–8)

### Goal
Build ingestion + basic retrieval

### Tasks
- PDF parsing + cleaning
- Chunking strategy implementation
- Metadata design (doc, section, page)
- Dense retrieval (embedding + FAISS)
- Basic query → top-k retrieval

### Deliverables
- Working ingestion pipeline
- Baseline retrieval module
- Initial dataset ready

---

## Sprint 3 — End-to-End RAG System (Weeks 9–12)

### Goal
First working chatbot

### Tasks
- Integrate LLM (API-based)
- Prompt design (grounded QA + citation)
- Build full pipeline: query → answer
- Simple interface (CLI or minimal UI)

### Deliverables
- Working RAG prototype
- Answer + citation output
- Demo-ready baseline system

---

## Sprint 4 — Evaluation & Baseline Results (Weeks 13–16)

### Goal
Make system measurable

### Tasks
- Create benchmark question set (50–100 Qs)
- Define ground truth answers
- Implement evaluation metrics:
  - Recall@k
  - answer correctness
  - citation accuracy
- Run baseline evaluation

### Deliverables
- Evaluation dataset
- Baseline metrics report
- Identified system weaknesses

---

## Sprint 5 — Improvements & Optimization (Weeks 17–20)

### Goal
Show measurable improvements

### Tasks
- Add hybrid retrieval (BM25 + dense + RRF)
- Add reranker (cross-encoder)
- Compare:
  - baseline vs hybrid vs rerank
- Optimize latency (limit top-k, rerank size)

### Deliverables
- Improved system version
- Comparative evaluation results
- Ablation-style analysis

---

## Sprint 6 — Finalization (Demo + Paper) (Weeks 21–24)

### Goal
Deliver complete project

### Tasks
- Build final demo (UI + clear flow)
- Prepare presentation (architecture + results)
- Write manuscript:
  - problem
  - method
  - experiments
  - results
- Final documentation (GitHub + reproducibility)

### Deliverables
- Final demo system
- Slides + presentation
- Written report / paper
- Clean, documented repository