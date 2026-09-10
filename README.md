<div align="center">

# 🤖 UNECE Policy RAG Chatbot

### Production-grade Retrieval-Augmented Generation over UNECE Policy Documents

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?logo=streamlit)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-purple)](https://trychroma.com)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.1:8b-black?logo=ollama)](https://ollama.ai)
[![CUDA](https://img.shields.io/badge/CUDA-12.7-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A fully local, production-quality RAG system that answers natural language questions over 9 UNECE policy documents on ageing, demographics, and workforce policy — with full source citations, table rendering, image display, and adaptive retrieval.

[Demo](#demo) · [Architecture](#architecture) · [Installation](#installation) · [Usage](#usage) · [Evaluation](#evaluation) · [Research](#research-foundation)

</div>

---

## ✨ What Makes This Different

Most RAG tutorials use LangChain with a fixed top-k and a generic prompt. This system goes significantly further:

| Feature | Standard RAG | This System |
|---|---|---|
| Chunking | Fixed window | Structure-aware (detects UNECE section headings) |
| Table extraction | Plain text only | pdfplumber → markdown → interactive table in UI |
| Image understanding | Not supported | LLaVA vision model on GPU → image displayed in UI |
| Search | Vector only | Hybrid BM25 + vector → RRF fusion → cross-encoder rerank |
| Retrieval depth | Fixed k | CAR algorithm (Xu et al. 2025) — adaptive k per query |
| Ingestion | Re-process all | SHA256 incremental — only processes what changed |
| Privacy | Cloud APIs | 100% local — no data leaves the machine |
| Cost | Per-query API fees | Zero ongoing cost |
| Source recall | ~81% (LangChain) | **89%** (measured on benchmark) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                            │
│                                                                   │
│   9 PDFs → PyMuPDF (text) ──────────────────→ Structure-aware   │
│          → pdfplumber (tables → markdown) ──→ chunker            │
│          → LLaVA/Ollama (images → text) ────→                   │
│                                    ↓                             │
│             all-MiniLM-L6-v2 embedder                           │
│                   ↓                  ↓                           │
│             ChromaDB              BM25 index                     │
│           (1342 chunks)          (keyword)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    RETRIEVAL PIPELINE                            │
│                                                                   │
│   Query → Query rewriting (pronoun resolution)                   │
│         → CAR Stage 1: classify → initial k                     │
│         → Vector search (ChromaDB HNSW cosine) ──┐              │
│         → BM25 keyword search ───────────────────┤              │
│                                             RRF fusion (k=60)   │
│                                                   ↓              │
│                                        Cross-encoder rerank      │
│                                        (ms-marco-MiniLM-L-6)    │
│                                                   ↓              │
│                                     CAR Stage 2: score gap δ=1.5│
│                                                   ↓              │
│                                              Top k chunks        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    GENERATION PIPELINE                           │
│                                                                   │
│   Chunks → Format context (strip URLs, add citation headers)    │
│          → Build RAG prompt (RULES-based system message)        │
│          → llama3.1:8b via Ollama (streaming)                   │
│          → Post-process (strip fake citations)                   │
│          → Self-correction retry (detect hallucination)         │
│          → Streamed answer + images + tables in UI              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 RAGAS Evaluation Results

Evaluated on 25 domain-specific questions with verified ground truth across all 9 UNECE documents:

| Metric | Score | Interpretation |
|---|---|---|
| **Answer Relevancy** | **0.927** | ✅ Excellent — system answers the right question |
| **Context Recall** | **0.974** | ✅ Near perfect — retrieval finds all needed info |
| **Faithfulness** | **0.780** | ✅ Good — low hallucination rate |
| **Context Precision** | **0.325** | ⚠️ Low — CAR adaptive k addresses this |
| Source Recall | 89% | ✅ Custom benchmark — 8% above LangChain baseline |
| Avg Retrieval Latency | 3818ms | ✅ Fast retrieval |

> Context Precision being low **proves the value of CAR** — 2 of 3 retrieved chunks are unused without adaptive k selection.

---

## 📁 Project Structure

```
rag-chatbot/
│
├── ingestion/
│   ├── config.py                  # all settings (reads from .env)
│   └── ingestion_manager.py       # complete ingestion pipeline
│
├── retrieval/
│   ├── retrieval_engine.py        # hybrid search + reranking + CAR
│   └── bm25_index_builder.py      # BM25 index management
│
├── api/
│   ├── main.py                    # FastAPI server (5 endpoints)
│   ├── models.py                  # Pydantic request/response schemas
│   ├── logger.py                  # structured request logging
│   └── feedback_store.py          # user feedback persistence
│
├── llm/
│   ├── ollama_client.py           # Ollama connection + streaming
│   └── prompt_builder.py          # RAG prompt + CAR algorithm
│
├── ui/
│   ├── app.py                     # Streamlit chat interface
│   └── pages/
│       ├── 1_Analytics.py         # feedback dashboard
│       └── 2_Search_Debug.py      # live retrieval inspector
│
├── evaluation/
│   ├── evaluator.py               # custom metrics (recall/precision)
│   ├── ragas_evaluator.py         # RAGAS evaluation (25 questions)
│   ├── benchmark_questions.json   # 15 retrieval benchmark questions
│   └── results/                   # auto-generated reports
│
├── data/
│   ├── raw/                       # source PDFs (9 documents)
│   ├── chroma_db/                 # ChromaDB vector store
│   └── processed/
│       ├── manifest.json          # SHA256 hashes for incremental ingestion
│       ├── bm25_index.pkl         # serialised BM25 index
│       ├── feedback.jsonl         # user feedback log
│       └── images/                # extracted images (PNG, for UI display)
│
├── .env                           # configuration (never committed)
├── .env.example                   # configuration template
├── requirements.txt
├── README.md
└── CODE_MAP.md                    # navigation guide
```

---

## ⚡ Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- NVIDIA GPU recommended (CUDA 12.7) — CPU also works but image understanding is slow

### 1. Clone and set up environment

```powershell
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Pull required Ollama models

```powershell
# Main LLM (required)
ollama pull llama3.1:8b

# Vision model for image understanding (recommended)
ollama pull llava
```

### 3. Configure environment

```powershell
copy .env.example .env
```

Edit `.env`:

```env
# LLM
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT=300

# Retrieval
RETRIEVAL_TOP_K=5
RETRIEVAL_MIN_SCORE=0.25
HYBRID_SEARCH_ENABLED=True
HYBRID_VECTOR_WEIGHT=0.7
HYBRID_RRF_K=60

# Ingestion
CHUNK_STRATEGY=structure       # static | sentence | structure
EXTRACT_TABLES=True
EXTRACT_IMAGES=True            # requires LLaVA pulled in Ollama
```

### 4. Add your PDFs

Place PDF documents in `data/raw/`. The system is pre-configured for UNECE policy documents but works with any PDFs.

### 5. Run ingestion

```powershell
# First-time ingestion (processes all PDFs)
python ingestion/ingestion_manager.py

# Force re-ingest all (use when code changes)
python ingestion/ingestion_manager.py --force
```

Expected output:
```
Ingestion complete
  Text chunks   : 1118
  Table chunks  :  172
  Image chunks  :   52
  Total chunks  : 1342
```

---

## 🚀 Usage

Start the two servers in separate terminals:

```powershell
# Terminal 1 — API server
python api/main.py

# Terminal 2 — Streamlit UI
streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser.

### API Endpoints

The FastAPI server runs on `http://localhost:8080`. Interactive docs at `http://localhost:8080/docs`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | Pipeline status and vector count |
| GET | `/api/v1/documents` | List all ingested documents |
| POST | `/api/v1/search` | Retrieve chunks with all 4 scores |
| POST | `/api/v1/context` | Formatted context string for prompting |
| POST | `/api/v1/feedback` | Save user rating |

**Example search request:**

```python
import requests

response = requests.post(
    "http://localhost:8080/api/v1/search",
    json={
        "query": "What policies help retain older workers?",
        "top_k": 5,
        "min_score": 0.25
    }
)

for chunk in response.json()["results"]:
    print(f"[{chunk['rerank_score']:.2f}] {chunk['citation']}")
    print(chunk["text"][:200])
    print()
```

---

## 🔬 How the Retrieval Works

### Hybrid Search + RRF

Every query is sent to both search systems simultaneously:

```
Vector search:  "older workers training" → finds semantically similar chunks
BM25 search:    "older workers training" → finds chunks with exact keywords

RRF fusion:     score(chunk) = 0.7/(60 + vector_rank) + 0.3/(60 + bm25_rank)
                Chunks ranked highly in BOTH systems float to the top
```

### CAR Adaptive k (Xu et al., Coinbase 2025)

Instead of always retrieving k=5 chunks, CAR selects the optimal number per query:

```
Stage 1 — Query classification:
  "What is the Level Up initiative?"     → factual     → k=3
  "What does Figure 3 show?"            → visual      → k=4
  "Compare pension systems in Europe"    → comparative → k=7
  (anything else)                        → general     → k=5

Stage 2 — Score gap analysis:
  Fetch k+2 candidates, examine rerank scores:
  [8.2, 7.5, 7.0, 5.5, 2.1]   gap=3.4 at position 4  → use 4
  [7.5, 7.3, 7.1, 6.9, 6.7]   no gap > 1.5           → use all 5

  UI shows: 🧠 k=3 (factual, score gap 4.1 at position 3)
```

### Confidence Scoring

Each retrieved chunk is labelled with confidence based on cross-encoder rerank score:

| Rerank Score | Confidence | Label |
|---|---|---|
| > 6.0 | High | 🟢 |
| 3.0 – 6.0 | Medium | 🟡 |
| < 3.0 | Low | 🔴 |
| < -5.0 | Filtered | (removed) |

---

## 📈 Running Evaluation

### Custom retrieval metrics (fast, ~2 min)

```powershell
python evaluation/evaluator.py --skip-llm
```

Measures source recall, source precision, keyword coverage, and latency across 15 benchmark questions.

### RAGAS evaluation (requires LLM, ~70 min)

```powershell
# Quick test — first 5 questions
python evaluation/ragas_evaluator.py --limit 5 --model llama3.1:8b

# Full 25-question evaluation
python evaluation/ragas_evaluator.py --model llama3.1:8b
```

Measures faithfulness, answer relevancy, context precision, and context recall with verified ground truth answers. Reports saved to `evaluation/results/`.

---

## 🧪 Testing Retrieval Quality

Use the **Search Debug** page in Streamlit (`localhost:8501/Search_Debug`) to run any query and inspect all four retrieval scores without LLM generation:

| Score | What It Measures |
|---|---|
| 🔵 Vector Similarity | Semantic meaning match (cosine, 0–1) |
| 🟡 BM25 Score | Keyword frequency match (0–50+) |
| 🟣 RRF Score | Combined hybrid fusion (0–0.02) |
| 🟢 Rerank Score | Cross-encoder final relevance (−5 to +10) |

---

## 🔧 Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:8b` | LLM model for generation |
| `OLLAMA_TIMEOUT` | `300` | Seconds before LLM timeout |
| `RETRIEVAL_TOP_K` | `5` | Default k (CAR adjusts this dynamically) |
| `RETRIEVAL_MIN_SCORE` | `0.25` | Minimum similarity to include chunk |
| `HYBRID_SEARCH_ENABLED` | `True` | Enable BM25 + vector fusion |
| `HYBRID_VECTOR_WEIGHT` | `0.7` | Vector weight in RRF (BM25 = 1 - this) |
| `HYBRID_RRF_K` | `60` | RRF smoothing constant (Cormack 2009) |
| `HYBRID_FETCH_K` | `20` | Candidates fetched before reranking |
| `RERANKER_ENABLED` | `True` | Enable cross-encoder reranking |
| `CHUNK_STRATEGY` | `structure` | `static` / `sentence` / `structure` |
| `CHUNK_SIZE` | `400` | Target words per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap words between chunks |
| `EXTRACT_TABLES` | `True` | Extract tables via pdfplumber |
| `EXTRACT_IMAGES` | `True` | Extract images via LLaVA |

---

## 📚 Research Foundation

Every architectural decision is backed by peer-reviewed research:

| Technique | Paper | Year |
|---|---|---|
| Reciprocal Rank Fusion (k=60) | Cormack, Clarke & Buettcher — SIGIR | 2009 |
| BM25 Okapi | Robertson & Zaragoza — FnTIR | 2009 |
| Cross-encoder reranking | ms-marco-MiniLM training | 2020 |
| RAGAS evaluation framework | Es et al. — EACL | 2024 |
| CAR adaptive retrieval | Xu et al. — Coinbase | Oct 2025 |
| StrictCitations prompting | Zhu et al. | 2026 |

---

## 🗂️ Documents Supported

The system is pre-configured for these 9 UNECE policy documents:

| Document | Topic | Chunks |
|---|---|---|
| `ECE-WG.1-42-PB28.pdf` | Older persons in vulnerable situations | 37 |
| `ECE_PB29_EN.pdf` | Mental health and older persons | 51 |
| `PB_30_EN_ECE_WG.1_45.pdf` | Unlocking the potential of ageing workforce | 42 |
| `ECE_WG.1_43_web_0.pdf` | UNECE Regional Implementation Strategy | 300 |
| `ECE_WG.1_41_WEB.pdf` | MIPAA+20 regional review | 276 |
| `ILO_UNECE_UNFPA_Demographic_Change.pdf` | Demographic change in Europe and Central Asia | 11 |
| `WSSD2-regional-social-development-challenges.pdf` | Social development challenges | 227 |
| `Analysis_of_the_impact_of_imported_used_vehicles.pdf` | Vehicle road safety | 310 |
| `Take_care_of_time_-_Ageing_in_Georgia.pdf` | Ageing policy in Georgia | 36 |

---

## 🖥️ Tech Stack

```
PDF Processing:    PyMuPDF · pdfplumber · LLaVA
Embeddings:        sentence-transformers/all-MiniLM-L6-v2
Vector Store:      ChromaDB (HNSW index)
Keyword Search:    rank-bm25 (BM25Okapi, k1=1.5, b=0.75)
Reranker:          cross-encoder/ms-marco-MiniLM-L-6-v2
LLM:               llama3.1:8b via Ollama
API:               FastAPI + Uvicorn
UI:                Streamlit
Evaluation:        RAGAS (local, no API keys needed)
Hardware:          NVIDIA GTX 1650 Ti · CUDA 12.7
```

---

## 🤝 Team
  - Ritu, Ritu
  - Trac, Way
  - Yang, Zhixiao

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for UNECE Policy Research**

*Fully local · Zero API cost · Complete data privacy*

</div>
