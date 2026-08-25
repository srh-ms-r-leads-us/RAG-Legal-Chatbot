# CODE MAP — UNECE Policy RAG Chatbot
## Quick navigation guide for code review and professor presentation

---

## 🗂️ FILE OVERVIEW

```
rag-chatbot/
├── ingestion/
│   ├── config.py               All settings loaded from .env
│   └── ingestion_manager.py    PDF ingestion pipeline (all stages)
│
├── retrieval/
│   ├── retrieval_engine.py     Core search: hybrid + reranking
│   └── bm25_index_builder.py   Manual BM25 rebuild utility
│
├── api/
│   ├── main.py                 FastAPI server + all 5 endpoints
│   ├── models.py               Request/response schemas
│   ├── logger.py               Shared structured logger
│   └── feedback_store.py       Saves user feedback to .jsonl
│
├── llm/
│   ├── ollama_client.py        Connects to Ollama, streams tokens
│   └── prompt_builder.py       Builds RAG prompt with citation rules
│
├── ui/
│   ├── app.py                  Streamlit chat interface
│   └── pages/
│       ├── 1_Analytics.py      Feedback + evaluation dashboard
│       └── 2_Search_Debug.py   Live retrieval score inspector
│
└── evaluation/
    ├── evaluator.py            Automated benchmark pipeline
    └── benchmark_questions.json 15 test questions with expected answers
```

---

## 🔍 WHERE IS EACH KEY CONCEPT?

### CHUNKING STRATEGIES
```
File    : ingestion/ingestion_manager.py
Function: static_chunks()          line ~100  — fixed sliding window
Function: sentence_aware_chunks()  line ~125  — cuts at sentence boundaries
Function: structure_aware_chunks() line ~175  — detects document sections
Function: get_chunker()            line ~225  — selects strategy from .env
Setting : CHUNK_STRATEGY in .env              — static | sentence | structure
```

### TABLE EXTRACTION
```
File    : ingestion/ingestion_manager.py
Function: extract_tables_from_page()  line ~240
Library : pdfplumber
Output  : markdown table format stored as chunk with chunk_type="table"
```

### IMAGE UNDERSTANDING
```
File    : ingestion/ingestion_manager.py
Function: describe_image_with_llava()  line ~290
Function: check_llava_available()      line ~335
Model   : llava (via Ollama)
Setting : EXTRACT_IMAGES=true in .env
```

### VECTOR SEARCH (semantic)
```
File    : retrieval/retrieval_engine.py
Function: _vector_search()   line ~290
Database: ChromaDB (HNSW index, cosine similarity)
Model   : all-MiniLM-L6-v2 (384 dimensions)
Score   : cosine similarity, range 0.0 – 1.0
```

### BM25 KEYWORD SEARCH
```
File    : retrieval/retrieval_engine.py
Function: _bm25_search()     line ~320
Function: _tokenise()        line ~115  — lowercase, strip punctuation
Library : rank_bm25 (BM25Okapi, k1=1.5, b=0.75)
Score   : BM25 score, range 0 – 50+
```

### RRF FUSION (combining vector + BM25)
```
File    : retrieval/retrieval_engine.py
Function: _reciprocal_rank_fusion()   line ~155
Formula : 0.7/(60+vector_rank) + 0.3/(60+bm25_rank)
Setting : HYBRID_VECTOR_WEIGHT=0.7 in .env  (BM25 gets 0.3)
Setting : HYBRID_RRF_K=60 in .env
Paper   : Cormack, Clarke & Buettcher (SIGIR 2009)
Score   : RRF score, range 0.0 – 0.02
```

### CROSS-ENCODER RERANKING
```
File    : retrieval/retrieval_engine.py
Function: _rerank()          line ~345
Model   : cross-encoder/ms-marco-MiniLM-L-6-v2
Input   : (query + chunk) pairs — reads both together
Score   : logit score, range -5 to +10 (higher = more relevant)
Confidence labels:
  score >= 6.0  → HIGH
  score >= 3.0  → MEDIUM
  score <  3.0  → LOW
  score < -5.0  → filtered out
```

### DEDUPLICATION
```
File    : retrieval/retrieval_engine.py
Function: search()           line ~395  (Step 5 — deduplicate)
Logic   : keeps only the highest-scoring chunk per page
Reason  : prevents same page appearing multiple times in LLM context
```

### CONFIDENCE SCORES (for UI display)
```
File    : retrieval/retrieval_engine.py
Function: get_chunk_confidences()   line ~570
Returns : citation, label, similarity, bm25_score, rrf_score, rerank_score
Used by : ui/app.py and api/main.py /context endpoint
```

### URL CLEANING (removes footnote links from chunks)
```
File    : retrieval/retrieval_engine.py
Function: clean_chunk_text()   inside format_context()   line ~533
Removes : https://, www. links, numbered footnote references
Reason  : prevents LLM from citing URLs instead of filenames
```

### RAG PROMPT CONSTRUCTION
```
File    : llm/prompt_builder.py
Variable: SYSTEM_MESSAGE       line ~55   — LLM rules and citation format
Function: build_rag_prompt()   line ~115  — assembles complete prompt
Parts   : system + conversation history + context + question + instruction
Key rule: "Never write [Source 1] — use actual filename"
```

### QUERY VALIDATION (greetings + off-topic)
```
File    : llm/prompt_builder.py
Function: is_greeting_or_offtopic()   line ~30
Function: get_greeting_response()     line ~55
Patterns: hi, hello, thanks, bye, ok, single characters
Result  : friendly response without hitting retrieval pipeline
```

### OLLAMA STREAMING
```
File    : llm/ollama_client.py
Function: stream()     line ~110  — yields tokens one by one
Function: generate()   line ~70   — returns complete response as string
URL     : http://localhost:11434/api/generate
Setting : OLLAMA_MODEL in .env  (default: mistral:latest)
```

### FASTAPI ENDPOINTS
```
File    : api/main.py

GET  /api/v1/health       line ~200  — liveness check
GET  /api/v1/documents    line ~230  — list ingested PDFs
POST /api/v1/search       line ~265  — ranked chunks as JSON
POST /api/v1/context      line ~315  — formatted context string
POST /api/v1/feedback     line ~370  — save thumbs up/down
```

### REQUEST LOGGING
```
File     : api/main.py
Function : request_logging_middleware()   line ~165
Format   : [request_id] → METHOD /path  |  ← STATUS  TIME ms
Purpose  : trace any request through logs by its 8-char ID
```

### FEEDBACK STORAGE
```
File    : api/feedback_store.py
Function: save_feedback()        line ~45   — appends to feedback.jsonl
Function: load_all_feedback()    line ~80   — reads all entries
Function: get_feedback_summary() line ~100  — counts helpful/unhelpful
Format  : JSON Lines (.jsonl) — one entry per line
Location: data/processed/feedback.jsonl
```

### ANSWER CACHING
```
File    : ui/app.py
Function: _cache_key()    line ~105  — md5(query+top_k+min_score+model)
Function: get_cached()    line ~118
Function: set_cached()    line ~122  — max 50 entries, drops oldest
Result  : identical queries return instantly (⚡ toast shown)
```

### STREAMLIT CACHING (prevents page reloads)
```
File     : ui/app.py
Functions: api_health()          — cached 30 seconds
           api_documents()       — cached 60 seconds
           get_available_models() — cached 60 seconds
           check_ollama_available() — cached 10 seconds
Decorator: @st.cache_data(ttl=N, show_spinner=False)
Reason   : without caching, every slider/button triggers network calls
```

### MANIFEST (incremental ingestion)
```
File    : ingestion/ingestion_manager.py
Function: load_manifest()   line ~65
Function: save_manifest()   line ~72
Function: hash_file()       line ~77  — SHA256 hash detects any change
Location: data/processed/manifest.json
Purpose : only re-ingest PDFs that are new or changed
```

### BM25 INDEX REBUILD
```
File    : ingestion/ingestion_manager.py
Function: rebuild_bm25_index()   line ~380
Trigger : runs automatically after any new/updated/deleted PDF
Output  : data/processed/bm25_index.pkl
```

### EVALUATION PIPELINE
```
File    : evaluation/evaluator.py
Function: evaluate_question()       line ~145  — scores one question
Function: score_source_recall()     line ~100  — did we find right docs?
Function: score_keyword_coverage()  line ~115  — does answer have key terms?
Function: score_source_precision()  line ~130  — were retrieved docs relevant?
Function: generate_report()         line ~215  — markdown report generator
Run     : python evaluation/evaluator.py --skip-llm
Output  : evaluation/results/eval_report_TIMESTAMP.md
```

---

## 🎯 COMMON PROFESSOR QUESTIONS — QUICK ANSWERS

| Question | Where to look |
|----------|--------------|
| "How do you do vector search?" | `retrieval_engine.py` → `_vector_search()` |
| "How do you do keyword search?" | `retrieval_engine.py` → `_bm25_search()` |
| "How do you combine them?" | `retrieval_engine.py` → `_reciprocal_rank_fusion()` |
| "What is your chunking strategy?" | `ingestion_manager.py` → `structure_aware_chunks()` |
| "How do you handle tables?" | `ingestion_manager.py` → `extract_tables_from_page()` |
| "How do you handle images?" | `ingestion_manager.py` → `describe_image_with_llava()` |
| "How does reranking work?" | `retrieval_engine.py` → `_rerank()` |
| "How do you build the prompt?" | `prompt_builder.py` → `build_rag_prompt()` |
| "How do you prevent hallucination?" | `prompt_builder.py` → `SYSTEM_MESSAGE` |
| "How do you store feedback?" | `feedback_store.py` → `save_feedback()` |
| "How do you evaluate the system?" | `evaluator.py` → `evaluate_question()` |
| "Why not FAISS?" | ChromaDB = same speed at 1086 vectors + built-in metadata |
| "How do you prevent duplicates?" | `ingestion_manager.py` → `hash_file()` + manifest |
| "Where are the API endpoints?" | `api/main.py` lines 200-400 |

---

## 🚀 QUICK DEMO SCRIPT (for professor presentation)

```
1. Open browser:
   localhost:8501            ← chatbot UI
   localhost:8501/Search_Debug ← score inspector
   localhost:8080/docs       ← API swagger

2. Show ingestion:
   python ingestion/ingestion_manager.py
   → demonstrates incremental ingestion

3. Show retrieval scores:
   Go to Search Debug page
   Type: "What policies help retain older workers?"
   → shows all 4 scores per result

4. Show chatbot answering:
   Go to chatbot page
   Ask the same question
   → streaming answer with citations
   → click Sources to see score breakdown

5. Show API:
   localhost:8080/docs
   POST /search → Try it out → Execute
   → shows raw JSON with all scores

6. Show evaluation:
   python evaluation/evaluator.py --skip-llm
   → generates benchmark report
```

---

## ⚙️ ALL .ENV SETTINGS EXPLAINED

```
# Paths
PDF_DIR              → where your PDFs live
OUTPUT_DIR           → where JSON/pickle files are saved
CHROMA_DIR           → where ChromaDB stores vectors

# Chunking
CHUNK_SIZE           → words per chunk (default: 400)
CHUNK_OVERLAP        → shared words between chunks (default: 80)
CHUNK_STRATEGY       → static | sentence | structure
EXTRACT_TABLES       → true/false — extract tables with pdfplumber
EXTRACT_IMAGES       → true/false — describe images with LLaVA

# Embedding
EMBED_MODEL          → sentence-transformers model name
TRANSFORMERS_OFFLINE → true = use cached model, no internet

# ChromaDB
CHROMA_COLLECTION    → collection name
CHROMA_DISTANCE_METRIC → cosine | l2 | ip

# Retrieval
RETRIEVAL_TOP_K      → results returned to caller (default: 5)
RETRIEVAL_MIN_SCORE  → minimum similarity threshold (default: 0.45)
RETRIEVAL_EXCLUDE_PAGES → pages to skip (bibliography pages)

# Hybrid search
HYBRID_SEARCH_ENABLED → true/false
HYBRID_FETCH_K        → candidates from each search method (default: 20)
HYBRID_RRF_K          → RRF smoothing constant (default: 60)
HYBRID_VECTOR_WEIGHT  → weight for vector search (default: 0.7)

# Reranking
RERANKER_ENABLED     → true/false
RERANKER_MODEL       → cross-encoder model name
RERANKER_CANDIDATES  → how many chunks to rerank (default: 20)

# API
API_HOST             → 0.0.0.0 (accessible on network)
API_PORT             → 8080
API_PREFIX           → /api/v1
API_MAX_TOP_K        → maximum top_k allowed (default: 20)

# LLM
OLLAMA_BASE_URL      → http://localhost:11434
OLLAMA_MODEL         → mistral:latest
OLLAMA_TEMPERATURE   → 0.1 (low = factual)
OLLAMA_MAX_TOKENS    → 1024
OLLAMA_TIMEOUT       → 300 seconds

# UI
UI_API_BASE_URL      → http://localhost:8080/api/v1
UI_PAGE_TITLE        → UNECE Policy Chatbot
UI_MAX_HISTORY       → 50 messages
```