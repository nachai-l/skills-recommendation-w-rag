# Skills Recommendation via Hybrid RAG

A deterministic, production-oriented Skill Recommendation system built using:

- **Hybrid Retrieval (FAISS cosine similarity + BM25 lexical)**
- **FAISS (MVP backend)**
- **LLM reasoning with strict Pydantic v2 validation** (generation stage implemented)
- **Optional Judge gating** (PASS required; judge output ignored once passed)
- **Modular pipeline architecture (Batch + Online)**
- **Artifact-first traceability**
- **Cloud Run ready design** (FastAPI + uvicorn verified locally)
- **Simple Web UI** (optional / planned)

This project extends the **Universal LLM Batch Generation Framework** into a real-time hybrid RAG system.

---

## 1) System Overview

The system recommends ranked skills based on a free-text query.

Final recommendation fields (API payload / Pipeline 5 output):

- `query`
- `analysis_summary`
- `recommended_skills[]` objects, each containing:
  - `skill_id`
  - `skill_name`
  - `source`
  - `relevance_score` (0–1)
  - `reasoning`
  - `evidence` (snippets copied from retrieved context)
  - `skill_text` (skill description)
  - `Foundational_Criteria`
  - `Intermediate_Criteria`
  - `Advanced_Criteria`

The architecture is designed for:

- Determinism (stable tie-break ordering)
- Traceability (manifest + aligned artifacts + cache_id)
- Schema safety (Pydantic v2 + `extra="forbid"`)
- Backend replaceability (FAISS → Vertex Vector Search later)
- Production deployment pattern (FastAPI + Cloud Run friendly)
- Minimal serving path: online pipelines only (no batch pipelines in serving)

---

## 2) Current Status (2026-02-25)
Implemented and verified:
- **Pipeline 0** — Schema Ensure (`schema/llm_schema.py`)
- **Pipeline 1** — Schema Text Ensure (`schema/llm_schema.txt`)
- **Pipeline 2** — Batch Ingest + Index Build (FAISS + BM25 artifacts)
- **Pipeline 3a** — Online Vector Search (FAISS)
- **Pipeline 3b** — Online BM25 Search (local BM25)
- **Pipeline 4a** — Hybrid merge + context builder 
- **Pipeline 4b** — LLM generation + strict Pydantic validation + retries + cache
- **Pipeline 4c** — LLM judge validation + retries + cache 
  - optimization: reuse 4b payload to avoid re-running generation 
- **Pipeline 5** — API payload builder (enrich LLM output with skill meta + criteria)
  - optimization: pass p4b payload into p4c to avoid duplicate 4b 
  - script saves API payload JSON to `artifacts/outputs/` 
- **FastAPI service layer** 
  - `POST /v1/recommend-skills` (returns Pipeline 5 output)
  - `GET /healthz`
  - `uvicorn + curl` verified locally 

Unit tests + real execution smoke scripts:

- Tests: **228 passed** (latest)
- Real execution scripts verified:
  - P0, P1, P2, P3a, P3b, P4a, P4b, P4c, P5

---

## 3) Architecture & Pipelines

### Pipeline 0 — Schema Ensure (Python) 

Ensures `schema/llm_schema.py` exists and is hardened:

- Pydantic v2
- `ConfigDict(extra="forbid")`
- Markdown fences removed
- Strict export control (`__all__`)
- AST safety checks (reject disallowed imports / eval)

Artifacts:

- `schema/llm_schema.py`
- archived snapshots under `archived/`

---

### Pipeline 1 — Schema Text Ensure 

Ensures `schema/llm_schema.txt` exists:

- Clean injection-ready schema text
- Stable formatting for prompt injection

Artifacts:

- `schema/llm_schema.txt`
- archived snapshots under `archived/`

---

### Pipeline 2 — Batch Ingest + Index Build 

Input:

- CSV skill taxonomy / corpus (`raw_data/skill_taxonomy.csv`)

Process:

- Build deterministic documents
- Generate embeddings using `gemini-embedding-001`
- Build **FAISS IndexFlatIP** (cosine similarity via normalized vectors)
- Build BM25 corpus JSONL + stats + manifest
- Persist artifacts under `artifacts/index_store/`

Outputs (actual paths from config):

- `artifacts/index_store/faiss.index`
- `artifacts/index_store/faiss_meta.jsonl`
- `artifacts/index_store/bm25_corpus.jsonl`
- `artifacts/index_store/bm25_stats.json`
- `artifacts/index_store/manifest.json`

Run verified:

- `num_docs=38120`
- `dim=3072`

#### BM25 corpus shaping (Option A) 

BM25 uses a **compact searchable text** for lexical search:

- `title + truncated skill_text`
- excludes scaffolding and criteria blocks **from the searchable text**

This improves BM25 relevance and performance.

#### Important revision: BM25 corpus retains criteria fields 

To ensure downstream context + payload enrichment always has criteria available, BM25 corpus rows include:

- `Foundational_Criteria`
- `Intermediate_Criteria`
- `Advanced_Criteria`

Final BM25 row schema includes:

- `{ "id", "title", "text", "source", "Foundational_Criteria", "Intermediate_Criteria", "Advanced_Criteria" }`

Note:

- Only `text` is indexed by BM25 (lexical scoring)
- Criteria fields are carried as metadata for context/payload later

---

### Pipeline 3a — Vector Search (Online) 

Input:

- user query string

Process:

- Embed query online (same embedding model)
- FAISS cosine similarity search
- Map internal indices to meta rows (`faiss_meta.jsonl`)
- Stable tie-break ordering:
  - score desc → skill_id asc → internal_idx asc

Output (payload shape):

```json
{
  "query": "...",
  "top_k": 20,
  "results": [
    {"skill_id": "...", "skill_name": "...", "score_vector": 0.83, "internal_idx": 123}
  ],
  "debug": {}
}
````

---

### Pipeline 3b — BM25 Search (Online) 

Input:

* user query string

Process:

* Build BM25Index in-process (cached per worker)
* Tokenize query and run BM25 scoring
* Stable tie-break ordering:

  * score desc → skill_id asc → internal_idx asc

BM25 corpus mapping (online):

* `id_key="id"`
* `name_key="title"`
* `doc_key="text"`
* `source_key="source"`

Output (payload shape):

```json
{
  "query": "...",
  "top_k": 20,
  "results": [
    {"skill_id": "...", "skill_name": "...", "score_bm25": 12.34, "internal_idx": 456}
  ],
  "debug": {}
}
```

---

### Pipeline 4a — Hybrid Merge + Context Builder 

Hybrid scoring:

```text
hybrid_score = alpha * normalized_vector_score
             + (1 - alpha) * normalized_bm25_score
```

Alpha configured in `configs/parameters.yaml`:

```yaml
rag:
  hybrid:
    alpha: 0.6
```

Stable tie-break rules enforced:

* hybrid desc → vector desc → bm25 desc → skill_id asc

Context rendering:

* Uses `configs/parameters.yaml` → `context.row_template`
* Enforces:

  * `truncate_field_chars`
  * `max_context_chars`
* Deterministic formatting (no dict-order dependency)

Output:

* merged `results`
* rendered `context`
* `debug` stats (counts, min/max, truncation flags)

---

### Pipeline 4b — LLM Generation + Strict Pydantic Validation 

Inputs:

* merged context (from 4a)
* `prompts/generation.yaml`
* `schema/llm_schema.txt` injected as `{llm_schema}`
* user query injected as `{query}`

Core behavior:

* Call 4a to build `{context}`
* Render prompt deterministically with safe placeholder replacement (brace-safe)
* Call Gemini via `google-genai`
* Extract first JSON from output (robust raw_decode)
* Validate with Pydantic model from `schema/llm_schema.py`
* Retry with corrective prefix if JSON/validation fails
* Cache validated payload under `artifacts/cache/` (deterministic cache_id)

Output (payload shape):

```json
{
  "query": "...",
  "top_k": 20,
  "cache_id": "p4b__...",
  "llm_validated": {
    "analysis_summary": "...",
    "recommended_skills": [
      {"skill_name": "...", "relevance_score": 0.9, "reasoning": "...", "evidence": ["..."]}
    ]
  }
}
```

---

### Pipeline 4c — Judge Validation (Optional Gate) 

Purpose:

* Evaluate whether 4b output is:

  * schema compliant
  * grounded in context
  * evidence is copied from context
  * sorted and reasonable

Inputs:

* `output_json` = stringified `llm_validated`
* `{context}` from 4b/4a
* `{llm_schema}` (structure contract)
* `{query}`

Output:

```json
{
  "query": "...",
  "cache_id": "p4c__...",
  "judge_validated": {"verdict": "PASS", "score": 90, "reasons": ["..."]},
  "generation_cache_id": "p4b__..."
}
```

---

### Pipeline 5 — API Payload Builder 

Purpose:

* Produce final API response payload
* Enrich each recommended skill with meta fields + criteria

Inputs:

* 4b output (must include retrieval_results/context)
* optional 4c gating (require PASS)

Join strategy:

* Use `retrieval_results` from 4b (already includes meta)
* Join recommended skills to meta
* Ensure output includes:

  * skill_id + skill_text + criteria fields

`require_all_meta`:

* if True, fail if any recommended item cannot be enriched

Output:

```json
{
  "payload": {
    "query": "...",
    "analysis_summary": "...",
    "recommended_skills": [
      {
        "skill_id": "...",
        "skill_name": "...",
        "source": "...",
        "relevance_score": 0.9,
        "reasoning": "...",
        "evidence": ["..."],
        "skill_text": "...",
        "Foundational_Criteria": "...",
        "Intermediate_Criteria": "...",
        "Advanced_Criteria": "..."
      }
    ]
  },
  "meta": {
    "generation_cache_id": "p4b__..."
  },
  "debug": { ... }   // optional
}
```

Script support:

* `scripts/run_pipeline_5_force.py` clears `artifacts/outputs/` and saves payload JSON.

---

## 4) Repository Structure (Actual)

Current structure (key parts):

```
functions/
  batch/
    pipeline_0_schema_ensure.py
    pipeline_1_schema_txt_ensure.py
    pipeline_2_ingest_index.py
  core/
    api_payload.py
    bm25.py
    context_render.py
    hybrid_merge.py
    index_store.py
    ingestion.py
    llm_generate.py
    llm_judge.py
    vector_search.py
    ...
  online/
    pipeline_3a_vector_search.py
    pipeline_3b_bm25_search.py
    pipeline_4a_hybrid_context.py
    pipeline_4b_generate.py
    pipeline_4c_judge.py
    pipeline_5_api_payload.py
  llm/
    client.py
    prompts.py
    runner.py
  utils/
    config.py
    logging.py
    paths.py
    text_embeddings.py
    ...

schema/
  llm_schema.py
  llm_schema.txt

prompts/
  generation.yaml
  judge.yaml
  schema_auto_py_generation.yaml
  schema_auto_json_summarization.yaml

scripts/
  run_pipeline_0_force.py
  run_pipeline_1_force.py
  run_pipeline_2_force.py
  run_pipeline_3a_force.py
  run_pipeline_3b_force.py
  run_pipeline_4a_force.py
  run_pipeline_4b_force.py
  run_pipeline_4c_force.py
  run_pipeline_5_force.py
  clear_llm_cache.py
  clear_archived.py

artifacts/
  index_store/
  cache/
  outputs/        # pipeline 5 exports JSON here
  ...
```

FastAPI:

```
app/
  main.py
```

---

## 5) Installation

### Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```
---

## 6) Run Pipelines (Local)

### Ensure Schema

```bash
python scripts/run_pipeline_0_force.py
python scripts/run_pipeline_1_force.py
```

### Build Index (FAISS + BM25 artifacts)

```bash
python scripts/run_pipeline_2_force.py
```

### Online Retrieval Smoke Tests

```bash
python scripts/run_pipeline_3a_force.py
python scripts/run_pipeline_3b_force.py
```

### Hybrid + LLM + Judge + API Payload

```bash
python scripts/run_pipeline_4a_force.py
python scripts/run_pipeline_4b_force.py
python scripts/run_pipeline_4c_force.py
python scripts/run_pipeline_5_force.py
```

Pipeline 5 exports a JSON payload under:

* `artifacts/outputs/pipeline5_api_payload__<timestamp>.json`

---

## 7) Run FastAPI (Local)

Start server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/healthz
```

Recommend skills (Pipeline 5 endpoint):

```bash
curl -X POST "http://localhost:8000/v1/recommend-skills" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "data scientist",
    "top_k": 20,
    "debug": true,
    "require_judge_pass": true,
    "top_k_vector": 20,
    "top_k_bm25": 20,
    "require_all_meta": false
  }'
```

---

## 8) Testing

Run all tests:

```bash
pytest -q
```

Notes:

* Retrieval tests use deterministic stubs where possible.
* Real execution scripts call Google APIs (schema generation + embeddings + LLM) when configured.

---

## 9) Configuration

Edit:

```text
configs/parameters.yaml
```

Key settings:

* embeddings model and dimension
* retrieval `top_k_default`
* BM25 tokenizer params
* hybrid alpha
* context template + limits
* LLM model + retries
* cache settings

---

## 10) Artifact Policy (.gitignore)

Runtime artifacts are ignored:

* `artifacts/**` ignored except `.gitkeep`
* Embedding cache batches (`*.npy`) ignored
* `archived/**` ignored except `.gitkeep`

---

## 11) Deployment (Cloud Run Ready Pattern)

Target:

* Docker + Cloud Run

Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Production plan (typical):

1. Bake artifacts/index_store into image, or mount from storage
2. Provide `GEMINI_API_KEY` (and model config) via Secret Manager/env
3. Configure autoscaling / concurrency
4. (Optional) Preload FAISS + BM25 on startup to reduce cold start
5. (Optional) Replace FAISS with Vertex Vector Search later

---

## 12) Design Principles

* Thin online pipeline wrappers
* Deterministic ranking and stable tie-breaks
* JSON-only LLM outputs with strict schema validation
* Pydantic v2 `extra="forbid"`
* Safe prompt injection (brace-safe renderer)
* Artifact-first debugging (but do not commit artifacts)
* Config-driven behavior
* Retrieval backend abstraction

---

## 13) Summary

This repo now implements a complete, production-oriented skill recommendation flow:

* Batch indexing: FAISS + BM25 ✅
* Online retrieval: vector + BM25 ✅
* Hybrid merge + context render ✅
* LLM generation + strict validation + retries + cache ✅
* Optional judge gating ✅
* API payload enrichment including full criteria + skill_text + skill_id ✅
* FastAPI endpoint serving pipeline 5 ✅

Current milestone achieved:

* **End-to-end online Skill Recommendation API output (P0–P5 + FastAPI) fully working** ✅


