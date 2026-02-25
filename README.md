# Skills Recommendation via Hybrid RAG

A deterministic, production-oriented Skill Recommendation API built using:

* **Hybrid Retrieval (BM25 + Cosine Similarity)**
* **FAISS (MVP backend)**
* **LLM reasoning with strict Pydantic v2 validation**
* **Modular pipeline architecture (Batch + Online)**
* **FastAPI service layer**
* **Simple Web UI for testing & demos**

This project extends the **Universal LLM Batch Generation Framework** into a real-time hybrid RAG system.

---

# 1. System Overview

The system recommends ranked skills based on a free-text query.

Each recommendation includes:

* `skill_id`
* `skill_name`
* `score`
* `reason` (LLM explanation)
* `evidence` (retrieved snippets)
* `source`

The architecture is designed for:

* Determinism
* Traceability
* Schema safety
* Backend replaceability (FAISS → Vertex AI Vector Search)
* Production deployment (Cloud Run ready)

---

# 2. Architecture

## Pipeline 0 — Schema Ensure (Python)

Ensures `schema/llm_schema.py` exists and is hardened:

* Pydantic v2
* `ConfigDict(extra="forbid")`
* No markdown fences
* Strict export control

---

## Pipeline 1 — Schema Text Ensure

Ensures `schema/llm_schema.txt` exists:

* Clean injection-ready schema text
* Stable formatting
* Used for prompt injection

---

## Pipeline 2 — Batch Ingest + Index Build

Input:

* CSV skill taxonomy / corpus

Process:

* Generate embeddings
* Build FAISS index
* Build BM25 corpus artifacts
* Persist artifacts

Output:

* `artifacts/faiss_index`
* `artifacts/bm25_corpus`
* `artifacts/metadata_store`

---

## Pipeline 3a — Vector Search (Online)

* Embed query
* Cosine similarity search
* Return vector candidates

---

## Pipeline 3b — BM25 Search (Online)

* Tokenize query
* Lexical BM25 search
* Return lexical candidates

---

## Hybrid Merge

Scores normalized and merged:

```
hybrid_score = alpha * normalized_vector_score
             + (1 - alpha) * normalized_bm25_score
```

Alpha configured in `parameters.yaml`.

Stable tie-break rules enforced.

---

## Pipeline 4a — Context Builder + LLM Generation

Inputs:

* Retrieval results
* `generation.yaml`
* `llm_schema.txt`
* User query

Produces structured JSON (strict schema format).

---

## Pipeline 4b — Optional Judge

Optional LLM quality validation stage using `judge.yaml`.

---

## Pipeline 5 — Schema Validation

* Parse JSON
* Validate using `recommendation_schema.py`
* Enforce strict schema
* Post-process (dedupe, clamp, sort)

Returns final API response.

---

# 3. API Contract

## Endpoint

```
POST /v1/recommend-skills
```

## Request

```json
{
  "query": "data science for healthcare",
  "top_k": 10,
  "language": "en",
  "filters": {},
  "debug": false
}
```

## Response

```json
{
  "skills": [
    {
      "skill_id": "DS101",
      "skill_name": "Healthcare Data Analysis",
      "score": 0.87,
      "reason": "This skill aligns strongly with the user's query...",
      "evidence": ["Snippet 1", "Snippet 2"],
      "source": "taxonomy"
    }
  ],
  "meta": {
    "correlation_id": "abc123",
    "model": "gemini-2.5-flash",
    "prompt_version": "v1",
    "retrieval_version": "hybrid_v1",
    "timings": {}
  }
}
```

---

# 4. Repository Structure

```
batch/
  pipeline_2_ingest_and_index.py

online/
  pipeline_3a_vector_search.py
  pipeline_3b_bm25_search.py
  pipeline_4a_generation.py
  pipeline_4b_judge.py
  pipeline_5_validate.py

functions/
  rag/
  service/
  llm/
  utils/
  io/

schema/
  llm_schema.py
  llm_schema.txt
  recommendation_schema.py

app/
  main.py
  middleware.py

ui/
  index.html
  app.js
  style.css

artifacts/
configs/
tests/
```

---

# 5. Installation

## Create Virtual Environment

```
python -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```
pip install -r requirements.txt
```

---

# 6. Run Pipelines

## Ensure Schema

```
python scripts/run_pipeline_0_force.py
python scripts/run_pipeline_1_force.py
```

## Build Index

```
python batch/pipeline_2_ingest_and_index.py
```

---

# 7. Run API Locally

```
uvicorn app.main:app --reload
```

Open:

```
http://localhost:8000/docs
```

---

# 8. Web UI

The simple UI:

* Query input
* top_k control
* Debug toggle
* Ranked results display
* Optional retrieval debug view

Open `ui/index.html` (or served via FastAPI static route).

No frontend framework required.

---

# 9. Configuration

Edit:

```
configs/parameters.yaml
```

Example:

```yaml
rag:
  hybrid:
    alpha: 0.6
  backend: faiss

llm:
  model: gemini-2.5-flash
  temperature: 0.0
```

Secrets:

* Stored in `configs/credentials.yaml`
* For Cloud Run, use environment variables / Secret Manager

---

# 10. Testing

Run all tests:

```
pytest
```

Deterministic tests use:

* Stub retriever
* Stub LLM runner
* Strict schema validation

---

# 11. Deployment (Cloud Run Ready)

### Docker Start Command

```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Production Plan

1. Build Docker image
2. Deploy to Cloud Run
3. Inject secrets via Secret Manager
4. Replace FAISS with Vertex AI Vector Search backend
5. Keep same retrieval interface

---

# 12. Design Principles

* Deterministic hybrid retrieval
* Strict JSON-only LLM outputs
* Pydantic v2 `extra="forbid"`
* Artifact-first debugging
* Config-driven behavior
* Thin pipeline entrypoints
* Retrieval backend abstraction
* Production-ready orchestration

---

# 13. Future Extensions

* Vertex AI Vector Search backend
* Redis caching
* Structured skill categories / levels
* Multi-language ranking
* Evaluation metrics dashboard
* Judge-based automated quality scoring
* CI guardrails for retrieval + generation regression

---

# 14. Summary

This project implements a modular, deterministic Skill Recommendation system built on:

* Hybrid RAG
* FAISS (MVP)
* Strict Pydantic validation
* Reusable LLM orchestration
* API-first architecture
* Cloud-ready deployment model

---
