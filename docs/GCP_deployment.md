# Skills Recommendation API — GCP Cloud Run Deployment

> Repo: `skills_recommendation_via_rag`  
> Target: Cloud Run (managed) in `asia-southeast1`  
> Service: `skills-recommendation-api`  
> Artifact Registry: `skills-reco-api`  
> Container: FastAPI (`uvicorn app.main:app`)  
> Critical runtime dependency: **`artifacts/index_store/*` must be inside the container image**

---

## 0) TL;DR

1) Ensure `artifacts/index_store/` is included in Cloud Build context (**.gcloudignore matters**)  
2) Build + push image to Artifact Registry  
3) Create Secret Manager secret `GEMINI_API_KEY`  
4) Deploy Cloud Run with:
   - service account
   - secret mount to env
   - env vars (`LOG_LEVEL`, `ENVIRONMENT`, etc.)
5) Test endpoint: `POST /v1/recommend-skills`

---

## 1) Prerequisites

### 1.1 Local requirements
- `gcloud` installed and authenticated
- `docker` installed and logged into Artifact Registry (or use `gcloud auth configure-docker`)
- Project access (roles to create/deploy Cloud Run, Artifact Registry, Secret Manager)

### 1.2 Repo requirements (must exist before deploy)
- `app/main.py` FastAPI entry
- `Dockerfile`
- Runtime retrieval artifacts generated:
  - `artifacts/index_store/faiss.index`
  - `artifacts/index_store/faiss_meta.jsonl`
  - `artifacts/index_store/bm25_corpus.jsonl`
  - `artifacts/index_store/bm25_stats.json`
  - `artifacts/index_store/manifest.json`

> Build these locally with:
```bash
python scripts/run_pipeline_2_force.py
````

---

## 2) Deployment Variables

Set once per terminal session:

```bash
PROJECT_ID="poc-piloturl-nonprod"
REGION="asia-southeast1"

AR_REPO="skills-reco-api"
SERVICE_NAME="skills-recommendation-api"
IMAGE_NAME="service"
TAG="test-$(date +%Y%m%d-%H%M%S)"

IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/$IMAGE_NAME:$TAG"
SA_NAME="skills-reco-api-sa"
```

---

## 3) Enable Required Services

```bash
gcloud config set project "$PROJECT_ID"

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com
```

---

## 4) Artifact Registry (Docker) Setup

Create the Docker repo if it doesn't exist:

```bash
gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" >/dev/null 2>&1 || \
gcloud artifacts repositories create "$AR_REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Docker images for Skills Recommendation API"
```

(Optional) Configure docker auth:

```bash
gcloud auth configure-docker "$REGION-docker.pkg.dev"
```

---

## 5) Service Account + IAM

### 5.1 Create service account (if missing)

```bash
gcloud iam service-accounts describe "$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com" >/dev/null 2>&1 || \
gcloud iam service-accounts create "$SA_NAME" \
  --display-name="Skills Recommendation API Service Account"
```

### 5.2 Add logging permission (project has conditional IAM bindings)

If your project has IAM conditions enabled, you must supply a condition.

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/logging.logWriter" \
  --condition='expression=true,title=allow-log-writer-skills-reco,description=Allow Cloud Run SA to write logs'
```

### 5.3 Allow secret access (Secret Manager)

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --condition='expression=true,title=allow-secret-access-skills-reco,description=Allow Cloud Run SA to access secrets'
```

Sanity check:

```bash
gcloud projects get-iam-policy "$PROJECT_ID" \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
  --format="table(bindings.role, bindings.condition.title)"
```

---

## 6) CRITICAL: Ensure `artifacts/index_store` is shipped to Cloud Build

Cloud Run failures like this:

```
FileNotFoundError: FAISS index not found: /app/artifacts/index_store/faiss.index
```

mean Cloud Build did not receive/build those files.

### 6.1 Use `.gcloudignore` (recommended)

Create a `.gcloudignore` file at repo root that:

* ignores heavy runtime caches
* BUT explicitly includes `artifacts/index_store/**`

Example:

```text
# --- common junk ---
.venv/
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
.git/

# --- ignore all artifacts by default ---
artifacts/**

# --- UNIGNORE PARENTS (critical) ---
!artifacts/
!artifacts/index_store/

# --- include index store files needed at runtime ---
!artifacts/index_store/**
!artifacts/index_store/faiss.index
!artifacts/index_store/faiss_meta.jsonl
!artifacts/index_store/bm25_corpus.jsonl
!artifacts/index_store/bm25_stats.json
!artifacts/index_store/manifest.json

# keep outputs ignored
artifacts/outputs/**
```

> Notes:
>
> * `.gcloudignore` affects **Cloud Build source upload**.
> * `.dockerignore` affects **docker build context** (if building locally).
> * For Cloud Build, `.gcloudignore` is the key file.

---

## 7) Build & Push (Cloud Build)

```bash
gcloud builds submit \
  --tag "$IMAGE_URI" \
  --project "$PROJECT_ID"
```

If you see:

* `Invalid value for [source]: Dockerfile required when specifying --tag`

It usually means:

* You ran from a directory without a `Dockerfile`, or
* Cloud Build context didn't include it.

Fix: ensure you run from repo root and `Dockerfile` exists.

---

## 8) Secret Manager: GEMINI_API_KEY

### 8.1 Create secret (first time only)

```bash
SECRET_NAME="GEMINI_API_KEY"
printf "%s" "$GEMINI_API_KEY" | gcloud secrets create "$SECRET_NAME" \
  --data-file=- \
  --replication-policy="automatic"
```

If it already exists:

```bash
printf "%s" "$GEMINI_API_KEY" | gcloud secrets versions add "GEMINI_API_KEY" --data-file=-
```

---

## 9) Deploy to Cloud Run

```bash
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE_URI" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --service-account "$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
  --set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest" \
  --set-env-vars "LOG_LEVEL=INFO,ENVIRONMENT=prod"
```

Get service URL:

```bash
SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)')"
echo "$SERVICE_URL"
```

---

## 10) Test the Endpoint

```bash
curl -X POST "$SERVICE_URL/v1/recommend-skills" \
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

Expected: HTTP 200 with JSON:

* `payload.query`
* `payload.analysis_summary`
* `payload.recommended_skills[]` including:

  * `skill_id`, `skill_name`, `skill_text`
  * criteria fields
  * `reasoning`, `evidence`
* `meta.generation_cache_id`
* optional `debug.join`

---

## 11) Troubleshooting

### 11.1 `Internal server error` + logs show missing FAISS index

**Symptom**

```
FileNotFoundError: FAISS index not found: /app/artifacts/index_store/faiss.index
```

**Cause**
Cloud Build context did not include `artifacts/index_store/*`

**Fix**

* Add `.gcloudignore` rules to unignore `artifacts/index_store/**`
* Rebuild + redeploy

---

### 11.2 `pip install -r requirements.txt` fails on `faiss-cpu==1.13.25`

**Symptom**

```
ERROR: No matching distribution found for faiss-cpu==1.13.25
```

**Cause**
Pinned version not available on PyPI for your environment.

**Fix**
Use an available version, e.g.:

* `faiss-cpu==1.13.2` (known published)
  or relax pin:
* `faiss-cpu>=1.13.2`

Then rebuild.

---

### 11.3 Secret already exists

**Symptom**

```
Secret ... already exists.
```

**Fix**
Add new version instead:

```bash
printf "%s" "$GEMINI_API_KEY" | gcloud secrets versions add "GEMINI_API_KEY" --data-file=-
```

---

### 11.4 IAM “specifying a condition is required”

**Cause**
Project IAM policy contains conditional bindings.

**Fix**
Always add `--condition='expression=true,title=...,description=...'` when binding roles.

---

## 12) Recommended Production Flags (optional)

* Increase request timeout if needed (LLM + retrieval can be slow on cold start):

  ```bash
  --timeout=300
  ```
* Set min instances to reduce cold start:

  ```bash
  --min-instances=1
  ```
* Set CPU always allocated for better latency:

  ```bash
  --cpu=2 --memory=2Gi --cpu-boost
  ```

---

## 13) Verification Checklist

* [ ] `artifacts/index_store/*` exists locally before build
* [ ] `.gcloudignore` includes `artifacts/index_store/**`
* [ ] Cloud Build completed with `SUCCESS`
* [ ] Cloud Run revision deployed with correct image tag
* [ ] Service account has:

  * `roles/logging.logWriter`
  * `roles/secretmanager.secretAccessor`
* [ ] `GEMINI_API_KEY` secret is set and mounted
* [ ] `POST /v1/recommend-skills` returns 200 with expected payload
