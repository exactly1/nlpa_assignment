# Architecture Design and Integration Flow

This document describes the end-to-end architecture of the Neural Machine Translation (NMT) solution, its components, deployment topology, configuration, and the interaction flows across the system.

## 1. Overview

- UI: Streamlit single-page app (`backend/app.py`).
- Translation Core: Model routing, transliteration, metrics, and light post-processing (`model/nmt_model.py`).
- Google Comparison: Optional comparison with Google Translate (`model/google_compare.py`).
- Batch Evaluation: Corpus-level and per-row scoring (`scripts/evaluate.py`).
- Data/Artifacts: CSV history and evaluation outputs under mounted volumes.
- Packaging: Docker image, docker-compose for local/dev; CI/CD to Docker Hub.

### Goals
- Real-time translation for English, Hindi, Marathi.
- Romanized input handling (ITRANS → Devanagari) for Eng→Indic.
- Quality metrics (BLEU/TER; METEOR when available) and Google comparison.
- Batch evaluation and downloadable results.

---

## 2. Component Architecture

- Streamlit UI (`backend/app.py`)
  - Controls: language selection, input text, optional reference.
  - Shows translation, metrics, model details, Google comparison, and evaluation UI.
  - Maintains history in `data/historical.csv`; triggers batch evaluator.

- Translation Core (`model/nmt_model.py`)
  - Pipeline selection: Helsinki-NLP Marian models via Hugging Face, or local fine-tuned models.
  - Pivot logic: falls back to English pivot for pairs lacking direct model.
  - Transliteration: Indic transliteration (ITRANS→Devanagari) for Eng→(Hindi|Marathi) romanized input.
  - Metrics: sacrebleu BLEU/TER; nltk METEOR when available.
  - Targeted Hindi post-processing: acronym letter-names (e.g., NLP→एनएलपी), progressive aspect fix for “I am loving …”.

- Google Compare (`model/google_compare.py`)
  - Uses `googletrans` (unofficial). Lazy import with clear failures; graceful messages on rate-limits/network issues.
  - Replaceable with official Google Cloud Translate for production reliability.

- Batch Evaluator (`scripts/evaluate.py`)
  - Inputs: `data/historical.csv` (source_lang, target_lang, src_text, ref_text).
  - Outputs: `out/eval_results.csv` with per-row metrics for our system and Google (when available), plus corpus BLEU/TER and average METEOR.

- Storage
  - History: `/app/data/historical.csv` (host `./data`).
  - Evaluation: `/app/out/eval_results.csv` (host `./out`).
  - HF cache: `/root/.cache/huggingface` (host `./.cache/huggingface`).

- Packaging & CI/CD
  - Dockerfile: copies `backend/`, `model/`, `scripts/`; installs `backend/requirements.txt`.
  - docker-compose: port 8501, volumes for cache/data/out.
  - GitHub Actions: builds and pushes `exactly1/nmt-app` on push to `main`.

### Logical Diagram (ASCII)

```
Browser
  │
  ▼
Streamlit UI (backend/app.py)
  ├─> Translation Core (model/nmt_model.py)
  │     ├─ HF Marian pipelines (cached)
  │     ├─ Transliteration (indic-transliteration)
  │     └─ Metrics (sacrebleu, NLTK)
  ├─> Google Compare (model/google_compare.py, googletrans)
  ├─> Append to /app/data/historical.csv
  └─> Batch Evaluator (scripts/evaluate.py)
          └─ Writes /app/out/eval_results.csv + corpus metrics
```

---

## 3. Deployment Architecture

- Container: Single Streamlit service on port 8501.
- Base image: `python:3.11-slim`.
- Volumes:
  - `./.cache/huggingface` ↔ `/root/.cache/huggingface`
  - `./data` ↔ `/app/data`
  - `./out` ↔ `/app/out`
- Image: `exactly1/nmt-app:latest` (local build via compose; CI pushes on `main`).

### docker-compose (summary)
- Service: `nmt-app`
- Ports: `8501:8501`
- Volumes: cache/data/out (as above)

---

## 4. Configuration

- Environment variables
  - `DATA_DIR` (default `/app/data`)
  - `OUT_DIR` (default `/app/out`)
  - `LOCAL_MODEL_ROOT` (default `models/local`)
  - `MT_MODEL_<src>_<tgt>` (e.g., `MT_MODEL_en_hi`) to override a pair with a specific HF path/local folder.

- Files
  - `backend/requirements.txt` drives all Python deps (transformers, sacrebleu, googletrans, etc.).

---

## 5. Interaction & Integration Flows

### 5.1 Translate (Interactive)
1) User selects source/target and enters text.
2) UI calls `translate_text(text, src, tgt, use_transliteration, reference)`.
3) Translation Core:
   - If Eng→Indic and input looks romanized (or forced): transliterate ITRANS→Devanagari.
   - Else: select pipeline (direct model or pivot via English).
   - If target Hindi: apply acronym and progressive aspect adjustments.
   - If reference provided: compute BLEU/TER/METEOR.
4) UI displays: translation, model name(s), metrics (if reference).
5) UI expander optionally calls Google compare (graceful if unavailable).
6) Row appended to `data/historical.csv`.

### 5.2 Batch Evaluation
1) User clicks Evaluate in sidebar or main page.
2) `scripts/evaluate.py` runs in-process:
   - Reads `data/historical.csv`.
   - Computes per-row metrics (ours + Google when available).
   - Computes corpus BLEU/TER and average METEOR.
   - Writes `out/eval_results.csv`.
3) UI renders corpus summary widgets and preview table; provides CSV download.

### 5.3 Fine-tuning & Model Selection
1) Train via `training/fine_tune_mt.py` to `models/local/<src>-<tgt>`.
2) At runtime, preference order: env override → local model → default Marian.

---

## 6. Data Model

- History CSV (`data/historical.csv`)
  - `source_lang`, `target_lang`, `src_text`, `ref_text`, `our_translation`

- Evaluation CSV (`out/eval_results.csv`)
  - History columns + `google_translation`, `bleu`, `ter`, `meteor`, `google_bleu`, `google_ter`, `google_meteor`

---

## 7. Quality Attributes

- Reliability
  - Pipeline caching reduces model re-load overhead.
  - HF cache volume avoids repeated downloads across restarts.
  - Google comparison is best-effort; failure does not break core flow.

- Performance
  - Pipeline memoization in-process.
  - Optional local fine-tuned models to reduce latency/boost quality.

- Scalability
  - Horizontal: replicate container behind a reverse proxy/load balancer.
  - Stateless UI; relies on mounted volumes for history/eval artifacts.

- Security & Privacy
  - No secrets by default.
  - If integrating Google Cloud Translate, pass API keys via env/secret store.
  - Data persisted in local volumes; handle per compliance needs.

---

## 8. CI/CD Pipeline

- GitHub Actions
  - Install deps, run pytest, build image.
  - Push `exactly1/nmt-app:latest` on push to `main` using `DOCKERHUB_TOKEN` secret.

- Local
  - `docker-compose up -d --build` to rebuild and run locally.

---

## 9. Operations

- Logs: Streamlit server logs in container stdout/stderr; use `docker logs`.
- Health: manual via opening `http://localhost:8501`.
- Storage: verify binds for `./data` and `./out` are working on host.

---

## 10. Extensibility

- Replace Marian with IndicTrans2 or mBART/mT5; set env override or place under `models/local`.
- Switch to official Google Cloud Translate for robust Google comparisons.
- Add REST API (FastAPI) alongside Streamlit for programmatic access.
- Extend Hindi post-processing or add rule-based grammatical adjustments.

---

## 11. Known Trade-offs

- googletrans is unofficial and may be rate-limited; errors are handled gracefully but results are not guaranteed.
- Transliteration relies on simple romanized hints and ITRANS; ambiguous inputs may still need manual disambiguation.
- Pivot translations (hi→mr) add latency and may compound errors compared to a direct fine-tuned pair.

