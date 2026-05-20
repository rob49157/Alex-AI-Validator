# Alexandria AI Validator

Python + FastAPI service that performs content validation for the Alexandria library. Called by the Node backend during uploads.

## Architecture

Two validation tiers:

| Tier | When | What |
|------|------|------|
| **Tier 1** — `POST /api/validate/upload` | Synchronous, blocks upload | PDF magic bytes, text extraction (pdfplumber + Tesseract OCR fallback), basic quality heuristics |
| **Tier 2** — `POST /api/validate/deep-analysis` | Async, after upload is stored | Semantic duplicate detection (sentence-transformers + ChromaDB), language detection, quality scoring, completeness estimation |

## Requirements

- Python 3.11+
- Tesseract OCR — `brew install tesseract` (macOS) or `apt install tesseract-ocr` (Linux)
- poppler (for PDF-to-image conversion) — `brew install poppler` or `apt install poppler-utils`

## Setup

```bash
# 1. Install Python 3.11+ if needed
brew install python@3.11          # macOS (requires Homebrew: brew.sh)

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — at minimum set a strong API_KEY

# 5. Start the server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Running tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `HOST` | `127.0.0.1` | Bind address |
| `API_KEY` | `change_me` | Shared secret with Node backend — **change this** |
| `TESSERACT_CMD` | _(empty)_ | Full path to tesseract binary if not on `$PATH` |
| `MAX_FILE_SIZE_MB` | `50` | Upload size limit |
| `MAX_PAGES_OCR` | `500` | Max pages to OCR per document |
| `JOB_TIMEOUT_SECONDS` | `120` | Per-job timeout for text extraction |
| `MAX_CONCURRENT_JOBS` | `5` | Max simultaneous Tier 2 analysis jobs |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Where ChromaDB stores embeddings |
| `SEMANTIC_SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity score that triggers a duplicate flag |

## API

All endpoints except `/health` require `Authorization: Bearer <API_KEY>`.

### `GET /health`
Returns `{"status": "ok"}`. No auth required. Used by Node backend for readiness checks.

### `POST /api/validate/upload`
Validates a PDF synchronously. Blocks until complete.

**Request:** `multipart/form-data` with field `file` (PDF bytes)

**Response:**
```json
{
  "valid": true,
  "flags": [],
  "extracted_text": "...",
  "page_count": 142
}
```

Possible flags: `possible_scan`, `ocr_failed`, `no_extractable_text`, `low_text_density`, `high_non_alpha_ratio`, `parse_timeout`

### `POST /api/validate/deep-analysis`
Starts an async deep analysis job. Returns immediately with a `job_id`.

**Request body:**
```json
{
  "arweave_hash": "abc123...",
  "mongo_upload_id": "64f1a2b3...",
  "extracted_text": "...",
  "page_count": 142
}
```

**Response:** `{"job_id": "uuid"}`

### `GET /api/validate/deep-analysis/{job_id}`
Polls job status.

**Response:**
```json
{
  "status": "complete",
  "results": {
    "semantic_duplicates": [],
    "quality_score": 0.82,
    "language": "en",
    "completeness": "full",
    "flags": []
  }
}
```

Statuses: `pending` → `processing` → `complete` | `failed`

## Integration with Node backend

The Node backend calls Tier 1 synchronously during the upload pipeline (Layer 4 validation), and fires Tier 2 asynchronously after the file is stored on Arweave.

```
POST /api/validate/upload           ← Node calls at upload time (blocking)
POST /api/validate/deep-analysis    ← Node calls after Arweave upload succeeds (non-blocking)
GET  /api/validate/deep-analysis/:id ← Node polls until complete or 14-day window expires
```

The API key must match `VALIDATION_SERVICE_URL` and the key set in the Node backend's `.env`.
