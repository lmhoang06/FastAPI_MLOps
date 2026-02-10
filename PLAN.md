# Plan: FastAPI Multi-Model Server

## Goals
- FastAPI server with **one route per model** (models live in `models/`).
- Each route accepts **text** or **batch of texts**; returns model results.
- **Batch size** is configurable per model (default: 8).
- Optional: **ONNX export + quantization** via script in `tools/` for performance.

---

## Step 1: Environment (Windows)
- Create virtual environment: `python -m venv venv`
- Activate: `.\venv\Scripts\activate`
- Install: `pip install -r requirements.txt`
- (If `ensurepip` failed during venv creation, run: `.\venv\Scripts\python.exe -m ensurepip --upgrade` then `pip install -r requirements.txt`.)

---

## Step 2: Project Structure
```
FastAPI_MLOps/
  app/
    __init__.py
    main.py          # FastAPI app, include routers
    config.py        # Settings: models dir, default batch_size
    schemas/         # Pydantic request/response
    inference/       # Model loading & inference (HuggingFace / ONNX)
    routers/         # One route per model (dynamic from registry)
  models/            # Existing: YELP-Review_Classifier, etc.
  tools/
    export_onnx.py   # Export HuggingFace → ONNX + quantize
  requirements.txt
  venv/
```

---

## Step 3: Route Design
- **One route per model**: e.g. `POST /models/yelp-review-classifier/predict`
- **Request body** (Pydantic):
  - `text: str | None` — single text
  - `texts: list[str] | None` — batch of texts
  - (Exactly one of `text` or `texts` required.)
- **Per-model config**: `batch_size` (default 8) used when processing `texts` (chunked).
- **Response**: single result or list of results (labels/scores).

---

## Step 4: Model Registry
- Scan `models/` for subdirectories (each = one model).
- Config per model: path, batch_size (default 8), optional ONNX path.
- Lazy-load model on first request (or load at startup).
- Current model: `YELP-Review_Classifier` (DistilBERT sequence classification, 5 labels).

---

## Step 5: ONNX (Optional, in `tools/`)
- Script: load HuggingFace model from `models/<name>/`, export to ONNX.
- Apply quantization (e.g. dynamic or static) for smaller/faster inference.
- Output under model folder or e.g. `models/<name>/model.onnx`.
- Server can prefer ONNX if present.

---

## Execution Order
1. Venv + install (Step 1) — **first**.
2. Add `app/` with config, schemas, inference, routers (Step 2 + 3 + 4).
3. Add `tools/export_onnx.py` (Step 5).
