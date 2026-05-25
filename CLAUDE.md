# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI backend for heavy equipment defect detection. Images of equipment components are submitted to the Google Gemini Vision API, which returns a structured JSON assessment of defects, severity (0–100), and recommended actions. Currently supports Compact Excavators from Bobcat, Caterpillar, and John Deere.

## Running the Server

```bash
# Activate virtualenv first
source venv/bin/activate

# Start dev server
uvicorn main:app --reload

# API docs available at
http://localhost:8000/docs
```

Requires a `.env` file with `GEMINI_API_KEY=your_key_here`.

## Active Gemini Model

`gemini-2.5-flash-lite` — configured in `main.py` `startup_event()`. Other tested models are left in comments on that line. When switching models, also update the hardcoded string in the `/health` endpoint response.

## Architecture

All business logic is separated from the FastAPI routes:

- **`core.py`** — image processing (`_open_and_convert`), prompt construction (`create_inspection_prompt`), Gemini response parsing (`parse_gemini_response`), scoring helpers, and the `inspect_one` coroutine used by the batch endpoint.
- **`equipment.py`** — static data: `EQUIPMENT_HIERARCHY` (equipment type → manufacturer → models) and `COMPACT_EXCAVATOR_SECTIONS` (section → component list). Also contains `validate_hierarchy()`, which URL-decodes and validates all five path parameters. Validation failures are logged as warnings but do **not** block the request.
- **`models.py`** — Pydantic models for all request/response shapes.
- **`main.py`** — FastAPI app, CORS middleware, startup hook (initializes the global `gemini_model`), and the two inspection routes (`POST /inspect` and `POST /inspect/batch`).

## Key Data Flow

```
Image upload → validate_hierarchy (warn only) → _open_and_convert (PIL)
→ create_inspection_prompt → gemini_model.generate_content_async
→ parse_gemini_response → scoring helpers → Pydantic response model
```

Batch endpoint fans out via `asyncio.gather` over `inspect_one` coroutines (one per component/image pair).

## Scoring Logic (in `core.py`)

- Gemini returns `severity` (0–100).
- `derive_condition_and_risk(severity)` → `(condition, risk_level)`:
  - 0–20: Good / Low
  - 21–60: Fair / Moderate
  - 61–100: Defect / High
- `component_score = 100 - severity`
- `section_score` = mean of component scores; `section_risk_from_score` thresholds at 85 (Low) and 60 (Moderate).

## Prompt Design

`create_inspection_prompt` in `core.py` instructs Gemini to first verify the image is plausibly the stated component (sets `image_verified: false` for clearly unrelated images), then score defects. The prompt includes four few-shot JSON examples. Gemini must return **only** valid JSON — `parse_gemini_response` strips markdown fences before parsing.

## Extending Equipment Coverage

To add new equipment types, manufacturers, or models: edit `EQUIPMENT_HIERARCHY` and (if needed) add a new sections dict alongside `COMPACT_EXCAVATOR_SECTIONS` in `equipment.py`. The `/sections` and `/components` endpoints currently hardcode `"Compact Excavator"` — update those guards when adding new types.
