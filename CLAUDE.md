# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

## Running

```bash
uvicorn main:app --reload
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## Architecture

FastAPI backend (POC) for the EquipmentGram platform. Accepts equipment component images and returns structured defect analysis via Google Gemini Vision.

**Module responsibilities:**

- [main.py](main.py) — FastAPI app, route handlers, startup (Gemini model init, thread pool). Single and batch inspection endpoints.
- [core.py](core.py) — Prompt construction, Gemini response parsing, image conversion (`_open_and_convert`), scoring helpers (`severity_to_component_score`, `section_risk_from_score`), and the `inspect_one` coroutine used by the batch endpoint.
- [models.py](models.py) — Pydantic models: `DefectResponse` (internal), `InspectionResult` (single inspect), `ComponentBatchResult` + `BatchInspectionResult` (batch), `EquipmentInfo`.
- [equipment.py](equipment.py) — Static data: `EQUIPMENT_HIERARCHY` (manufacturers/models) and `COMPACT_EXCAVATOR_SECTIONS` (9 sections, ~85 components). `validate_hierarchy()` soft-validates inputs — failures log a warning but don't block the request.

**Request flow:**

1. `POST /inspect` or `POST /inspect/batch` receives multipart form data (equipment metadata + images)
2. `validate_hierarchy()` checks inputs against the static hierarchy (warning-only, non-blocking)
3. Image bytes decoded to PIL via `asyncio.to_thread(_open_and_convert, ...)` (thread pool, max 5 workers)
4. `create_inspection_prompt()` builds a structured prompt with equipment context and few-shot examples
5. `gemini_model.generate_content_async()` sends image + prompt to Gemini
6. `parse_gemini_response()` extracts JSON from Gemini's text, derives `condition`/`risk_level` from severity score
7. Batch: `asyncio.gather()` fans out concurrent `inspect_one()` calls; section score = mean of component scores (`100 - severity`)

**Scoring scale:**
- Severity 0–20 → Good / Low risk; 21–60 → Fair / Moderate; 61–100 → Defect / High
- `component_score = 100 - severity`; section score ≥85 → Low risk, ≥60 → Moderate, <60 → High

**Intentional POC limitations:**
- No persistent storage — inspection history resets on restart
- CORS is open (`*`)
- Only "Compact Excavator" is supported for section/component lookups
- Gemini model name is hardcoded in [main.py:94](main.py#L94)
