# Heavy Equipment Defect Detection API

A FastAPI backend that uses Google Gemini Pro Vision to automatically detect defects in heavy construction equipment from uploaded images. Built as a proof-of-concept for the EquipmentGram platform.

---

## What It Does

- Accepts an image of an equipment component alongside metadata (equipment type, manufacturer, model, section, component)
- Validates the input against a structured equipment hierarchy database
- Sends the image and context to Gemini Pro Vision with a tailored inspection prompt
- Returns a structured defect analysis including:
  - Whether a defect is present
  - Defect type (cracks, leaks, corrosion, wear, etc.)
  - Severity score (0–100)
  - Technical observations
  - Recommended action

---

## Supported Equipment

**Type:** Compact Excavator

| Manufacturer | Models |
|---|---|
| Bobcat | E10, E20, E26, E32, E35 |
| Caterpillar | 300.9D, 302CR, 303.5E2CR, 305E2CR, 307.5CR |
| John Deere | 17 G-Tier, 26 G-Tier, 35 G-Tier, 50 G-Tier, 60 G-Tier |

**Inspection sections:** General Appearance, Safety, Control Station, Engine, Drivetrain, Hydraulics, Boom Condition, Undercarriage, Specialty (~85 components total)

---

## Severity Scale

| Range | Meaning |
|---|---|
| 0–20 | No defect / cosmetic wear |
| 21–40 | Minor — monitor |
| 41–60 | Moderate — plan maintenance within 1–2 weeks |
| 61–80 | Significant — urgent, within 1–3 days |
| 81–100 | Critical — immediate action, safety risk |

---

## Requirements

- Python 3.9+
- Google Gemini API key

---

## Setup

**1. Clone the repo and navigate to the project folder**

```bash
cd defect_model
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Create a `.env` file with your Gemini API key**

```
GEMINI_API_KEY=your_api_key_here
```

---

## Running the App

```bash
uvicorn main:app --reload
```

The API will be available at:
- **Base URL:** `http://localhost:8000`
- **Interactive docs (Swagger UI):** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info and available endpoints |
| GET | `/health` | Health check and model status |
| GET | `/equipment-info` | Full equipment hierarchy |
| GET | `/manufacturers` | List manufacturers by equipment type |
| GET | `/models` | List models by manufacturer |
| GET | `/sections` | List inspection sections |
| GET | `/components` | List components by section |
| POST | `/inspect` | **Submit an image for defect analysis** |
| GET | `/inspections` | Inspection history with optional filters |
| GET | `/inspections/{id}` | Single inspection detail |
| DELETE | `/inspections/{id}` | Delete an inspection record |

---

## Example: Submit an Inspection

```bash
curl -X POST http://localhost:8000/inspect \
  -F "equipment_type=Compact Excavator" \
  -F "manufacturer=Bobcat" \
  -F "model=E20" \
  -F "section=Engine" \
  -F "component=Radiator" \
  -F "image=@/path/to/image.jpg"
```

**Example response:**

```json
{
  "id": "INS-20240422143012-1",
  "equipment_type": "Compact Excavator",
  "manufacturer": "Bobcat",
  "model": "E20",
  "section": "Engine",
  "component": "Radiator",
  "timestamp": "2024-04-22T14:30:12.000Z",
  "defect_present": true,
  "defect_type": "Corrosion",
  "severity": 55,
  "observations": "Visible rust buildup on upper radiator fins with minor coolant residue near inlet hose.",
  "recommended_action": "Schedule radiator flush and inspect hose connections within 1–2 weeks."
}
```

---

## Architecture Notes

- **Async:** All endpoints are fully async. PIL image processing runs in a bounded thread pool (`max_workers=5`). Gemini calls use `generate_content_async()`.
- **Concurrency:** Up to 5 images can be decoded simultaneously; additional requests queue automatically.
- **Storage:** In-memory only — inspection records reset on server restart. Intended to be replaced with a persistent database for production.
- **CORS:** Currently open (`*`) — restrict origins before deploying to production.
