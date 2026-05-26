import asyncio
import base64
import io
import json
import logging
import time

from openai import AuthenticationError
from PIL import Image

from models import ComponentBatchResult, DefectResponse

OPENROUTER_PRIMARY_MODEL = "google/gemini-2.5-flash"
OPENROUTER_FALLBACK_MODEL = "openai/gpt-4o-mini"

logger = logging.getLogger("defect_model")


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def _open_and_convert(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    return img.convert("RGB") if img.mode != "RGB" else img


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def derive_condition_and_risk(severity: int) -> tuple[str, str]:
    if severity <= 20:
        return "Good", "Low"
    elif severity <= 60:
        return "Fair", "Moderate"
    else:
        return "Defect", "High"


def severity_to_component_score(severity: int) -> int:
    return 100 - severity


def section_risk_from_score(score: float) -> str:
    if score >= 85:
        return "Low"
    elif score >= 60:
        return "Moderate"
    else:
        return "High"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def create_inspection_prompt(
    equipment_type: str, manufacturer: str, model: str,
    section: str, component: str
) -> str:
    return f"""You are a technical vision inspector for heavy construction equipment, specifically trained in equipment inspection protocols.

EQUIPMENT DETAILS:
- Equipment Type: {equipment_type}
- Manufacturer: {manufacturer}
- Model: {model}
- Section: {section}
- Component: {component}

COMPONENT CONTEXT:
The image shows the "{component}" component, which is part of the "{section}" section of a {manufacturer} {model} {equipment_type}.

INSPECTION TASK:
Perform a detailed visual inspection of this specific component following industry standards for heavy equipment evaluation.

0. IMAGE VERIFICATION:
   Before inspecting, determine if this image plausibly shows the "{component}" from
   the "{section}" section of a {manufacturer} {model} {equipment_type}.

   Set image_verified: false if the image clearly shows something unrelated — a person,
   landscape, food, interior room, or machinery with no visible resemblance to the
   stated component.

   Set image_verified: true if the image plausibly shows the component or anything
   consistent with heavy equipment of this type.

   If image_verified is false:
   - Set verification_reason to one sentence explaining why
   - Set all other fields to null / false / 0
   - Do not attempt the inspection

1. DEFECT IDENTIFICATION:
   Analyze the image for any of the following defects:
   - Cracks, fractures, or structural damage
   - Rust, corrosion, or oxidation
   - Wear, erosion, or material loss
   - Deformation, bending, or misalignment
   - Leaks (hydraulic fluid, oil, fuel, coolant)
   - Missing parts or components
   - Loose or damaged fasteners
   - Excessive play or movement in joints/pins
   - Damaged seals or gaskets
   - Paint damage indicating stress or impact
   - Contamination or debris buildup

2. SEVERITY SCORING (0-100):
   - 0-20: No defect or minimal wear (normal operation, cosmetic only)
   - 21-40: Minor defect (monitor condition, schedule routine maintenance)
   - 41-60: Moderate defect (plan maintenance within 1-2 weeks)
   - 61-80: Significant defect (urgent maintenance required within 1-3 days)
   - 81-100: Critical defect (immediate action required, safety risk, stop operation)

3. OBSERVATIONS:
   One sentence summarising the key visual evidence found.

4. RECOMMENDED ACTION:
   One sentence stating the single most important action to take.

5. CONFIDENCE SCORING (0-100):
   Rate your confidence in the overall assessment based on image quality and component visibility:
   - 90–100: High — clear image, unambiguous result
   - 70–89: Moderate — some uncertainty (lighting, angle, partial view)
   - 50–69: Low — significant ambiguity; recommend re-inspection
   - 0–49: Very low — image quality or component visibility too poor to be certain

OUTPUT FORMAT:
Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{
  "image_verified": true or false,
  "verification_reason": "one sentence reason if not verified, else null",
  "defect_present": true or false,
  "defect_type": "specific defect description or null",
  "severity": 0-100,
  "confidence": 0-100,
  "observations": "one sentence of key visual evidence",
  "recommended_action": "one sentence stating the most important action"
}}

REFERENCE EXAMPLES:

Example 0 - Image Mismatch:
{{
  "image_verified": false,
  "verification_reason": "The image shows a grassy field, not a boom pin or any excavator component.",
  "defect_present": false,
  "defect_type": null,
  "severity": 0,
  "confidence": 0,
  "observations": null,
  "recommended_action": null
}}

Example 1 - Healthy Component (Hydraulic Cylinder):
{{
  "image_verified": true,
  "verification_reason": null,
  "defect_present": false,
  "defect_type": null,
  "severity": 5,
  "confidence": 92,
  "observations": "Cylinder rod surface is smooth with no scoring or leakage and only normal operational wear visible.",
  "recommended_action": "Continue normal operation and monitor at next scheduled inspection."
}}

Example 2 - Moderate Defect (Track Belt):
{{
  "image_verified": true,
  "verification_reason": null,
  "defect_present": true,
  "defect_type": "Excessive wear on track grouser with rust formation",
  "severity": 55,
  "confidence": 74,
  "observations": "Track grouser height reduced to ~60% of spec with rust forming on three consecutive worn grousers.",
  "recommended_action": "Schedule track replacement within 2 weeks and avoid high-stress applications until then."
}}

Example 3 - Critical Defect (Boom Pin):
{{
  "image_verified": true,
  "verification_reason": null,
  "defect_present": true,
  "defect_type": "Severe crack in boom pin with material displacement",
  "severity": 92,
  "confidence": 96,
  "observations": "An 8cm crack with rust staining and visible metal displacement indicates imminent failure of this load-bearing pin.",
  "recommended_action": "Remove equipment from service immediately and schedule emergency pin replacement with a certified technician."
}}

Now analyze the provided image of the {component} and provide your assessment in JSON format only:"""


# ---------------------------------------------------------------------------
# Gemini response parser
# ---------------------------------------------------------------------------

def parse_gemini_response(response_text: str) -> DefectResponse:
    try:
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        data = json.loads(response_text)
        image_verified = data.get("image_verified", True)
        verification_reason = data.get("verification_reason")

        if image_verified:
            severity = min(max(data.get("severity", 0), 0), 100)
            condition, risk_level = derive_condition_and_risk(severity)
            confidence = min(max(data.get("confidence") or 0, 0), 100)
        else:
            severity, condition, risk_level = 0, "Good", "Low"
            confidence = 0

        return DefectResponse(
            image_verified=image_verified,
            verification_reason=verification_reason,
            defect_present=data.get("defect_present", False),
            defect_type=data.get("defect_type"),
            severity=severity,
            confidence=confidence,
            condition=condition,
            risk_level=risk_level,
            observations=data.get("observations") or "No observations provided",
            recommended_action=data.get("recommended_action") or "No action specified",
        )
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Gemini response as JSON: %s | response=%s", e, response_text)
        raise ValueError(f"Failed to parse Gemini response as JSON: {e}\nResponse: {response_text}")
    except Exception as e:
        logger.error("Error processing Gemini response: %s", e)
        raise ValueError(f"Error processing Gemini response: {e}")


# ---------------------------------------------------------------------------
# OpenRouter + Gemini fallback
# ---------------------------------------------------------------------------

async def _call_openrouter(client, model: str, prompt: str, image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
    )
    return resp.choices[0].message.content


async def call_vision_model(
    openrouter_client, gemini_model,
    prompt: str, image_bytes: bytes, component: str,
) -> str:
    # 1. Try primary model on OpenRouter
    try:
        t0 = time.monotonic()
        text = await _call_openrouter(openrouter_client, OPENROUTER_PRIMARY_MODEL, prompt, image_bytes)
        logger.info("OpenRouter response | model=%s component=%s duration=%.2fs",
                    OPENROUTER_PRIMARY_MODEL, component, time.monotonic() - t0)
        return text
    except AuthenticationError:
        logger.warning("OpenRouter auth failed — skipping to Gemini SDK")
    except Exception as e:
        logger.warning("Primary model %s failed: %s — trying OpenRouter fallback", OPENROUTER_PRIMARY_MODEL, e)
        # 2. Try fallback model on OpenRouter
        try:
            t0 = time.monotonic()
            text = await _call_openrouter(openrouter_client, OPENROUTER_FALLBACK_MODEL, prompt, image_bytes)
            logger.info("OpenRouter response | model=%s component=%s duration=%.2fs",
                        OPENROUTER_FALLBACK_MODEL, component, time.monotonic() - t0)
            return text
        except AuthenticationError:
            logger.warning("OpenRouter auth failed on fallback — skipping to Gemini SDK")
        except Exception as e2:
            logger.warning("Fallback model %s failed: %s — using Gemini SDK", OPENROUTER_FALLBACK_MODEL, e2)

    # 3. Last resort: direct Gemini SDK
    logger.info("Using Gemini SDK fallback | component=%s", component)
    pil_image = await asyncio.to_thread(_open_and_convert, image_bytes)
    t0 = time.monotonic()
    response = await gemini_model.generate_content_async([prompt, pil_image])
    logger.info("Gemini SDK response | component=%s duration=%.2fs", component, time.monotonic() - t0)
    return response.text


# ---------------------------------------------------------------------------
# Single-component inspection coroutine (used by batch endpoint)
# ---------------------------------------------------------------------------

async def inspect_one(
    openrouter_client, gemini_model,
    equipment_type: str, manufacturer: str, model: str,
    section: str, component: str, image_bytes: bytes,
) -> ComponentBatchResult:
    try:
        prompt = create_inspection_prompt(equipment_type, manufacturer, model, section, component)
        response_text = await call_vision_model(openrouter_client, gemini_model, prompt, image_bytes, component)
        defect = parse_gemini_response(response_text)

        if not defect.image_verified:
            logger.warning("Image mismatch | component=%s reason=%s", component, defect.verification_reason)
            return ComponentBatchResult(
                component=component,
                success=False,
                error=defect.verification_reason or "Image does not match the stated component.",
            )

        return ComponentBatchResult(
            component=component,
            success=True,
            defect_present=defect.defect_present,
            defect_type=defect.defect_type,
            severity=defect.severity,
            confidence=defect.confidence,
            condition=defect.condition,
            component_score=severity_to_component_score(defect.severity),
            risk_level=defect.risk_level,
            observations=defect.observations,
            recommended_action=defect.recommended_action,
        )
    except Exception as e:
        logger.warning("Component inspection failed | component=%s error=%s", component, e)
        return ComponentBatchResult(component=component, success=False, error=str(e))
