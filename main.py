"""
Heavy Equipment Defect Detection System - FastAPI Backend
Using Google Gemini Pro Vision

Usage:
1. Create .env file with: GEMINI_API_KEY=your_key_here
2. Run: uvicorn main:app --reload
3. Access API docs at: http://localhost:8000/docs
"""

import asyncio
import base64
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core import (
    create_inspection_prompt,
    inspect_one,
    parse_gemini_response,
    _open_and_convert,
    section_risk_from_score,
    severity_to_component_score,
)
from equipment import (
    COMPACT_EXCAVATOR_SECTIONS,
    EQUIPMENT_HIERARCHY,
    validate_hierarchy,
)
from models import (
    BatchInspectionResult,
    DefectResponse,
    EquipmentInfo,
    InspectionResult,
)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("defect_model")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="Heavy Equipment Defect Detection API",
    description="POC for detecting defects in heavy equipment using Gemini Pro Vision",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with a database in production)
inspections_db: list = []
db_lock = asyncio.Lock()

gemini_model = None

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global gemini_model
    max_workers = 5
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=max_workers))
    logger.info("Thread pool initialized with max_workers=%d", max_workers)
    try:
        gemini_model = genai.GenerativeModel("gemini-3-flash-preview")
        logger.info("Gemini model initialized successfully")
    except Exception as e:
        logger.critical("Failed to initialize Gemini model: %s", e)
        raise

# ---------------------------------------------------------------------------
# Utility routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "Heavy Equipment Defect Detection API - EquipmentGram",
        "version": "1.0.0",
        "equipment_coverage": "15 models across 3 manufacturers",
        "total_components": "~85 per equipment model",
        "endpoints": {
            "POST /inspect": "Analyze equipment component for defects",
            "POST /inspect/batch": "Inspect multiple components in a section",
            "GET /inspections": "Get all inspection history",
            "GET /inspections/{id}": "Get specific inspection",
            "GET /equipment-info": "Get complete equipment hierarchy",
            "GET /manufacturers": "Get list of manufacturers",
            "GET /models": "Get models by manufacturer",
            "GET /sections": "Get sections by equipment type",
            "GET /components": "Get components by section",
            "GET /health": "Health check",
        },
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    if gemini_model is None:
        return {
            "status": "unhealthy",
            "error": "Gemini model not initialized",
            "timestamp": datetime.now().isoformat(),
        }
    return {
        "status": "healthy",
        "gemini_model": "gemini-3-flash-preview",
        "equipment_types": len(EQUIPMENT_HIERARCHY),
        "total_sections": len(COMPACT_EXCAVATOR_SECTIONS),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/equipment-info", response_model=EquipmentInfo)
async def get_equipment_info():
    manufacturers_by_type = {
        eq: list(mfrs.keys()) for eq, mfrs in EQUIPMENT_HIERARCHY.items()
    }
    models_structure = {
        eq: {mfr: data["models"] for mfr, data in mfrs.items()}
        for eq, mfrs in EQUIPMENT_HIERARCHY.items()
    }
    return EquipmentInfo(
        equipment_types=list(EQUIPMENT_HIERARCHY.keys()),
        manufacturers=manufacturers_by_type,
        models=models_structure,
        sections={"Compact Excavator": list(COMPACT_EXCAVATOR_SECTIONS.keys())},
        components=COMPACT_EXCAVATOR_SECTIONS,
    )


@app.get("/manufacturers")
async def get_manufacturers(equipment_type: str = "Compact Excavator"):
    if equipment_type not in EQUIPMENT_HIERARCHY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid equipment type. Must be one of: {list(EQUIPMENT_HIERARCHY.keys())}",
        )
    return {
        "equipment_type": equipment_type,
        "manufacturers": list(EQUIPMENT_HIERARCHY[equipment_type].keys()),
    }


@app.get("/models")
async def get_models(equipment_type: str = "Compact Excavator", manufacturer: Optional[str] = None):
    if equipment_type not in EQUIPMENT_HIERARCHY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid equipment type. Must be one of: {list(EQUIPMENT_HIERARCHY.keys())}",
        )
    if manufacturer:
        if manufacturer not in EQUIPMENT_HIERARCHY[equipment_type]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid manufacturer. Must be one of: {list(EQUIPMENT_HIERARCHY[equipment_type].keys())}",
            )
        return {
            "equipment_type": equipment_type,
            "manufacturer": manufacturer,
            "models": EQUIPMENT_HIERARCHY[equipment_type][manufacturer]["models"],
        }
    return {
        "equipment_type": equipment_type,
        "models_by_manufacturer": {
            mfr: data["models"] for mfr, data in EQUIPMENT_HIERARCHY[equipment_type].items()
        },
    }


@app.get("/sections")
async def get_sections(equipment_type: str = "Compact Excavator"):
    if equipment_type != "Compact Excavator":
        raise HTTPException(
            status_code=400,
            detail=f"Sections not defined for equipment type: {equipment_type}",
        )
    return {
        "equipment_type": equipment_type,
        "sections": list(COMPACT_EXCAVATOR_SECTIONS.keys()),
        "total_sections": len(COMPACT_EXCAVATOR_SECTIONS),
    }


@app.get("/components")
async def get_components(section: Optional[str] = None):
    if section:
        if section not in COMPACT_EXCAVATOR_SECTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid section. Must be one of: {list(COMPACT_EXCAVATOR_SECTIONS.keys())}",
            )
        return {
            "section": section,
            "components": COMPACT_EXCAVATOR_SECTIONS[section],
            "total_components": len(COMPACT_EXCAVATOR_SECTIONS[section]),
        }
    total = sum(len(comps) for comps in COMPACT_EXCAVATOR_SECTIONS.values())
    return {"all_sections": COMPACT_EXCAVATOR_SECTIONS, "total_components": total}

# ---------------------------------------------------------------------------
# Inspection routes
# ---------------------------------------------------------------------------

@app.post("/inspect/batch", response_model=BatchInspectionResult)
async def batch_inspect_equipment(
    equipment_type: str = Form(...),
    manufacturer: str = Form(...),
    model: str = Form(...),
    section: str = Form(...),
    component_names: str = Form(..., description="Comma-separated list of component names"),
    images: List[UploadFile] = File(...),
):
    """
    Inspect multiple components in a section concurrently.
    Returns individual component scores and an aggregated section score.

    component_names: comma-separated e.g. "Boom Condition,Stick Condition"
    images: one image per component, in the same order as component_names
    """
    component_names_list = [n.strip() for n in component_names.split(",") if n.strip()]

    if len(component_names_list) != len(images):
        raise HTTPException(
            status_code=400,
            detail=f"component_names ({len(component_names_list)}) and images ({len(images)}) must have the same length",
        )

    logger.info(
        "Batch inspection request | %s > %s > %s > %s | components=%d",
        equipment_type, manufacturer, model, section, len(component_names_list),
    )

    for component in component_names_list:
        is_valid, error_msg = validate_hierarchy(equipment_type, manufacturer, model, section, component)
        if not is_valid:
            logger.warning("Batch hierarchy validation failed: %s", error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

    if gemini_model is None:
        raise HTTPException(status_code=503, detail="Gemini model not initialized.")

    image_bytes_list = []
    for img in images:
        if not img.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File '{img.filename}' is not an image")
        data = await img.read()
        if len(data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File '{img.filename}' exceeds 10MB limit")
        image_bytes_list.append(data)

    results = await asyncio.gather(*[
        inspect_one(gemini_model, equipment_type, manufacturer, model, section, name, data)
        for name, data in zip(component_names_list, image_bytes_list)
    ])

    scores = [r.component_score for r in results if r.success and r.component_score is not None]
    section_score = round(sum(scores) / len(scores), 2) if scores else None
    section_risk_level = section_risk_from_score(section_score) if section_score is not None else None

    logger.info(
        "Batch complete | section=%s succeeded=%d/%d section_score=%s section_risk=%s",
        section, len(scores), len(results), section_score, section_risk_level,
    )

    batch_id = f"BATCH-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(inspections_db) + 1}"
    batch_result = BatchInspectionResult(
        batch_id=batch_id,
        equipment_type=equipment_type,
        manufacturer=manufacturer,
        model=model,
        section=section,
        timestamp=datetime.now().isoformat(),
        components_inspected=len(results),
        components_succeeded=len(scores),
        component_results=list(results),
        section_score=section_score,
        section_risk_level=section_risk_level,
    )

    async with db_lock:
        inspections_db.append(batch_result.model_dump())

    return batch_result


@app.post("/inspect", response_model=InspectionResult)
async def inspect_equipment(
    equipment_type: str = Form(...),
    manufacturer: str = Form(...),
    model: str = Form(...),
    section: str = Form(...),
    component: str = Form(...),
    image: UploadFile = File(...),
):
    """Analyze a single equipment component image for defects."""
    logger.info(
        "Inspection request received | %s > %s > %s > %s > %s",
        equipment_type, manufacturer, model, section, component,
    )

    is_valid, error_msg = validate_hierarchy(equipment_type, manufacturer, model, section, component)
    if not is_valid:
        logger.warning("Hierarchy validation failed: %s", error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    if not image.content_type.startswith("image/"):
        logger.warning("Invalid file type uploaded: %s", image.content_type)
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        image_bytes = await image.read()
        image_size_kb = len(image_bytes) / 1024
        logger.info("Image received | filename=%s size=%.1fKB", image.filename, image_size_kb)

        if len(image_bytes) > 10 * 1024 * 1024:
            logger.warning("Image too large: %.1fKB", image_size_kb)
            raise HTTPException(status_code=400, detail="Image size must be less than 10MB")

        pil_image = await asyncio.to_thread(_open_and_convert, image_bytes)
        logger.info("Image processed | mode=%s size=%s", pil_image.mode, pil_image.size)

        prompt = create_inspection_prompt(equipment_type, manufacturer, model, section, component)

        if gemini_model is None:
            logger.error("Gemini model is not initialized")
            raise HTTPException(status_code=503, detail="Gemini model not initialized.")

        logger.info("Sending request to Gemini | component=%s", component)
        t0 = time.monotonic()
        response = await gemini_model.generate_content_async([prompt, pil_image])
        logger.info("Gemini response received | duration=%.2fs", time.monotonic() - t0)

        defect_result = parse_gemini_response(response.text)
        logger.info(
            "Inspection result | defect_present=%s condition=%s severity=%d risk_level=%s",
            defect_result.defect_present, defect_result.condition,
            defect_result.severity, defect_result.risk_level,
        )

        inspection_id = f"INS-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(inspections_db) + 1}"
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        inspection_result = InspectionResult(
            id=inspection_id,
            equipment_type=equipment_type,
            manufacturer=manufacturer,
            model=model,
            section=section,
            component=component,
            timestamp=datetime.now().isoformat(),
            defect_present=defect_result.defect_present,
            defect_type=defect_result.defect_type,
            severity=defect_result.severity,
            condition=defect_result.condition,
            risk_level=defect_result.risk_level,
            observations=defect_result.observations,
            recommended_action=defect_result.recommended_action,
            image_base64=image_base64[:100] + "..." if len(image_base64) > 100 else image_base64,
        )

        async with db_lock:
            inspections_db.append(inspection_result.model_dump())
        logger.info("Inspection stored | id=%s", inspection_id)

        return inspection_result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during inspection: %s", e)
        raise HTTPException(status_code=500, detail=f"Error processing inspection: {e}")


@app.get("/inspections")
async def get_inspections(
    limit: int = 50,
    equipment_type: Optional[str] = None,
    manufacturer: Optional[str] = None,
    section: Optional[str] = None,
    defect_only: bool = False,
):
    filtered = inspections_db.copy()
    if equipment_type:
        filtered = [i for i in filtered if i.get("equipment_type") == equipment_type]
    if manufacturer:
        filtered = [i for i in filtered if i.get("manufacturer") == manufacturer]
    if section:
        filtered = [i for i in filtered if i.get("section") == section]
    if defect_only:
        filtered = [i for i in filtered if i.get("defect_present")]
    return {
        "inspections": filtered[-limit:],
        "total": len(filtered),
        "filters_applied": {
            "equipment_type": equipment_type,
            "manufacturer": manufacturer,
            "section": section,
            "defect_only": defect_only,
        },
    }


@app.get("/inspections/{inspection_id}")
async def get_inspection(inspection_id: str):
    for inspection in inspections_db:
        if inspection.get("id") == inspection_id or inspection.get("batch_id") == inspection_id:
            return inspection
    raise HTTPException(status_code=404, detail=f"Inspection {inspection_id} not found")


@app.delete("/inspections/{inspection_id}")
async def delete_inspection(inspection_id: str):
    for i, inspection in enumerate(inspections_db):
        if inspection.get("id") == inspection_id or inspection.get("batch_id") == inspection_id:
            async with db_lock:
                deleted = inspections_db.pop(i)
            logger.info("Inspection deleted | id=%s", inspection_id)
            return {"message": "Inspection deleted successfully", "deleted_inspection": deleted}

    logger.warning("Delete failed — inspection not found | id=%s", inspection_id)
    raise HTTPException(status_code=404, detail=f"Inspection {inspection_id} not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
