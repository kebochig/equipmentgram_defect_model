"""
Heavy Equipment Defect Detection System - FastAPI Backend
Using Google Gemini Pro Vision

Installation:
pip install fastapi uvicorn python-multipart google-generativeai pillow pydantic python-dotenv

Usage:
1. Create .env file with: GEMINI_API_KEY=your_key_here
2. Run: uvicorn main:app --reload
3. Access API docs at: http://localhost:8000/docs
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import google.generativeai as genai
from PIL import Image
import io
import base64
import os
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI
app = FastAPI(
    title="Heavy Equipment Defect Detection API",
    description="POC for detecting defects in heavy equipment using Gemini Pro Vision",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Equipment Hierarchy Database
EQUIPMENT_HIERARCHY = {
    "Compact Excavator": {
        "Bobcat": {
            "models": ["E10", "E20", "E26", "E32", "E35"]
        },
        "Caterpillar": {
            "models": ["300.9D", "302CR", "303.5E2CR", "305E2CR", "307.5CR"]
        },
        "John Deere": {
            "models": ["17 G-Tier", "26 G-Tier", "35 G-Tier", "50 G-Tier", "60 G-Tier"]
        }
    }
}

# Sections and Components for Compact Excavator
COMPACT_EXCAVATOR_SECTIONS = {
    "General Appearance": [
        "Exterior Light",
        "Glass",
        "Hand Rails",
        "Paint",
        "Sheet Metal (Fiberglass Condition)"
    ],
    "Safety": [
        "Current Operator and Maintenance Manual",
        "Current Safety Manual",
        "Horn",
        "Safety Lock Out and Stop",
        "Seat Belt",
        "Swing Break",
        "Travel Alarm"
    ],
    "Control Station": [
        "Air Conditioner",
        "Dash Console",
        "Drivetrain Controls",
        "Engine Oil Pressure",
        "Gauges",
        "Heater",
        "Hour Meter",
        "Hydraulic Controls",
        "Limited Function Check (Control Station)",
        "Mirrors",
        "Seats ands Armrest",
        "Steering Controls",
        "Warning Light"
    ],
    "Engine": [
        "AC Compressor",
        "Blow By (Subjective Observation)",
        "Cooling System Leaks",
        "Emission Label",
        "Engine - Left Side",
        "Engine - Right Side",
        "Exhaust System",
        "Fuel Leaks",
        "Limited Function Check (Engine)",
        "Oil Leaks",
        "Radiator",
        "Starter"
    ],
    "Drivetrain": [
        "Left Drive Motor",
        "Left Final Drive",
        "Limited Function Check (Drivetrain)",
        "Right Motor Drive",
        "Right Final Drive"
    ],
    "Hydraulics": [
        "Auxiliary Hydraulic Plumbing",
        "Blade Lift Cylinder",
        "Boom Lift Cylinder(s)",
        "Boom Swing Cylinders",
        "Bucket Cylinder",
        "Hose (Hydraulics)",
        "Hydraulic Center Swivel",
        "Hydraulic Control Pattern Changer",
        "Hydraulic Tank",
        "Limited Function Check (Hydrualics)",
        "Pump (Hydraulics)",
        "Stick Cylinder",
        "Swing Motor",
        "Valves"
    ],
    "Boom Condition": [
        "Boom Condition",
        "Stick Condition",
        "Swing Tower Pivot",
        "Boom Base Pin and Bushings",
        "PIn and Bushings Boom to Stick",
        "Pin and Bushings Stick to Coupler",
        "Turntable Bearing",
        "Bottom Covers",
        "Limited Function Check"
    ],
    "Undercarriage": [
        "Left Front Idler",
        "Left Grouser Height",
        "Left Roller Frame",
        "Left Rubber Belt Drive Lugs",
        "Left Sprocket",
        "Left Track Belt Condition",
        "Left Track Rollers",
        "Left Track Tensioner",
        "Right Front Idler",
        "Right Grouser Height",
        "Right Roller Frame",
        "Right Rubber Belt Drive Lugs",
        "Right Sprocket",
        "Right Track Belt Condition",
        "Right Track Rollers",
        "Right Track Tensioners"
    ],
    "Speciality": [
        "Blade Condition",
        "Blade Cutting Edge Condition",
        "Cutting Edge Teeth Adapter",
        "Excavator Bucket Condition"
    ]
}

# Pydantic Models
class DefectResponse(BaseModel):
    defect_present: bool
    defect_type: Optional[str] = None
    severity: int = Field(ge=0, le=100)
    observations: str
    recommended_action: str

class InspectionResult(BaseModel):
    id: str
    equipment_type: str
    manufacturer: str
    model: str
    section: str
    component: str
    timestamp: str
    defect_present: bool
    defect_type: Optional[str]
    severity: int
    observations: str
    recommended_action: str
    image_base64: Optional[str] = None

class EquipmentInfo(BaseModel):
    equipment_types: List[str]
    manufacturers: Dict[str, List[str]]
    models: Dict[str, Dict[str, List[str]]]
    sections: Dict[str, List[str]]
    components: Dict[str, List[str]]

# In-memory storage (replace with database in production)
inspections_db = []

# Global model instance
gemini_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize Gemini model on startup"""
    global gemini_model
    try:
        gemini_model = genai.GenerativeModel("gemini-3-flash")
        print("✓ Gemini model initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Gemini model: {str(e)}")
        raise

def validate_hierarchy(equipment_type: str, manufacturer: str, model: str, 
                       section: str, component: str) -> tuple[bool, str]:
    """Validate the equipment hierarchy"""
    
    # Validate equipment type
    if equipment_type not in EQUIPMENT_HIERARCHY:
        return False, f"Invalid equipment type. Must be one of: {list(EQUIPMENT_HIERARCHY.keys())}"
    
    # Validate manufacturer
    if manufacturer not in EQUIPMENT_HIERARCHY[equipment_type]:
        valid_manufacturers = list(EQUIPMENT_HIERARCHY[equipment_type].keys())
        return False, f"Invalid manufacturer for {equipment_type}. Must be one of: {valid_manufacturers}"
    
    # Validate model
    valid_models = EQUIPMENT_HIERARCHY[equipment_type][manufacturer]["models"]
    if model not in valid_models:
        return False, f"Invalid model for {manufacturer}. Must be one of: {valid_models}"
    
    # Validate section (currently only Compact Excavator sections defined)
    if equipment_type == "Compact Excavator":
        if section not in COMPACT_EXCAVATOR_SECTIONS:
            return False, f"Invalid section. Must be one of: {list(COMPACT_EXCAVATOR_SECTIONS.keys())}"
        
        # Validate component
        if component not in COMPACT_EXCAVATOR_SECTIONS[section]:
            return False, f"Invalid component for section '{section}'. Must be one of: {COMPACT_EXCAVATOR_SECTIONS[section]}"
    
    return True, "Valid"

def create_inspection_prompt(equipment_type: str, manufacturer: str, model: str, 
                            section: str, component: str) -> str:
    """Create structured prompt for Gemini Pro Vision with full equipment context"""
    
    prompt = f"""You are a technical vision inspector for heavy construction equipment, specifically trained in equipment inspection protocols.

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
   Document specific visual evidence:
   - Exact location of any defects
   - Size/extent of damage (measurements if possible)
   - Color changes indicating corrosion or heat damage
   - Texture changes indicating wear or fatigue
   - Any functional indicators visible in the image

4. RECOMMENDED ACTION:
   Provide specific, actionable recommendations based on findings.

OUTPUT FORMAT:
Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{
  "defect_present": true or false,
  "defect_type": "specific defect description or null",
  "severity": 0-100,
  "observations": "detailed technical observations with specific evidence",
  "recommended_action": "specific action steps required"
}}

REFERENCE EXAMPLES:

Example 1 - Healthy Component (Hydraulic Cylinder):
{{
  "defect_present": false,
  "defect_type": null,
  "severity": 5,
  "observations": "Hydraulic cylinder rod shows smooth chrome surface with no visible scoring, pitting, or corrosion. Cylinder body paint intact. Rod seal appears dry with no hydraulic fluid leakage. No visible dents or deformation. Normal operational wear only.",
  "recommended_action": "Continue normal operation. Perform routine inspection per manufacturer's schedule. Monitor for any changes in next scheduled inspection."
}}

Example 2 - Moderate Defect (Track Belt):
{{
  "defect_present": true,
  "defect_type": "Excessive wear on track grouser with rust formation",
  "severity": 55,
  "observations": "Track grouser height reduced to approximately 60% of original specification. Visible rust formation on worn surfaces indicating prolonged wear. Three consecutive grousers show similar wear pattern. Belt tension appears normal. No cracks visible in track links.",
  "recommended_action": "Schedule track replacement within 2 weeks. Monitor daily for accelerated wear or track separation. Avoid high-stress applications until replacement. Order replacement track belt to minimize downtime."
}}

Example 3 - Critical Defect (Boom Pin):
{{
  "defect_present": true,
  "defect_type": "Severe crack in boom pin with material displacement",
  "severity": 92,
  "observations": "Large crack approximately 8cm long visible on boom base pin surface. Crack shows rust staining indicating it has propagated over time. Metal displacement visible suggesting structural compromise. Pin bushing shows excessive wear and lateral movement. Critical load-bearing component failure imminent.",
  "recommended_action": "IMMEDIATE ACTION REQUIRED: Remove equipment from service immediately. Tag out equipment. Do not operate under any circumstances. Schedule emergency repair with certified technician. Perform ultrasonic or magnetic particle inspection to assess crack depth. Replace pin and bushings. Inspect surrounding boom structure for stress damage."
}}

Now analyze the provided image of the {component} and provide your assessment in JSON format only:"""
    
    return prompt

def parse_gemini_response(response_text: str) -> DefectResponse:
    """Parse Gemini response and extract JSON"""
    try:
        # Try to find JSON in the response
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Parse JSON
        data = json.loads(response_text)
        
        # Validate and create response
        return DefectResponse(
            defect_present=data.get("defect_present", False),
            defect_type=data.get("defect_type"),
            severity=min(max(data.get("severity", 0), 0), 100),
            observations=data.get("observations", "No observations provided"),
            recommended_action=data.get("recommended_action", "No action specified")
        )
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse Gemini response as JSON: {str(e)}\nResponse: {response_text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing response: {str(e)}"
        )

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Heavy Equipment Defect Detection API - EquipmentGram",
        "version": "1.0.0",
        "equipment_coverage": "15 models across 3 manufacturers",
        "total_components": "~85 per equipment model",
        "endpoints": {
            "POST /inspect": "Analyze equipment component for defects",
            "GET /inspections": "Get all inspection history",
            "GET /inspections/{id}": "Get specific inspection",
            "GET /equipment-info": "Get complete equipment hierarchy",
            "GET /manufacturers": "Get list of manufacturers",
            "GET /models": "Get models by manufacturer",
            "GET /sections": "Get sections by equipment type",
            "GET /components": "Get components by section",
            "GET /health": "Health check"
        },
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if gemini_model is None:
            raise Exception("Gemini model not initialized")
        return {
            "status": "healthy",
            "gemini_api": "connected",
            "gemini_model": "gemini-3-flash",
            "equipment_types": len(EQUIPMENT_HIERARCHY),
            "total_sections": len(COMPACT_EXCAVATOR_SECTIONS),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/equipment-info", response_model=EquipmentInfo)
async def get_equipment_info():
    """Get complete equipment hierarchy information"""
    
    # Build manufacturers list by equipment type
    manufacturers_by_type = {}
    for eq_type, manufacturers in EQUIPMENT_HIERARCHY.items():
        manufacturers_by_type[eq_type] = list(manufacturers.keys())
    
    # Build models structure
    models_structure = {}
    for eq_type, manufacturers in EQUIPMENT_HIERARCHY.items():
        models_structure[eq_type] = {}
        for manufacturer, data in manufacturers.items():
            models_structure[eq_type][manufacturer] = data["models"]
    
    return EquipmentInfo(
        equipment_types=list(EQUIPMENT_HIERARCHY.keys()),
        manufacturers=manufacturers_by_type,
        models=models_structure,
        sections={"Compact Excavator": list(COMPACT_EXCAVATOR_SECTIONS.keys())},
        components=COMPACT_EXCAVATOR_SECTIONS
    )

@app.get("/manufacturers")
async def get_manufacturers(equipment_type: str = "Compact Excavator"):
    """Get list of manufacturers for a specific equipment type"""
    if equipment_type not in EQUIPMENT_HIERARCHY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid equipment type. Must be one of: {list(EQUIPMENT_HIERARCHY.keys())}"
        )
    
    return {
        "equipment_type": equipment_type,
        "manufacturers": list(EQUIPMENT_HIERARCHY[equipment_type].keys())
    }

@app.get("/models")
async def get_models(equipment_type: str = "Compact Excavator", manufacturer: str = None):
    """Get list of models for a manufacturer"""
    if equipment_type not in EQUIPMENT_HIERARCHY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid equipment type. Must be one of: {list(EQUIPMENT_HIERARCHY.keys())}"
        )
    
    if manufacturer:
        if manufacturer not in EQUIPMENT_HIERARCHY[equipment_type]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid manufacturer. Must be one of: {list(EQUIPMENT_HIERARCHY[equipment_type].keys())}"
            )
        return {
            "equipment_type": equipment_type,
            "manufacturer": manufacturer,
            "models": EQUIPMENT_HIERARCHY[equipment_type][manufacturer]["models"]
        }
    
    # Return all models by manufacturer
    all_models = {}
    for mfr, data in EQUIPMENT_HIERARCHY[equipment_type].items():
        all_models[mfr] = data["models"]
    
    return {
        "equipment_type": equipment_type,
        "models_by_manufacturer": all_models
    }

@app.get("/sections")
async def get_sections(equipment_type: str = "Compact Excavator"):
    """Get list of sections for equipment type"""
    if equipment_type == "Compact Excavator":
        return {
            "equipment_type": equipment_type,
            "sections": list(COMPACT_EXCAVATOR_SECTIONS.keys()),
            "total_sections": len(COMPACT_EXCAVATOR_SECTIONS)
        }
    
    raise HTTPException(
        status_code=400,
        detail=f"Sections not defined for equipment type: {equipment_type}"
    )

@app.get("/components")
async def get_components(section: str = None):
    """Get list of components for a section"""
    if section:
        if section not in COMPACT_EXCAVATOR_SECTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid section. Must be one of: {list(COMPACT_EXCAVATOR_SECTIONS.keys())}"
            )
        return {
            "section": section,
            "components": COMPACT_EXCAVATOR_SECTIONS[section],
            "total_components": len(COMPACT_EXCAVATOR_SECTIONS[section])
        }
    
    # Return all components by section
    total = sum(len(comps) for comps in COMPACT_EXCAVATOR_SECTIONS.values())
    return {
        "all_sections": COMPACT_EXCAVATOR_SECTIONS,
        "total_components": total
    }

@app.post("/inspect", response_model=InspectionResult)
async def inspect_equipment(
    equipment_type: str = Form(...),
    manufacturer: str = Form(...),
    model: str = Form(...),
    section: str = Form(...),
    component: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Analyze equipment component image for defects using Gemini Pro Vision
    
    Parameters:
    - equipment_type: Type of equipment (e.g., "Compact Excavator")
    - manufacturer: Manufacturer name (e.g., "Bobcat", "Caterpillar", "John Deere")
    - model: Specific model (e.g., "E10", "300.9D", "26 G-Tier")
    - section: Equipment section (e.g., "Engine", "Hydraulics")
    - component: Specific component (e.g., "Radiator", "Bucket Cylinder")
    - image: Image file of the component
    
    Returns:
    - InspectionResult with defect analysis
    """
    
    # Validate hierarchy
    is_valid, error_msg = validate_hierarchy(equipment_type, manufacturer, model, section, component)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image"
        )
    
    try:
        # Read image file
        image_bytes = await image.read()
        
        # Validate image size (max 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image size must be less than 10MB"
            )
        
        # Open image with PIL
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # Create prompt with full hierarchy context
        prompt = create_inspection_prompt(equipment_type, manufacturer, model, section, component)
        
        # Call Gemini Pro Vision using pre-initialized model
        if gemini_model is None:
            raise HTTPException(
                status_code=503,
                detail="Gemini model not initialized. Please check server startup logs."
            )
        
        response = gemini_model.generate_content([prompt, pil_image])
        
        # Parse response
        defect_result = parse_gemini_response(response.text)
        
        # Create inspection record
        inspection_id = f"INS-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(inspections_db) + 1}"
        
        # Convert image to base64 for storage (optional, truncated for response)
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
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
            observations=defect_result.observations,
            recommended_action=defect_result.recommended_action,
            image_base64=image_base64[:100] + "..." if len(image_base64) > 100 else image_base64
        )
        
        # Store in database
        inspections_db.append(inspection_result.dict())
        
        return inspection_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing inspection: {str(e)}"
        )

@app.get("/inspections")
async def get_inspections(
    limit: int = 50,
    equipment_type: str = None,
    manufacturer: str = None,
    section: str = None,
    defect_only: bool = False
):
    """Get inspection history with optional filters"""
    filtered = inspections_db.copy()
    
    if equipment_type:
        filtered = [i for i in filtered if i["equipment_type"] == equipment_type]
    
    if manufacturer:
        filtered = [i for i in filtered if i["manufacturer"] == manufacturer]
    
    if section:
        filtered = [i for i in filtered if i["section"] == section]
    
    if defect_only:
        filtered = [i for i in filtered if i["defect_present"]]
    
    return {
        "inspections": filtered[-limit:],
        "total": len(filtered),
        "filters_applied": {
            "equipment_type": equipment_type,
            "manufacturer": manufacturer,
            "section": section,
            "defect_only": defect_only
        }
    }

@app.get("/inspections/{inspection_id}")
async def get_inspection(inspection_id: str):
    """Get specific inspection by ID"""
    for inspection in inspections_db:
        if inspection["id"] == inspection_id:
            return inspection
    
    raise HTTPException(
        status_code=404,
        detail=f"Inspection {inspection_id} not found"
    )

@app.delete("/inspections/{inspection_id}")
async def delete_inspection(inspection_id: str):
    """Delete specific inspection"""
    global inspections_db
    
    for i, inspection in enumerate(inspections_db):
        if inspection["id"] == inspection_id:
            deleted = inspections_db.pop(i)
            return {
                "message": "Inspection deleted successfully",
                "deleted_inspection": deleted
            }
    
    raise HTTPException(
        status_code=404,
        detail=f"Inspection {inspection_id} not found"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

