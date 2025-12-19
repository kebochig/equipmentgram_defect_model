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
from typing import Optional, List
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

# Pydantic Models
class DefectResponse(BaseModel):
    defect_present: bool
    defect_type: Optional[str] = None
    severity: int = Field(ge=0, le=100)
    observations: str
    recommended_action: str

class InspectionResult(BaseModel):
    id: str
    part_name: str
    timestamp: str
    defect_present: bool
    defect_type: Optional[str]
    severity: int
    observations: str
    recommended_action: str
    image_base64: Optional[str] = None

# In-memory storage (replace with database in production)
inspections_db = []

# Heavy equipment parts list
EQUIPMENT_PARTS = [
    "Excavator Bucket",
    "Excavator Boom",
    "Excavator Arm",
    "Hydraulic Cylinder",
    "Track Links",
    "Truck Bed",
    "Truck Tire",
    "Engine Block",
    "Fuel Tank",
    "Radiator",
    "Cabin Glass",
    "Hydraulic Hoses",
    "Boom Pin",
    "Bucket Teeth",
    "Undercarriage",
    "Swing Bearing"
]

def create_inspection_prompt(part_name: str) -> str:
    """Create structured prompt for Gemini Pro Vision"""
    
    prompt = f"""You are a technical vision inspector for heavy equipment.
The provided image is focused entirely on the following part:
PART NAME: {part_name}

Reference information:
- You have limited examples of what the part normally looks like.
- If unsure, assume the part is correctly identified by the user.
- Look for common defects like cracks, rust, wear, deformation, leaks, or damage.

TASK:
1. Carefully analyze the image for any visible defects or damage.
2. Score severity from 0-100 where:
   - 0-20: No defect or minimal wear (normal operation)
   - 21-40: Minor defect (monitor)
   - 41-60: Moderate defect (schedule maintenance)
   - 61-80: Significant defect (urgent maintenance required)
   - 81-100: Critical defect (immediate action, safety risk)
3. Describe exactly what visual cues led to your decision.
4. Provide a recommended action.

Output your analysis in the following JSON format ONLY (no other text):
{{
  "defect_present": true or false,
  "defect_type": "description of defect type or null if no defect",
  "severity": 0-100,
  "observations": "detailed description of what you see",
  "recommended_action": "specific action to take"
}}

EXAMPLES:

Example 1 - Healthy Part:
{{
  "defect_present": false,
  "defect_type": null,
  "severity": 5,
  "observations": "The {part_name} appears to be in good condition with normal surface wear. No cracks, deformation, or unusual damage visible. Paint/coating intact.",
  "recommended_action": "Continue normal operation. Schedule routine inspection as per maintenance schedule."
}}

Example 2 - Defective Part:
{{
  "defect_present": true,
  "defect_type": "Surface crack and rust formation",
  "severity": 65,
  "observations": "Visible crack approximately 15cm long on the main surface. Heavy rust formation around the crack area indicating moisture penetration. Metal shows signs of stress.",
  "recommended_action": "Urgent maintenance required. Remove equipment from service. Inspect crack depth with NDT. Consider welding repair or part replacement."
}}

Now analyze the provided image and output ONLY the JSON response:"""
    
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
        "message": "Heavy Equipment Defect Detection API",
        "version": "1.0.0",
        "endpoints": {
            "POST /inspect": "Analyze equipment part for defects",
            "GET /inspections": "Get all inspection history",
            "GET /inspections/{id}": "Get specific inspection",
            "GET /parts": "Get list of supported parts",
            "GET /health": "Health check"
        },
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Gemini API connection
        model = genai.GenerativeModel("gemini-1.5-flash")
        return {
            "status": "healthy",
            "gemini_api": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/parts")
async def get_parts():
    """Get list of supported equipment parts"""
    return {
        "parts": EQUIPMENT_PARTS,
        "count": len(EQUIPMENT_PARTS)
    }

@app.post("/inspect", response_model=InspectionResult)
async def inspect_equipment(
    part_name: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Analyze equipment part image for defects using Gemini Pro Vision
    
    Parameters:
    - part_name: Name of the equipment part being inspected
    - image: Image file of the equipment part
    
    Returns:
    - InspectionResult with defect analysis
    """
    
    # Validate part name
    if part_name not in EQUIPMENT_PARTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid part name. Must be one of: {', '.join(EQUIPMENT_PARTS)}"
        )
    
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
        
        # Create prompt
        prompt = create_inspection_prompt(part_name)
        
        # Call Gemini Pro Vision
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([prompt, pil_image])
        
        # Parse response
        defect_result = parse_gemini_response(response.text)
        
        # Create inspection record
        inspection_id = f"INS-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(inspections_db) + 1}"
        
        # Convert image to base64 for storage (optional)
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        inspection_result = InspectionResult(
            id=inspection_id,
            part_name=part_name,
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
async def get_inspections(limit: int = 50):
    """Get inspection history"""
    return {
        "inspections": inspections_db[-limit:],
        "total": len(inspections_db)
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
