from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class DefectResponse(BaseModel):
    image_verified: bool = True
    verification_reason: Optional[str] = None
    defect_present: bool
    defect_type: Optional[str] = None
    severity: int = Field(ge=0, le=100)
    condition: str
    risk_level: str
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
    image_verified: bool
    verification_reason: Optional[str] = None
    defect_present: Optional[bool] = None
    defect_type: Optional[str] = None
    severity: Optional[int] = None
    condition: Optional[str] = None
    risk_level: Optional[str] = None
    observations: Optional[str] = None
    recommended_action: Optional[str] = None
    image_base64: Optional[str] = None


class ComponentBatchResult(BaseModel):
    component: str
    success: bool
    error: Optional[str] = None
    defect_present: Optional[bool] = None
    defect_type: Optional[str] = None
    severity: Optional[int] = None
    condition: Optional[str] = None
    component_score: Optional[int] = None
    risk_level: Optional[str] = None
    observations: Optional[str] = None
    recommended_action: Optional[str] = None


class BatchInspectionResult(BaseModel):
    batch_id: str
    equipment_type: str
    manufacturer: str
    model: str
    section: str
    timestamp: str
    components_inspected: int
    components_succeeded: int
    component_results: List[ComponentBatchResult]
    section_score: Optional[float] = None
    section_risk_level: Optional[str] = None


class EquipmentInfo(BaseModel):
    equipment_types: List[str]
    manufacturers: Dict[str, List[str]]
    models: Dict[str, Dict[str, List[str]]]
    sections: Dict[str, List[str]]
    components: Dict[str, List[str]]
