from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict, Any, Union


class DegradedBlock(BaseModel):
    page: int
    location: str
    partial_text: str
    reason: str
    confidence: Literal["medium", "low"]


class FieldValue(BaseModel):
    raw_value: str
    normalized_value: Optional[Union[str, float, int]] = None
    source_page: int
    confidence: Literal["high", "medium", "low"]
    confidence_reason: Optional[str] = None
    evidence: Optional[str] = None
    corrected: bool = False
    correction_source_page: Optional[int] = None
    correction_rule_applied: Optional[str] = None
    original_value: Optional[str] = None
    correction_page: Optional[int] = None
    correction_date: Optional[str] = None


class ExtractedTable(BaseModel):
    name: str
    source_page: int
    rows: List[Dict[str, str]]
    confidence: Literal["high", "medium", "low"]


class DetectedElement(BaseModel):
    element_type: Literal["signature", "stamp", "watermark", "logo", "image", "handwriting", "checkbox"]
    page: int
    description: str
    confidence: Literal["high", "medium", "low"]


class CorrectionRecord(BaseModel):
    field: str
    original_value: str
    original_page: int
    corrected_value: str
    correction_page: int
    correction_date: Optional[str] = None
    resolution_rule: str


class MismatchFlag(BaseModel):
    field: str
    computed_value: Optional[str] = None
    stated_value: str
    description: str
    severity: Literal["warning", "error"] = "error"
    action: Literal["flagged", "left_as_stated"] = "flagged"
    difference: Optional[float] = None
    component_fields: List[str] = Field(default_factory=list)
    missing_components: List[str] = Field(default_factory=list)


class PageAnalysis(BaseModel):
    page_number: int
    is_scanned: bool = False
    has_degraded_text: bool = False
    text_quality: Literal["clean", "noisy", "degraded", "unreadable"] = "clean"
    is_correction_page: bool = False
    raw_text: str = ""
    detected_elements: List[DetectedElement] = []
    extracted_fields: Dict[str, Any] = {}
    extracted_tables: List[Dict] = []
    correction_targets: List[Dict] = []


class ExtractionResult(BaseModel):
    document_id: str
    filename: str
    document_type: str
    document_type_confidence: Literal["high", "medium", "low"]
    document_type_reason: str
    fields: Dict[str, FieldValue]
    tables: List[ExtractedTable]
    detected_elements: List[DetectedElement]
    degraded_text_blocks: List[DegradedBlock]
    corrections_applied: List[CorrectionRecord]
    mismatches: List[MismatchFlag]
    page_analyses: List[PageAnalysis]
    processing_time_seconds: float
    pages_processed: int