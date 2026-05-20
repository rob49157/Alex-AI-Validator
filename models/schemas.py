from enum import Enum
from typing import Optional
from pydantic import BaseModel


class ValidationFlag(str, Enum):
    no_extractable_text = "no_extractable_text"
    possible_scan = "possible_scan"
    ocr_failed = "ocr_failed"
    low_text_density = "low_text_density"
    high_non_alpha_ratio = "high_non_alpha_ratio"
    parse_timeout = "parse_timeout"


# ── Tier 1 ────────────────────────────────────────────────────────────────────

class Tier1Response(BaseModel):
    valid: bool
    flags: list[ValidationFlag] = []
    # Passed to Tier 2 so it doesn't re-extract
    extracted_text: Optional[str] = None
    page_count: Optional[int] = None
    message: Optional[str] = None


# ── Tier 2 ────────────────────────────────────────────────────────────────────

class DeepAnalysisRequest(BaseModel):
    arweave_hash: str
    mongo_upload_id: str
    extracted_text: Optional[str] = None
    page_count: Optional[int] = None


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    complete = "complete"
    failed = "failed"


class SemanticDuplicate(BaseModel):
    arweave_hash: str
    similarity: float


class DeepAnalysisResults(BaseModel):
    semantic_duplicates: list[SemanticDuplicate] = []
    quality_score: Optional[float] = None         # 0.0 – 1.0
    metadata_match: Optional[bool] = None
    language: Optional[str] = None
    category: Optional[str] = None
    completeness: Optional[str] = None            # "full" | "partial" | "excerpt"
    flags: list[str] = []


class DeepAnalysisJobResponse(BaseModel):
    job_id: str


class DeepAnalysisStatusResponse(BaseModel):
    status: JobStatus
    results: Optional[DeepAnalysisResults] = None
    error: Optional[str] = None
