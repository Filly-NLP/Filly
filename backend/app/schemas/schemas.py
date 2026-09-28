from typing import Any, Literal

from pydantic import BaseModel, Field
from datetime import datetime

from app.core.config import settings


# ─── Analysis ────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str = Field(..., max_length=settings.MAX_INPUT_LENGTH)


class AnalyzeRequest(TextRequest):
    """Legacy /api/analyze request, with the same bounded text contract."""


class NormalizationItem(BaseModel):
    word: str
    suggestion: str
    start: int
    end: int
    type: str = "normalization"
    confidence: float = 0.0
    category: str = "abbreviation"


class GrammarCorrectionItem(BaseModel):
    original: str
    correction: str
    start: int
    end: int
    type: str = "grammar"
    rule: str = ""
    message: str = ""


class AnalyzeResponse(BaseModel):
    normalizations: list[NormalizationItem] = Field(default_factory=list)
    grammar_corrections: list[GrammarCorrectionItem] = Field(default_factory=list)


class SuggestionResponse(BaseModel):
    id: str
    start: int
    end: int
    original: str
    replacement: str
    source: Literal["normalization", "gec", "combined"]
    tag: str
    confidence: float | None = None
    gec_iteration: int | None = None


class GECChangeResponse(BaseModel):
    original: str
    replacement: str
    start: int
    end: int
    tag: str
    confidence: float
    iteration: int


class GECIterationResponse(BaseModel):
    iteration: int
    input_text: str
    output_text: str
    model_invoked: bool = True
    input_tokens: list[str] = Field(default_factory=list)
    output_tokens: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    changes: list[GECChangeResponse] = Field(default_factory=list)


class GECStageResponse(BaseModel):
    iterations: int
    iteration_outputs: list[str] = Field(default_factory=list)
    changes: list[GECChangeResponse] = Field(default_factory=list)
    passes: list[GECIterationResponse] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizeResponse(BaseModel):
    original_text: str
    normalized_text: str
    changes: list[NormalizationItem] = Field(default_factory=list)


class GECCorrectResponse(BaseModel):
    original_text: str
    corrected_text: str
    gec: GECStageResponse


class PipelineAnalyzeResponse(BaseModel):
    original_text: str
    normalized_text: str
    corrected_text: str
    suggestions: list[SuggestionResponse] = Field(default_factory=list)
    normalization: NormalizeResponse
    gec: GECStageResponse


# ─── Document ────────────────────────────────────────────────

class DocumentCreate(BaseModel):
    title: str = "Untitled"
    content: str = Field(default="", max_length=settings.MAX_INPUT_LENGTH)
    ignored_suggestions: list[str] | None = None


class DocumentUpdate(BaseModel):
    title: str | None = None
    content: str | None = Field(default=None, max_length=settings.MAX_INPUT_LENGTH)
    ignored_suggestions: list[str] | None = None


class DocumentResponse(BaseModel):
    id: int
    title: str
    content: str
    word_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ─── Upload ──────────────────────────────────────────────────

class UploadResponse(BaseModel):
    text: str


# ─── Suggestion Actions ─────────────────────────────────────

class SuggestionActionResponse(BaseModel):
    success: bool
    message: str = ""


# ─── Analytics ───────────────────────────────────────────────

class NormalizationStats(BaseModel):
    slang_count: int = 0
    abbreviation_count: int = 0
    spelling_variation_count: int = 0
    total: int = 0


class GrammarStats(BaseModel):
    issues_found: int = 0
    issues_fixed: int = 0
    issues_ignored: int = 0


class AnalyticsResponse(BaseModel):
    normalization_stats: NormalizationStats = NormalizationStats()
    grammar_stats: GrammarStats = GrammarStats()
    quality_score: int = 100
