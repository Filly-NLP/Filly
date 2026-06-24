from pydantic import BaseModel, Field
from datetime import datetime


# ─── Analysis ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., max_length=5000)


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
    normalizations: list[NormalizationItem] = []
    grammar_corrections: list[GrammarCorrectionItem] = []


# ─── Document ────────────────────────────────────────────────

class DocumentCreate(BaseModel):
    title: str = "Untitled"
    content: str = ""
    ignored_suggestions: list[str] | None = None


class DocumentUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
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
