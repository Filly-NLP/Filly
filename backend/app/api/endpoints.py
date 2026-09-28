import logging
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.database import get_db
from app.models.models import Document, Suggestion
from app.schemas.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    TextRequest,
    NormalizeResponse,
    GECCorrectResponse,
    PipelineAnalyzeResponse,
    GrammarCorrectionItem,
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    UploadResponse,
    SuggestionActionResponse,
    AnalyticsResponse,
    NormalizationStats,
    GrammarStats,
)
from app.services.filly_pipeline import FillyPipeline
from app.services.gec import GECInputTooLong

logger = logging.getLogger(__name__)

router = APIRouter()
v1_router = APIRouter()


def get_filly_pipeline(request: Request) -> FillyPipeline:
    """Return the startup-loaded pipeline; never initialize models per request."""
    pipeline = getattr(request.app.state, "filly_pipeline", None)
    if pipeline is None or not getattr(request.app.state, "filly_ready", False):
        raise HTTPException(status_code=503, detail="FILLY services are not ready")
    return pipeline


# ─── Health ──────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "FILLY"}


@v1_router.get("/health")
def versioned_health_check(request: Request):
    """Report process readiness without loading any model resources."""
    ready = bool(getattr(request.app.state, "filly_ready", False))
    status_code = 200 if ready else 503
    if not ready:
        raise HTTPException(status_code=status_code, detail={"status": "not_ready", "service": "FILLY"})
    return {
        "status": "ok",
        "service": "FILLY",
        "ready": True,
        "gec_iterations": settings.GECTOR_ITERATIONS,
        "device": getattr(request.app.state, "filly_device", settings.DEVICE),
    }


# ─── Text Analysis ──────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(
    request: AnalyzeRequest,
    pipeline: FillyPipeline = Depends(get_filly_pipeline),
):
    """
    Analyze Filipino text for normalization suggestions and grammar errors.
    """
    try:
        result = pipeline.analyze(request.text)
    except GECInputTooLong as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    grammar_corrections = []
    # Preserve the old endpoint's input-relative offsets only where they can
    # be proven: first-pass GEC offsets target the raw text iff normalization
    # left it byte-for-byte unchanged. Later pass offsets are iteration-local.
    if result.normalized_text == result.original_text and result.gec.passes:
        grammar_corrections = [
            GrammarCorrectionItem(
                original=change.original,
                correction=change.replacement,
                start=change.start,
                end=change.end,
                type="grammar",
                rule=change.tag,
                message="",
            )
            for change in result.gec.passes[0].changes
        ]

    return AnalyzeResponse(
        normalizations=result.normalization.changes,
        grammar_corrections=grammar_corrections,
    )


@v1_router.post("/analyze", response_model=PipelineAnalyzeResponse)
def analyze_pipeline(
    request: TextRequest,
    pipeline: FillyPipeline = Depends(get_filly_pipeline),
):
    """Run normalization followed by five dependent GEC passes."""
    logger.info("Running FILLY pipeline (%d chars)", len(request.text))
    try:
        return pipeline.analyze(request.text)
    except GECInputTooLong as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@v1_router.post("/normalize", response_model=NormalizeResponse)
def normalize_text(
    request: TextRequest,
    pipeline: FillyPipeline = Depends(get_filly_pipeline),
):
    """Run the normalization ablation independently."""
    return pipeline.normalize(request.text)


@v1_router.post("/gec", response_model=GECCorrectResponse)
def correct_grammar(
    request: TextRequest,
    pipeline: FillyPipeline = Depends(get_filly_pipeline),
):
    """Run iterative GEC directly on the submitted text."""
    try:
        return pipeline.gec(request.text)
    except GECInputTooLong as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ─── Documents ───────────────────────────────────────────────────────

def _count_words(text: str) -> int:
    """Count words in a text string."""
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def _populate_document_suggestions(
    db: Session,
    doc_id: int,
    content: str,
    pipeline: FillyPipeline,
):
    """Analyze document content and save suggestions to database."""
    db.query(Suggestion).filter(Suggestion.document_id == doc_id).delete()

    if not content.strip():
        db.commit()
        return

    result = pipeline.analyze(content)
    for suggestion in result.suggestions:
        db_sugg = Suggestion(
            document_id=doc_id,
            type=(
                f"normalization:{next((c.category for c in result.normalization.changes if c.start == suggestion.start and c.end == suggestion.end), 'spelling_variation')}"
                if suggestion.source == "normalization"
                else "grammar"
            ),
            original=suggestion.original,
            suggestion=suggestion.replacement,
            start_pos=suggestion.start,
            end_pos=suggestion.end,
            accepted=False,
            ignored=False,
        )
        db.add(db_sugg)

    db.commit()


@router.post("/document", response_model=DocumentResponse)
# Plain synchronous endpoints run in FastAPI's worker threadpool. Document
# creation and updates may invoke the five-pass model pipeline.
def create_document(
    doc: DocumentCreate,
    db: Session = Depends(get_db),
    pipeline: FillyPipeline = Depends(get_filly_pipeline),
):
    """Create a new document."""
    db_doc = Document(
        title=doc.title,
        content=doc.content,
        word_count=_count_words(doc.content),
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    
    _populate_document_suggestions(db, db_doc.id, db_doc.content, pipeline)
    
    logger.info("Created document id=%d title=%s", db_doc.id, db_doc.title)
    return db_doc


@router.get("/document/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: int, db: Session = Depends(get_db)):
    """Retrieve a document by ID."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.put("/document/{doc_id}", response_model=DocumentResponse)
def update_document(
    doc_id: int,
    update: DocumentUpdate,
    db: Session = Depends(get_db),
    pipeline: FillyPipeline = Depends(get_filly_pipeline),
):
    """Update a document's title and/or content."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if update.title is not None:
        doc.title = update.title
    if update.content is not None:
        doc.content = update.content
        doc.word_count = _count_words(update.content)

    db.commit()
    db.refresh(doc)
    
    if update.content is not None:
        _populate_document_suggestions(db, doc.id, doc.content, pipeline)

    logger.info("Updated document id=%d", doc.id)
    return doc


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(db: Session = Depends(get_db)):
    """List all documents, most recent first."""
    docs = db.query(Document).order_by(Document.updated_at.desc()).all()
    return docs


# ─── File Upload ─────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a .txt or .docx file and extract its text content.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename_lower = file.filename.lower()

    if filename_lower.endswith(".txt"):
        content_bytes = await file.read()
        text = content_bytes.decode("utf-8", errors="replace")

    elif filename_lower.endswith(".docx"):
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="python-docx is not installed. Run: pip install python-docx",
            )
        content_bytes = await file.read()
        doc = DocxDocument(BytesIO(content_bytes))
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only .txt and .docx files are accepted.",
        )

    if len(text) > settings.MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Extracted text exceeds the {settings.MAX_INPUT_LENGTH}-character input limit.",
        )

    logger.info("Uploaded file '%s' (%d chars extracted)", file.filename, len(text))
    return UploadResponse(text=text)


# ─── Suggestion Actions ─────────────────────────────────────────────

@router.post("/suggestion/{suggestion_id}/accept", response_model=SuggestionActionResponse)
async def accept_suggestion(
    suggestion_id: int,
    db: Session = Depends(get_db),
):
    """Mark a suggestion as accepted."""
    suggestion = db.query(Suggestion).filter(Suggestion.id == suggestion_id).first()
    if not suggestion:
        return SuggestionActionResponse(success=True, message="Suggestion accepted")

    suggestion.accepted = True
    suggestion.ignored = False
    db.commit()
    return SuggestionActionResponse(success=True, message="Suggestion accepted")


@router.post("/suggestion/{suggestion_id}/ignore", response_model=SuggestionActionResponse)
async def ignore_suggestion(
    suggestion_id: int,
    db: Session = Depends(get_db),
):
    """Mark a suggestion as ignored."""
    suggestion = db.query(Suggestion).filter(Suggestion.id == suggestion_id).first()
    if not suggestion:
        return SuggestionActionResponse(success=True, message="Suggestion ignored")

    suggestion.ignored = True
    suggestion.accepted = False
    db.commit()
    return SuggestionActionResponse(success=True, message="Suggestion ignored")


# ─── Analytics ───────────────────────────────────────────────────────

@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics_global(db: Session = Depends(get_db)):
    """Get global analytics across all documents."""
    return _compute_analytics(db)


@router.get("/analytics/{document_id}", response_model=AnalyticsResponse)
async def get_analytics_for_document(
    document_id: int,
    db: Session = Depends(get_db),
):
    """Get analytics for a specific document."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _compute_analytics(db, document_id=document_id)


def _compute_analytics(
    db: Session,
    document_id: int | None = None,
) -> AnalyticsResponse:
    """Compute analytics from stored suggestions."""
    query = db.query(Suggestion)
    if document_id is not None:
        query = query.filter(Suggestion.document_id == document_id)

    suggestions = query.all()

    slang_count = sum(1 for s in suggestions if s.type == "normalization:slang")
    abbrev_count = sum(1 for s in suggestions if s.type == "normalization:abbreviation")
    spelling_count = sum(1 for s in suggestions if s.type == "normalization:spelling_variation")

    norm_suggestions = [s for s in suggestions if s.type.startswith("normalization")]
    gram_suggestions = [s for s in suggestions if s.type == "grammar"]

    norm_accepted = sum(1 for s in norm_suggestions if s.accepted)
    norm_ignored = sum(1 for s in norm_suggestions if s.ignored)
    gram_accepted = sum(1 for s in gram_suggestions if s.accepted)
    gram_ignored = sum(1 for s in gram_suggestions if s.ignored)

    total_issues = len(suggestions)
    total_fixed = norm_accepted + gram_accepted
    total_ignored = norm_ignored + gram_ignored

    unresolved = total_issues - total_fixed - total_ignored
    quality_score = max(0, 100 - (unresolved * 5))

    return AnalyticsResponse(
        normalization_stats=NormalizationStats(
            slang_count=slang_count,
            abbreviation_count=abbrev_count,
            spelling_variation_count=spelling_count,
            total=len(norm_suggestions),
        ),
        grammar_stats=GrammarStats(
            issues_found=len(gram_suggestions),
            issues_fixed=gram_accepted,
            issues_ignored=gram_ignored,
        ),
        quality_score=quality_score,
    )
