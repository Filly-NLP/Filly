# FILLY_PROJECT_MEMORY.md

## Project Overview

**Project Name:** FILLY
**Project Type:** Filipino Grammarly-like Writing Assistant
**Description:** FILLY is a web-based Filipino writing assistance system that combines:
1. N-gram-based text normalization
2. Transformer-based grammar error correction
to provide real-time writing feedback.

---

## Core Technologies

### Frontend
- **React**
- **TypeScript**
- **Vite**
- **TailwindCSS**
- **TipTap Editor**
- **Zustand**
- **Axios**

### Backend
- **Python**
- **FastAPI**
- **Pydantic**
- **SQLAlchemy**
- **Uvicorn**

### Database
- **SQLite** (Development)
- **PostgreSQL** (Production)

### NLP Components
- **PyTorch**
- **Transformers**
- **GECToR**
- **RoBERTa Tagalog Large**

---

## Research Foundations

### Text Normalization
- **Based on:** *Look Ma, Only 400 Samples! Revisiting the Effectiveness of Automatic N-Gram Rule Generation for Spelling Normalization in Filipino*
- **Repository:** [efficient-spelling-normalization-filipino](https://github.com/ljyflores/efficient-spelling-normalization-filipino)
- **Purpose:** Detect and normalize slang, abbreviations, spelling variations, and noisy text.
- **Examples:**
  - `aq` → `ako`
  - `u` → `ikaw`
  - `nmn` → `naman`

### Grammar Error Correction
- **Based on:** *Balarila: Deep Learning for Semantic Grammar Error Correction in Low-Resource Settings*
- **Model:** GECToR + RoBERTa Tagalog Large
- **Purpose:** Detect grammatical errors, generate correction suggestions, and allow user acceptance or rejection.

---

## UI Design Requirements

### Layout
1. **Left Sidebar:**
   - FILLY logo
   - Write
   - Analytics
   - Guide
2. **Center Panel:**
   - Document title
   - Upload button
   - Save button
   - Editor
   - Word counter
3. **Right Panel:**
   - Recommendations
   - Accept suggestion
   - Ignore suggestion

### Normalization Suggestions Behavior
- Underline word in red.
- Hover displays normalization suggestion (e.g., `aq → ako`).
- Accept replaces text.
- Ignore dismisses suggestion.

### Grammar Suggestions Behavior
- Highlight grammar issue.
- Display recommendation in side panel.
- Accept updates text.
- Ignore removes recommendation.

---

## Architecture

### Frontend
```text
frontend/
├── src/
│   ├── pages/
│   │   ├── AnalyticsPage.tsx
│   │   ├── GuidePage.tsx
│   │   └── WritePage.tsx
│   ├── components/
│   │   ├── Editor.tsx
│   │   ├── RecommendationsPanel.tsx
│   │   ├── Sidebar.tsx
│   │   ├── SuggestionCard.tsx
│   │   ├── Toolbar.tsx
│   │   └── WordCounter.tsx
│   ├── editor/
│   │   ├── GrammarMark.ts
│   │   └── NormalizationMark.ts
│   ├── hooks/
│   │   └── useDebounce.ts
│   ├── services/
│   │   └── api.ts
│   ├── store/
│   │   ├── analyticsStore.ts
│   │   └── editorStore.ts
│   └── types/
│       └── index.ts
```

### Backend
```text
backend/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py       # All API routes (analyze, document, upload, etc.)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── normalizer.py      # N-gram Filipino text normalizer (148 mappings)
│   │   └── gec.py             # Rule-based grammar checker (stub for GECToR)
│   ├── models/
│   │   └── models.py
│   ├── schemas/
│   │   └── schemas.py
│   ├── database/
│   │   └── database.py
│   ├── core/
│   │   └── config.py
│   └── main.py                # FastAPI entrypoint with CORS & lifespan
│
├── normalization/              # TODO: External n-gram training data (Milestone 3)
├── gec/                        # TODO: GECToR model weights (Milestone 3)
├── requirements.txt
└── venv/
```

---

## PROJECT STATUS

### COMPLETED
- Frontend UI components: Sidebar, Navigation, Writer interface, Word Counter, and Recommendation Panel cards.
- TipTap Editor integration with custom text decoration marks: `GrammarMark` and `NormalizationMark`.
- Frontend state management via Zustand stores (`editorStore.ts` and `analyticsStore.ts`).
- Frontend service configuration (`api.ts`) pointing to `/api/*` backend routes.
- Backend database schema using SQLAlchemy (`models.py`) with tables for `documents` and `suggestions`.
- Backend Pydantic schema validation (`schemas.py`) for requesting analysis, saving docs, and retrieving stats.
- **FastAPI server entrypoint** (`main.py`) with CORS middleware, lifespan-managed startup, and router mounting.
- **API endpoints** (`endpoints.py`): All 8 routes — `/api/analyze`, `/api/document` (CRUD), `/api/documents`, `/api/upload`, `/api/suggestion/{id}/accept|ignore`, `/api/analytics`, `/api/health`.
- **N-gram normalizer** (`normalizer.py`): 148-entry Filipino slang/abbreviation dictionary with Damerau-Levenshtein fuzzy matching fallback.
- **Rule-based GEC** (`gec.py`): Covers repeated words, din/rin rule, daw/raw rule, and ng/nang confusion.
- **Python venv** created with all dependencies installed (Python 3.14 compatible).
- **All endpoints verified** via HTTP tests (health, analyze, document CRUD, documents list, analytics).

### IN PROGRESS
- None

### NEXT TASKS
- Integrate GECToR + RoBERTa Tagalog Large transformer model for deep grammar correction (replace rule-based stub).
- Download and configure model weights under `backend/gec/`.
- Connect frontend saving/loading functionality to actual API paths (end-to-end integration testing).
- Add external n-gram training data under `backend/normalization/` for expanded coverage.
- Implement full analytics tracking (persist suggestions to DB when analyzing, categorize normalization types).

### BLOCKERS
- None

---

## DEVELOPMENT LOG

- **Date:** 2026-06-16 (Session 1)
- **Task:** Initialized persistent project memory document (`FILLY_PROJECT_MEMORY.md`).
- **Files Created:** [FILLY_PROJECT_MEMORY.md](file:///d:/files/Online%20Classes/College/3rd%20Year/2nd%20Sem/random/filly/FILLY_PROJECT_MEMORY.md)
- **Files Modified:** None
- **Reasoning:** Created documentation mapping current progress, technology stack, and architecture schemas so future sessions can immediately resume development.
- **Next Step:** Create backend entrypoint server under `backend/app/main.py`.

---

- **Date:** 2026-06-16 (Session 2 — Milestone 2)
- **Task:** Built the complete FastAPI backend server with API endpoints, NLP services, and verified all routes.
- **Files Created:**
  - [main.py](file:///d:/files/Online%20Classes/College/3rd%20Year/2nd%20Sem/random/filly/backend/app/main.py) — FastAPI entrypoint with CORS, lifespan, router mounting
  - [endpoints.py](file:///d:/files/Online%20Classes/College/3rd%20Year/2nd%20Sem/random/filly/backend/app/api/endpoints.py) — All 8 API routes matching documented contracts
  - [normalizer.py](file:///d:/files/Online%20Classes/College/3rd%20Year/2nd%20Sem/random/filly/backend/app/services/normalizer.py) — Filipino N-gram normalizer (148 slang/abbreviation mappings + fuzzy matching)
  - [gec.py](file:///d:/files/Online%20Classes/College/3rd%20Year/2nd%20Sem/random/filly/backend/app/services/gec.py) — Rule-based grammar checker (repeated words, din/rin, daw/raw, ng/nang)
  - [__init__.py (api)](file:///d:/files/Online%20Classes/College/3rd%20Year/2nd%20Sem/random/filly/backend/app/api/__init__.py)
  - [__init__.py (services)](file:///d:/files/Online%20Classes/College/3rd%20Year/2nd%20Sem/random/filly/backend/app/services/__init__.py)
- **Files Modified:**
  - [requirements.txt](file:///d:/files/Online%20Classes/College/3rd%20Year/2nd%20Sem/random/filly/backend/requirements.txt) — Updated to use >= constraints for Python 3.14 compatibility; commented out heavy ML deps until Milestone 3
- **Reasoning:**
  - NLP services are implemented as internal modules under `app/services/` (not `backend/normalization/` and `backend/gec/`) because the normalizer uses a built-in dictionary and the GEC is rule-based. The top-level `normalization/` and `gec/` directories are reserved for external training data and model weights in Milestone 3.
  - Used `>=` version constraints instead of `==` pins because Python 3.14 requires newer `pydantic-core` with PyO3 >=0.23.
  - GEC is a rule-based stub covering 4 grammar patterns. This will be replaced by GECToR + RoBERTa in Milestone 3.
  - Normalizer skip-list includes all valid formal Filipino words to prevent false positives from fuzzy matching.
- **Verification Results:**
  - `GET /api/health` → `{"status": "ok", "service": "FILLY"}` ✅
  - `POST /api/analyze` with `"aq po ay nag-aaral nmn kc tlga gusto ko"` → 4 normalizations (aq→ako, nmn→naman, kc→kasi, tlga→talaga) ✅
  - `POST /api/analyze` with `"siya din ang ang bata ay kumain nang pagkain"` → 3 grammar corrections (din→rin, repeated "ang ang", nang→ng) ✅
  - `POST /api/document` → Document created with auto word count ✅
  - `GET /api/document/1` → Document retrieved ✅
  - `GET /api/documents` → List returned ✅
  - `GET /api/analytics` → Stats returned ✅
- **Next Step:** Integrate GECToR + RoBERTa Tagalog Large for deep grammar error correction (Milestone 3).

---

## API DOCUMENTATION

### POST /api/analyze
- **Purpose:** Analyze text for normalizations and grammar corrections.
- **Request Schema:**
  ```json
  {
    "text": "string"
  }
  ```
- **Response Schema:**
  ```json
  {
    "normalizations": [
      {
        "word": "string",
        "suggestion": "string",
        "start": 0,
        "end": 0,
        "type": "normalization",
        "confidence": 0.0
      }
    ],
    "grammar_corrections": [
      {
        "original": "string",
        "correction": "string",
        "start": 0,
        "end": 0,
        "type": "grammar",
        "rule": "string",
        "message": "string"
      }
    ]
  }
  ```

### POST /api/document
- **Purpose:** Create or save a new document.
- **Request Schema:**
  ```json
  {
    "title": "string",
    "content": "string"
  }
  ```
- **Response Schema:**
  ```json
  {
    "id": 0,
    "title": "string",
    "content": "string",
    "word_count": 0,
    "created_at": "string",
    "updated_at": "string"
  }
  ```

### GET /api/document/{id}
- **Purpose:** Retrieve a document by ID.
- **Response Schema:** `DocumentResponse` (matches fields in POST `/api/document`)

### POST /api/upload
- **Purpose:** Extract text from an uploaded file (.docx or .txt).
- **Request Schema:** Multipart form-data with field name `file`.
- **Response Schema:**
  ```json
  {
    "text": "string"
  }
  ```

---

## DATABASE DOCUMENTATION

### Table: `documents`
- `id` (Integer, Primary Key)
- `title` (String, default "Untitled")
- `content` (Text, default "")
- `word_count` (Integer, default 0)
- `created_at` (DateTime, UTC timezone)
- `updated_at` (DateTime, UTC timezone, updates on change)

### Table: `suggestions`
- `id` (Integer, Primary Key)
- `document_id` (Integer, ForeignKey to `documents.id`)
- `type` (String, 'normalization' or 'grammar')
- `original` (Text)
- `suggestion` (Text)
- `start_pos` (Integer)
- `end_pos` (Integer)
- `accepted` (Boolean, default False)
- `ignored` (Boolean, default False)
- `created_at` (DateTime, UTC timezone)

---

## HOW TO RESUME DEVELOPMENT

1. **Current State:**
   - Frontend is fully structured and running on Vite dev server at `http://localhost:5173`.
   - Backend is fully operational with FastAPI server, all API endpoints, N-gram normalizer, and rule-based GEC.
   - Python venv exists at `backend/venv/` with all dependencies installed.
   - SQLite database (`filly.db`) is auto-created on first server startup.
2. **Last Completed Task:** Milestone 2 — Built complete FastAPI backend with API endpoints, normalization service, and grammar checker.
3. **Current Unfinished Task:** None — Milestone 2 is complete.
4. **Immediate Next Task (Milestone 3):** Integrate GECToR + RoBERTa Tagalog Large transformer model to replace the rule-based GEC stub.
5. **Commands Needed to Run Project:**
   - **Frontend:**
     ```bash
     cd frontend
     npm install
     npm run dev
     ```
   - **Backend:**
     ```bash
     cd backend
     # If venv already exists:
     # Windows: .\venv\Scripts\activate
     # Bash: source venv/bin/activate
     # If venv does not exist:
     python -m venv venv
     .\venv\Scripts\activate   # Windows
     pip install -r requirements.txt
     # Start server:
     uvicorn app.main:app --reload --port 8000
     ```
6. **Important Implementation Decisions:**
   - Vite is configured to proxy all `/api` traffic to `http://localhost:8000`.
   - NLP services use built-in dictionary + rules (no model download required).
   - GECToR + RoBERTa integration is deferred to Milestone 3 (requires `torch` and `transformers` plus model weight downloads).
   - `requirements.txt` uses `>=` version constraints (not `==` pins) for Python 3.14 compatibility.
   - The normalizer has 148 slang/abbreviation mappings and a comprehensive skip-list of valid Filipino words.
   - The GEC covers 4 rules: repeated words, din/rin, daw/raw, ng/nang.
