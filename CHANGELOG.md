# Changelog

All notable changes to the FILLY project will be documented in this file.

## [0.1.0] - 2026-06-24

### Added
- **Backend Infrastructure (Phase 1 Migration)**
  - Initialized backend environment config structure in `backend/app/core/config.py` using `pydantic-settings` to load environment dynamic settings.
  - Implemented database engine, SessionLocal generator, and table auto-initialization script in `backend/app/database/database.py`.
  - Defined SQLAlchemy schema mappings for `documents` and `suggestions` tables in `backend/app/models/models.py`, configuring column properties and cascade delete relationships.
  - Created web shell entrypoint and configuration in `backend/app/main.py` with CORS middleware policies, lifespan events, and `/api` prefix mounts.
  - Established routing shell structure in `backend/app/api/endpoints.py` containing a `/health` endpoint to verify operational status.
  - Added dependency management references in `backend/requirements.txt`.
- **API & Core NLP Services Integration (Phase 2 Migration)**
  - Ported Pydantic request/response payload validation contracts in `backend/app/schemas/schemas.py`.
  - Implemented `FilNormalizer` text spelling correction service in `backend/app/services/normalizer.py` supporting dictionary lookups and fuzzy Damerau-Levenshtein distance matching.
  - Implemented rule-based `GECService` grammar error checks in `backend/app/services/gec.py` correcting repeated words, Din/Rin and Daw/Raw vowel allophones, and Ng/Nang preposition confusion, with overlapping flags de-duplication.
  - Fully mapped all 8 core REST API handlers in `backend/app/api/endpoints.py` for health checks, text analysis routing, documents database CRUD persistence, suggestion accepting/ignoring, and analytics reporting.
- **GEC Deep Learning Pipeline (Phase 3 Migration)**
  - Imported GECToR deep learning modeling layers, tokenizers, vocab indices, and dataset mappers in `backend/app/services/gector/` directory.
  - Copied Tagalog grammatical vocabularies and verb inflection dictionaries under `backend/gec/data/`.
  - Configured backend venv to install PyTorch (`torch`) and Transformers (`transformers`) libraries.
  - Verified lifespan model pre-warming triggers and validation handlers.
- **Frontend Integration (Phase 4 Migration)**
  - Configured Vite development server proxy rules in `frontend/vite.config.js` to redirect all `/api/*` traffic to the backend server (port 8000).
  - Modified unified application logic in `frontend/src/fillyApp.js` to dispatch text analysis requests to `/api/analyze` instead of running locally.
  - Integrated `processApiResponse` dynamic reconstruction mapping in `frontend/src/fillyApp.js` to safely compile text replacements from back-to-front, avoiding index shifts.
- **Testing & Verification (Phase 5 Migration)**
  - Integrated `pytest` and `httpx` testing frameworks in the backend virtual environment.
  - Implemented unit tests for dictionary-based spelling lookups and fuzzy matching algorithm in `backend/tests/test_normalizer.py`.
  - Implemented unit tests for repeated words, din/rin, daw/raw, and ng/nang rules in `backend/tests/test_gec.py`.
  - Implemented API endpoints integration tests and documents database transaction CRUD checks using TestClient in `backend/tests/test_api.py`.
  - Verified that all 9 test suites execute and pass successfully.
- **Documentation (Post-Migration)**
  - Created a comprehensive `README.md` in the target repository root detailing system architecture, directory layouts, setup/run instructions, and test execution procedures.
