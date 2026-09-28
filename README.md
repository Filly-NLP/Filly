# FILLY — Filipino Writing Assistant

FILLY's React/Vite editor calls a FastAPI backend for Filipino spelling normalization and grammatical error correction. The backend runs the stages in this order:

```text
Input → character n-gram + Damerau–Levenshtein normalization
      → normalized text → GECToR best.pt (five dependent stages)
      → corrected text → original-text suggestions
```

The supplied `best.pt` is a **provisional Stage 2 checkpoint**, not a validated thesis-performance result. Review suggestions before accepting them.

## Prerequisites

- **Python 3.11** is the tested backend version. `backend/requirements.txt` requires PyTorch 2.6+ and exactly `transformers==5.16.1`.
- **Node.js 22.12+** (or 20.19.x) and npm. The installed Vite 8.2.2 and React plugin 6.1.1 declare these Node versions; this workspace was checked with Node 25.3.0 and npm 11.6.2.
- Git if cloning. Allow several gigabytes of disk space and memory for the checkpoint and dependencies.
- CUDA is optional. `DEVICE=auto` selects CUDA when available and otherwise uses CPU, which may be slower.

## Quick start

Use this after first-time setup. Start the backend first, then use a **second PowerShell terminal** for the frontend. Commands below start from the repository root.

**Terminal 1 — backend**

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — frontend**

```powershell
cd frontend
npm run dev
```

Open `http://localhost:5173`. Backend startup loads the normalizer and checkpoint before it reports ready.

## First-time setup

The configured repository remote is `https://github.com/Filly-NLP/Filly.git`:

```powershell
git clone https://github.com/Filly-NLP/Filly.git
cd Filly
```

**Current checkout limitation:** the integrated source, `best.pt`, normalizer artifacts, training inputs, and builder are presently uncommitted or untracked. A clone of the current Git `HEAD` alone cannot run this documented pipeline. Obtain the current source and the required runtime files from the project maintainers until they are published. Do not substitute another GEC checkpoint or invented normalization data.

### Backend environment

From the repository root on Windows PowerShell:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Unix/macOS, use `python3.11 -m venv .venv` and `source .venv/bin/activate`. There is no Poetry, Pipenv, or Conda configuration in this checkout.

If PowerShell blocks activation, use Command Prompt's `.venv\Scripts\activate.bat` or invoke `.venv\Scripts\python.exe` directly. If appropriate for your machine, `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` is another way to allow PowerShell activation; it is not a required setup step.

### Normalizer files

Startup requires these three matching files:

```text
backend/artifacts/normalizer/rules.json
backend/artifacts/normalizer/vocabulary.txt
backend/artifacts/normalizer/metadata.json
```

They are present in this workspace, so ordinary startup does **not** require rebuilding them here. They are currently untracked; provide them separately to a fresh clone or rebuild from the paired examples and base vocabulary below. The loader verifies hashes from `metadata.json` and fails startup on missing or mismatched resources.

### GEC checkpoint and encoder assets

Place the matching pretrained checkpoint at the repository root, next to this README: `Filly/best.pt`. Set `GECTOR_MODEL_PATH` if it is elsewhere. FILLY does **not** train or download `best.pt` at startup.

The checkpoint embeds its GEC label vocabulary and model configuration. The loader also needs the fast tokenizer and encoder configuration for `jcblaise/roberta-tagalog-large` at revision `acb6b204dfb1afdd7476eae5da234cbcf8899846`. It fetches those pinned assets through Hugging Face if they are not cached. `GECTOR_LOCAL_FILES_ONLY=true` requires them to be cached already. The encoder weights come from `best.pt`, not a replacement download.

### Optional backend configuration

Defaults need no environment file. From `backend/`, create `backend/.env` only to override settings. This example keeps the normal runtime behavior:

```env
GECTOR_ITERATIONS=5
DEVICE=auto
MAX_INPUT_LENGTH=5000
FILLY_DEBUG=false
```

Supported path overrides are `GECTOR_MODEL_PATH`, `NORMALIZER_RULES_PATH`, and `NORMALIZER_VOCAB_PATH`; prefer absolute paths. `metadata.json` must be beside the configured rules file. Other settings include `DATABASE_URL` (default SQLite at `backend/filly.db` when started from `backend/`) and `CORS_ORIGINS` (JSON array of browser origins). `GECTOR_ITERATIONS=5` is the production default; changing it changes the correction trace. Startup creates missing SQLite tables.

`GECTOR_CACHE_DIR` and `GECTOR_LOCAL_FILES_ONLY` are read directly from the **process environment**, not from `backend/.env`. Set them in the backend PowerShell terminal before starting the server when needed, for example `$env:GECTOR_LOCAL_FILES_ONLY='true'`.

### Frontend dependencies

From the repository root in another terminal:

```powershell
cd frontend
npm ci
```

Use the versions in `frontend/package.json` and `frontend/package-lock.json`; do not independently upgrade Vite or its React plugin. The frontend defaults to `/api/v1`, and `frontend/vite.config.js` proxies `/api` to `http://localhost:8000`. No frontend environment file is needed with the default backend port. For a different backend address, set `VITE_API_BASE_URL` in `frontend/.env.local` to its full API prefix, such as `http://127.0.0.1:8001/api/v1`, and allow the frontend origin in backend `CORS_ORIGINS`.

## Run and verify

Start both servers as in [Quick start](#quick-start). Backend: `http://127.0.0.1:8000`; frontend: `http://localhost:5173`. FastAPI documentation is at `http://127.0.0.1:8000/docs` and `/redoc`.

In PowerShell, check that the backend loaded both modules:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
```

A ready CPU instance with default settings returns this shape:

```json
{"status":"ok","service":"FILLY","ready":true,"gec_iterations":5,"device":"cpu"}
```

`device` may report CUDA on a GPU machine. Startup fails rather than reporting ready if required resources cannot load. Test the complete route with:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/analyze -Method Post -ContentType 'application/json' -Body '{"text":"aq ay masaya."}'
```

`POST /api/v1/normalize` and `POST /api/v1/gec` accept the same JSON body for module-only experiments.

## Use FILLY

Enter Filipino text in the editor and click **Analyze**. The normalizer runs first; its exact output feeds the GEC checkpoint. Select highlighted text or a suggestion card, then **Accept**, **Ignore**, or **Accept All**. The fully corrected text is a model preview; copy and download use only the changes accepted into the editor. Editing invalidates stale positions and triggers reanalysis. Long text can receive HTTP 422 if it exceeds the encoder's subword limit even when below the API's 5,000-character limit.

## Tests and build

From `backend/` with the Python environment active:

```powershell
python -m pip install pytest httpx
python -m pytest tests -q --basetemp "$env:TEMP\filly-pytest"
```

From `frontend/` after `npm ci`:

```powershell
node --test tests/*.test.js
npm run build
```

There is no frontend `npm test` script. The full backend API suite requires the runtime resources and fails when they cannot load; a few standalone checkpoint tests skip if `best.pt` is absent. A unit-test subset with skips does not establish model readiness—check the versioned health endpoint too.

## Rebuild normalization resources

This builds character n-gram transformation rules and vocabulary; it does **not** train GECToR or another neural model. From `backend/`:

```powershell
python scripts/train_normalizer.py
```

Default inputs are `backend/normalization/data/train_pairs.csv` and `backend/normalization/data/base_vocabulary.txt`, relative to the repository root. Output is the three files under `backend/artifacts/normalizer/`. These inputs are currently untracked, so a fresh clone needs them before rebuilding. The script also accepts `--pairs`, `--base-vocabulary`, and `--output`. Rebuilding replaces target artifacts; retain any set used for an experiment.

## Project structure

```text
Filly/
├── best.pt                         # supplied checkpoint; currently untracked
├── backend/
│   ├── app/                        # FastAPI routes, pipeline, model services
│   ├── artifacts/normalizer/       # rules, vocabulary, hash metadata
│   ├── normalization/data/         # rule-induction inputs
│   ├── scripts/train_normalizer.py
│   ├── tests/
│   └── requirements.txt
└── frontend/
    ├── src/                        # editor and API client
    ├── tests/
    ├── package-lock.json
    └── vite.config.js              # development API proxy
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `best.pt` is missing or rejected | Supply the matching checkpoint at repository-root `best.pt` or set `GECTOR_MODEL_PATH` to its absolute path. Another architecture is rejected. |
| Normalizer resource or hash error | Supply matching `rules.json`, `vocabulary.txt`, and `metadata.json`, or rebuild from the stated training inputs. |
| Pinned tokenizer/config cannot load | Allow access to the stated Hugging Face revision on first startup, or pre-cache it. Check `GECTOR_CACHE_DIR` and `GECTOR_LOCAL_FILES_ONLY`. |
| `ModuleNotFoundError` or wrong Transformers version | Activate the Python 3.11 environment and run `python -m pip install -r requirements.txt` from `backend/`. |
| PowerShell blocks activation | Use Command Prompt activation, invoke the venv Python directly, or use the `CurrentUser` policy step above. |
| Frontend cannot reach the backend | Wait for startup, check `/api/v1/health`, and keep port 8000 for the default Vite proxy. For another address, configure `VITE_API_BASE_URL` and CORS. |
| Port 8000 or 5173 is in use | Stop the other process or use a new port with matching frontend API base and CORS settings. |
| CUDA unavailable | Leave `DEVICE=auto` for CPU fallback or set `DEVICE=cpu`. |
| npm dependency conflict | Use `npm ci` with the checked-in lockfile and a supported Node version; avoid defaulting to `--force` or `--legacy-peer-deps`. |
| Analyze returns HTTP 422 on long text | Shorten or split the text to fit the model's subword limit. |

The older `backend/app/services/gector/` package expects another checkpoint format. The current runtime uses `gec.py` and `gec_checkpoint/` with the supplied Stage 2 checkpoint.
