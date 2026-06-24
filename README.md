# FILLY — Filipino Writing Assistant

FILLY is a web-based Filipino writing assistant that combines N-gram-based spelling normalization and transformer-based grammar error correction (GEC) to provide real-time writing feedback.

---

## 1. System Architecture

The application is structured as a decoupled monorepo composed of a Python FastAPI backend and a React/Vite frontend.

```text
Filly-1/ (Repository Root)
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints.py          # API routers & endpoint controllers
│   │   ├── core/
│   │   │   └── config.py             # App environments configurations
│   │   ├── database/
│   │   │   └── database.py           # Database connection engines & helpers
│   │   ├── models/
│   │   │   └── models.py             # SQLAlchemy DB schemas
│   │   ├── schemas/
│   │   │   └── schemas.py            # Pydantic data schemas
│   │   ├── services/
│   │   │   ├── normalizer.py         # N-gram spelling normalizer
│   │   │   ├── gec.py                # GEC checker & deep learning wrapper
│   │   │   └── gector/               # Core GECToR neural network codebase
│   │   └── main.py                   # FastAPI main launcher & middlewares
│   ├── gec/
│   │   └── data/                     # Output vocabularies & Tagalog verb lookups
│   ├── tests/                        # Automated unit and API test suites
│   └── requirements.txt              # Backend dependency requirements
│
└── frontend/
    ├── src/
    │   ├── assets/                   # SVG and static assets
    │   ├── fillyApp.js               # Unified application logic & REST fetch bindings
    │   ├── fillyTemplate.html        # HTML layout template
    │   ├── index.css                 # Base stylesheet & color tokens
    │   └── main.jsx                  # React initialization entrypoint
    ├── package.json                  # Frontend dependencies
    └── vite.config.js                # Vite proxy & server settings
```

### Core Technologies

* **Frontend:** React (v18), Vite, Vanilla CSS, Recharts (for analytics dashboard visualization)
* **Backend:** Python (v3.14 compatible), FastAPI, SQLAlchemy ORM, SQLite (local development), PostgreSQL (production-ready)
* **NLP Pipeline:** PyTorch, Hugging Face Transformers, GECToR (Grammar Error Correction Sequence Tagging), RoBERTa Tagalog Large

---

## 2. Key Features

1. **Filipino Spelling Normalizer:** Detects and flags texting slang, SMS shortcuts, and dialect variations (e.g., `aq` $\rightarrow$ `ako`, `nmn` $\rightarrow$ `naman`). If no exact dictionary mapping is found, it falls back to a Damerau-Levenshtein fuzzy matching threshold check.
2. **Grammar Error Correction:** 
   - *Neural Pathway:* Integrates a sequence tagging pipeline utilizing fine-tuned RoBERTa Tagalog weights to predict and execute edit tags ($KEEP$, $DELETE$, $APPEND$, $TRANSFORM$).
   - *Rule-Based Fallback:* If model weights are absent, a local regex-based engine resolves consecutively repeated words, din/rin and daw/raw allophonic rules, and ng/nang preposition confusions.
3. **Interactive Workspace:** Word-counter tracking budget constraints (250 words max), and interactive suggestion cards to accept or ignore corrections in real-time.
4. **Analytics Dashboard:** Provides a document-level and global quality score (out of 100), tracking accepted/ignored corrections, slang frequency, and abbreviations.

---

## 3. Setup & Installation

Follow these steps to run the application locally on your machine.

### Prerequisites
* Python 3.10 or higher
* Node.js (v18 or higher) and npm

---

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   * **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * **Linux/macOS:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start the FastAPI development server:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   *Note: On first startup, SQLite database `filly.db` will be auto-generated and tables initialized.*

---

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node modules:**
   ```bash
   npm install
   ```

3. **Start the Vite development server:**
   ```bash
   npm run dev
   ```
   *Note: The frontend server starts at `http://localhost:5173`. Vite is pre-configured to proxy all incoming `/api/*` traffic to the backend running on port 8000.*

---

## 4. Running Tests

To verify that the database CRUD, services, and REST APIs are working correctly:

1. **Ensure your backend virtual environment is active.**
2. **Install testing libraries:**
   ```bash
   pip install pytest httpx
   ```
3. **Execute the test suite:**
   ```bash
   python -m pytest
   ```
