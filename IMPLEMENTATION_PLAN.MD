# FILLY System Implementation Plan

## 1. Objective

Build the complete FILLY application as a web-based Filipino text correction system with:

1. A Grammarly-like frontend text editor.
2. A backend API that receives text from the frontend.
3. A Filipino spelling/text normalization module based on:

   * `efficient-spelling-normalization-filipino-main.zip`
   * Character N-gram transformation rules.
   * Damerau–Levenshtein Distance candidate ranking.
4. A trained GECToR grammatical error correction model using:

   * `best.pt`
   * `iteration_count = 5`
5. A strict sequential correction pipeline:

```text
User Input
   ↓
Filipino Normalization
   ↓
Normalized Text
   ↓
GECToR Iteration 1
   ↓
GECToR Iteration 2
   ↓
GECToR Iteration 3
   ↓
GECToR Iteration 4
   ↓
GECToR Iteration 5
   ↓
Final Corrected Text
```

The normalization module and GEC module MUST remain separately callable even though the main FILLY endpoint executes them sequentially.

---

# 2. Core System Rule

FILLY's production pipeline MUST always follow:

```text
raw Filipino text
→ normalization
→ GECToR
→ corrected Filipino text
```

Do NOT reverse the order.

The normalization module is responsible for nonstandard word forms such as:

* spelling variations
* abbreviations
* shortcut spellings
* informal Filipino forms
* slang covered by the normalization resources

The GECToR module is responsible for grammatical correction.

The normalizer must NOT attempt sentence-level grammatical correction.

The GEC module must NOT be used as a substitute for normalization.

---

# 3. Proposed Technology Stack

## Frontend

Use:

* React
* TypeScript
* Vite
* TailwindCSS
* TipTap
* Zustand
* Axios

TipTap should serve as the primary editable text surface because FILLY needs inline decorations and suggestion interaction.

## Backend

Use:

* Python
* FastAPI
* Pydantic
* Uvicorn
* PyTorch
* existing GECToR inference dependencies
* normalization dependencies from the provided repository

The MVP does not require a database.

Models and normalization resources should be loaded once when the backend starts rather than reloaded for every request.

---

# 4. High-Level Architecture

```text
┌─────────────────────────────┐
│        FILLY Frontend       │
│                             │
│  TipTap Editor              │
│  Suggestion Sidebar         │
│  Accept / Ignore            │
│  Highlighting               │
└──────────────┬──────────────┘
               │
               │ POST /api/v1/analyze
               ▼
┌─────────────────────────────┐
│         FastAPI API         │
│                             │
│ Analysis Controller         │
│ Request Validation          │
│ Response Serialization      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       FILLY Pipeline        │
│                             │
│  1. Normalization           │
│  2. GECToR × 5              │
│  3. Edit Alignment          │
│  4. Suggestion Generation   │
└──────────┬─────────┬────────┘
           │         │
           ▼         ▼
   ┌────────────┐ ┌─────────────┐
   │ Normalizer │ │   GECToR    │
   │ N-Gram+DLD │ │   best.pt   │
   └────────────┘ └─────────────┘
```

---

# 5. Recommended Project Structure

```text
FILLY/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   │
│   │   ├── api/
│   │   │   ├── routes_analysis.py
│   │   │   ├── routes_normalization.py
│   │   │   ├── routes_gec.py
│   │   │   └── routes_health.py
│   │   │
│   │   ├── schemas/
│   │   │   ├── analysis.py
│   │   │   ├── normalization.py
│   │   │   ├── gec.py
│   │   │   └── suggestion.py
│   │   │
│   │   ├── services/
│   │   │   ├── normalization_service.py
│   │   │   ├── gec_service.py
│   │   │   ├── filly_pipeline.py
│   │   │   ├── alignment_service.py
│   │   │   └── suggestion_service.py
│   │   │
│   │   ├── normalization/
│   │   │   ├── ngram.py
│   │   │   ├── candidate_generator.py
│   │   │   ├── dld.py
│   │   │   ├── tokenizer.py
│   │   │   ├── vocabulary.py
│   │   │   └── model.py
│   │   │
│   │   ├── gec/
│   │   │   ├── model_loader.py
│   │   │   ├── inference.py
│   │   │   ├── tag_decoder.py
│   │   │   └── iterative_corrector.py
│   │   │
│   │   └── config.py
│   │
│   ├── scripts/
│   │   ├── train_normalizer.py
│   │   ├── evaluate_normalizer.py
│   │   ├── validate_gec_checkpoint.py
│   │   └── run_pipeline_test.py
│   │
│   ├── models/
│   │   └── gec/
│   │       └── best.pt
│   │
│   ├── artifacts/
│   │   └── normalizer/
│   │       ├── rules.json
│   │       ├── vocabulary.txt
│   │       └── metadata.json
│   │
│   ├── data/
│   │   └── normalization/
│   │       ├── train.csv
│   │       ├── validation.csv
│   │       └── test.csv
│   │
│   ├── tests/
│   │   ├── test_normalizer.py
│   │   ├── test_gec.py
│   │   ├── test_pipeline.py
│   │   ├── test_alignment.py
│   │   └── test_api.py
│   │
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── editor/
│   │   │   │   ├── FillyEditor.tsx
│   │   │   │   └── SuggestionDecoration.ts
│   │   │   │
│   │   │   ├── suggestions/
│   │   │   │   ├── SuggestionSidebar.tsx
│   │   │   │   ├── SuggestionCard.tsx
│   │   │   │   └── SuggestionLegend.tsx
│   │   │   │
│   │   │   └── AnalysisToolbar.tsx
│   │   │
│   │   ├── api/
│   │   │   └── fillyApi.ts
│   │   │
│   │   ├── store/
│   │   │   └── editorStore.ts
│   │   │
│   │   ├── types/
│   │   │   └── analysis.ts
│   │   │
│   │   ├── pages/
│   │   │   └── EditorPage.tsx
│   │   │
│   │   └── App.tsx
│   │
│   └── package.json
│
└── README.md
```

---

# 6. Phase 1 — Integrate the Filipino Normalization Repository

Use the provided:

```text
efficient-spelling-normalization-filipino-main.zip
```

The useful algorithmic path from the repository should be extracted and adapted into FILLY rather than embedding the entire experimental repository unchanged.

The important algorithm is the N-Gram + Damerau–Levenshtein approach.

The repository's relevant behavior consists approximately of:

```text
informal → normalized training pairs
          ↓
collect_rules()
          ↓
N-gram transformation dictionary
          ↓
recursive candidate generation
          ↓
Filipino vocabulary filtering
          ↓
Damerau–Levenshtein ranking
          ↓
best normalized candidate
```

The existing experimental implementation should therefore be converted into a reusable FILLY `Normalizer` class.

Example interface:

```python
class FilipinoNormalizer:
    def __init__(self, rules_path, vocabulary_path):
        ...

    def normalize_word(self, word: str) -> NormalizationResult:
        ...

    def normalize_text(self, text: str) -> NormalizationResult:
        ...
```

Do NOT include the repository's neural baseline models in FILLY unless required by some unrelated dependency.

The production normalizer is specifically the N-Gram + DLD implementation.

---

# 7. Phase 2 — Train the Normalization Module

The normalizer does not train a neural `.pt` model.

Its "training" means learning N-gram transformations from paired examples.

Input format:

```csv
informal,normalized
aq,ako
bka,baka
d2,dito
...
```

Implement:

```text
backend/scripts/train_normalizer.py
```

The script must:

1. Load the normalization training pairs.
2. Validate malformed or duplicate rows.
3. Generate character N-gram transformations.
4. Count transformation frequencies.
5. Resolve conflicting transformations deterministically.
6. Produce the normalizer's rule dictionary.
7. Build/load the Filipino vocabulary used for candidate validation.
8. Persist the trained resources.

Output:

```text
artifacts/normalizer/
├── rules.json
├── vocabulary.txt
└── metadata.json
```

Example metadata:

```json
{
  "algorithm": "ngram_dld",
  "ngram_min": 1,
  "ngram_max": 4,
  "training_pairs": 400,
  "candidate_ranking": "damerau_levenshtein",
  "version": "1.0"
}
```

Do not hard-code the number of training pairs. Record the actual number.

---

# 8. Sentence-Level Normalization Wrapper

The original normalization algorithm primarily works on words.

FILLY must add sentence-level processing.

Implement:

```python
normalize_text(text)
```

Flow:

```text
Sentence
   ↓
Tokenize
   ↓
For every eligible word:
   ↓
Determine whether normalization should be attempted
   ↓
Generate candidate corrections
   ↓
Rank candidates
   ↓
Select candidate
   ↓
Reconstruct sentence
```

Preserve:

* punctuation
* whitespace
* capitalization where possible
* numbers
* URLs
* email addresses
* already-correct Filipino words

Avoid aggressively changing words when no sufficiently reliable normalization candidate exists.

Return both the resulting text and individual word edits.

Example:

```json
{
  "input": "aq ay pupunta d2",
  "output": "ako ay pupunta dito",
  "changes": [
    {
      "original": "aq",
      "replacement": "ako"
    },
    {
      "original": "d2",
      "replacement": "dito"
    }
  ]
}
```

---

# 9. Phase 3 — Integrate `best.pt`

Create a dedicated GEC service.

```text
backend/app/gec/
```

FILLY must use the user's trained:

```text
best.pt
```

Do NOT retrain GECToR as part of normal application startup.

`best.pt` must only be loaded for inference.

The checkpoint must be loaded once during backend initialization.

Example:

```python
class GECService:
    def __init__(self, model_path, device):
        self.model = load_model(model_path)
        self.model.eval()

    def correct(self, text):
        ...
```

Use:

```python
torch.inference_mode()
```

during inference.

Support:

```text
CUDA
CPU fallback
```

where appropriate.

The implementation must use the exact tokenizer, encoder architecture, transformation labels, label vocabulary, and preprocessing expected by `best.pt`.

Do not guess the architecture from the filename.

If these resources are embedded in the checkpoint, load them from the checkpoint.

If they are external, package them beside `best.pt`.

---

# 10. GECToR Five-Iteration Inference

FILLY must configure:

```python
iteration_count = 5
```

The output of one iteration becomes the input of the next.

```text
normalized text
     │
     ▼
GEC iteration 1
     │
     ▼
output 1
     │
     ▼
GEC iteration 2
     │
     ▼
output 2
     │
     ▼
GEC iteration 3
     │
     ▼
output 3
     │
     ▼
GEC iteration 4
     │
     ▼
output 4
     │
     ▼
GEC iteration 5
     │
     ▼
final corrected text
```

Implement:

```python
def correct_iteratively(text: str, iteration_count: int = 5):
    current = text

    for iteration in range(iteration_count):
        current = run_single_gec_iteration(current)

    return current
```

If the existing GECToR implementation already supports iterative inference, reuse it rather than creating a conflicting second implementation.

Keep the configuration explicitly set to:

```text
5
```

for FILLY.

Record which iteration generated each GEC correction when possible.

---

# 11. Phase 4 — Build the Main FILLY Pipeline

Implement:

```text
backend/app/services/filly_pipeline.py
```

Primary interface:

```python
class FillyPipeline:
    def analyze(self, text: str):
        ...
```

Execution:

```python
original_text = text

normalization_result = normalizer.normalize_text(original_text)

normalized_text = normalization_result.text

gec_result = gec.correct_iteratively(
    normalized_text,
    iteration_count=5
)

final_text = gec_result.text
```

Return all three states:

```text
original_text
normalized_text
corrected_text
```

This is important for both the frontend and research evaluation.

---

# 12. Preserve Module Independence

Although `/analyze` performs:

```text
Normalization → GEC
```

the system must also expose both modules independently.

Required backend operations:

```text
normalize(text)
gec(text)
analyze(text)
```

This allows FILLY research experiments to compare:

```text
Raw → GEC
```

against:

```text
Raw → Normalizer → GEC
```

without modifying production code.

---

# 13. Phase 5 — Correction Alignment

This is essential for a Grammarly-like UI.

Normalization modifies the text before GECToR sees it.

Therefore GEC offsets cannot simply be placed directly onto the original input.

Create:

```text
alignment_service.py
```

The service must track:

```text
Original
    ↓
Normalized
    ↓
GEC corrected
```

Compute:

```text
original → normalized edits

normalized → corrected edits

original → corrected final edits
```

Each suggestion must retain its provenance.

For example:

```json
{
  "id": "suggestion-12",
  "start": 4,
  "end": 7,
  "original": "bka",
  "replacement": "baka",
  "source": "normalization",
  "tag": "NORMALIZATION"
}
```

Grammar example:

```json
{
  "id": "suggestion-15",
  "start": 15,
  "end": 20,
  "original": "...",
  "replacement": "...",
  "source": "gec",
  "tag": "GRAMMAR",
  "iteration": 2
}
```

If both modules modify the same textual region, the suggestion composer must merge or reconcile the changes rather than returning overlapping invalid offsets.

Such a suggestion may contain:

```json
{
  "sources": [
    "normalization",
    "gec"
  ]
}
```

---

# 14. Suggestion Data Model

Use one consistent frontend/backend schema.

```typescript
interface FillySuggestion {
    id: string;

    start: number;
    end: number;

    original: string;
    replacement: string;

    source: "normalization" | "gec" | "combined";

    tag: string;

    confidence?: number;

    gecIteration?: number;

    explanation?: string;
}
```

Possible high-level tags:

```text
NORMALIZATION
GRAMMAR
SPELLING
WORD_FORM
VERB_FORM
INSERTION
DELETION
REPLACEMENT
OTHER
```

Do not invent detailed grammatical explanations unless the underlying GEC tag allows them to be determined reliably.

---

# 15. Phase 6 — FastAPI Endpoints

## Main Endpoint

### `POST /api/v1/analyze`

Request:

```json
{
  "text": "aq ay punta d2 kahapon"
}
```

Pipeline:

```text
request
↓
normalization
↓
5-pass GECToR
↓
alignment
↓
suggestion generation
↓
response
```

Response:

```json
{
  "original_text": "aq ay punta d2 kahapon",
  "normalized_text": "ako ay punta dito kahapon",
  "corrected_text": "...",

  "suggestions": [],

  "normalization": {
    "changes": []
  },

  "gec": {
    "iterations": 5,
    "changes": []
  }
}
```

---

## Normalization Endpoint

### `POST /api/v1/normalize`

Used for research/testing.

```json
{
  "text": "..."
}
```

Returns:

```json
{
  "original_text": "...",
  "normalized_text": "...",
  "changes": []
}
```

---

## GEC Endpoint

### `POST /api/v1/gec`

Used for raw-GEC evaluation or debugging.

```json
{
  "text": "...",
  "iterations": 5
}
```

Production/default iteration count remains 5.

---

## Health Endpoint

### `GET /api/v1/health`

Return information such as:

```json
{
  "status": "ok",
  "normalizer_loaded": true,
  "gec_loaded": true
}
```

Do not expose unnecessary internal filesystem information.

---

# 16. Phase 7 — Grammarly-Like Frontend

The user should interact primarily with one editor.

Recommended layout:

```text
┌───────────────────────────────────────────────────────────┐
│ FILLY                                                     │
├─────────────────────────────────┬─────────────────────────┤
│                                 │ Suggestions             │
│ Filipino Text Editor            │                         │
│                                 │ [Normalization]         │
│ Ako ...                         │ aq → ako                │
│      ~~~~~                      │                         │
│                                 │ Accept     Ignore       │
│                                 │                         │
│                                 │ [Grammar]               │
│                                 │ ... → ...               │
│                                 │                         │
│                                 │ Accept     Ignore       │
└─────────────────────────────────┴─────────────────────────┘
```

Do not attempt to clone Grammarly exactly.

Implement the interaction model:

* text editor
* inline correction markers
* clickable highlighted errors
* suggestion sidebar
* original form
* recommended replacement
* suggestion category
* Accept
* Ignore
* Accept All
* correction count
* analysis loading state

---

# 17. Tagging and Highlighting

The frontend must distinguish corrections produced by the two FILLY stages.

At minimum distinguish:

```text
Normalization suggestion
Grammar suggestion
```

Use different visual underline/decorations for the categories.

Example:

```text
NORMALIZATION
aq
^^

GRAMMAR
ako punta sa paaralan
    ~~~~~
```

Clicking an underline must select the matching suggestion card.

Clicking a suggestion card must highlight its corresponding text.

---

# 18. Accept Suggestion

When the user selects:

```text
Accept
```

the frontend should replace:

```text
text[start:end]
```

with:

```text
suggestion.replacement
```

Then remove that suggestion.

After modifications invalidate the old offsets, the frontend should request a fresh analysis instead of attempting to keep stale offsets indefinitely.

---

# 19. Ignore Suggestion

When the user selects:

```text
Ignore
```

the suggestion disappears from the active list without changing the text.

Ignoring is a frontend interaction only unless FILLY later adds persistent preferences.

---

# 20. Analyze Request Lifecycle

Frontend workflow:

```text
User enters Filipino text
        ↓
Frontend sends POST /api/v1/analyze
        ↓
Backend validates request
        ↓
Normalizer executes
        ↓
Normalized sentence produced
        ↓
GECToR executes 5 iterations
        ↓
Correction alignment executes
        ↓
Backend generates suggestions
        ↓
JSON response
        ↓
Frontend decorates editor
        ↓
User reviews suggestions
```

---

# 21. Frontend API Client

Create:

```text
frontend/src/api/fillyApi.ts
```

Example responsibilities:

```typescript
analyzeText(text)
normalizeText(text)
correctGrammar(text)
healthCheck()
```

Do not place API calls directly inside editor components.

---

# 22. Frontend State

Store:

```text
editorText
analyzedText
normalizedText
correctedText
suggestions
selectedSuggestion
ignoredSuggestions
analysisStatus
analysisError
```

Suggested states:

```text
idle
analyzing
success
error
```

The UI must remain responsive while the backend performs inference.

---

# 23. GECToR Iteration Metadata

During development and testing, retain iteration outputs.

Example:

```json
{
  "iteration_outputs": [
    {
      "iteration": 1,
      "text": "..."
    },
    {
      "iteration": 2,
      "text": "..."
    },
    {
      "iteration": 3,
      "text": "..."
    },
    {
      "iteration": 4,
      "text": "..."
    },
    {
      "iteration": 5,
      "text": "..."
    }
  ]
}
```

These do not necessarily need to be displayed in the production frontend.

They are useful for debugging and thesis evaluation.

---

# 24. Phase 8 — Backend Testing

## Normalizer Tests

Test:

```text
known abbreviation
known slang
known spelling variation
correct Filipino word
punctuation
capitalization
numbers
empty input
multiple spaces
unknown word
```

Verify that normalization does not unnecessarily modify valid words.

---

# 25. GEC Tests

Verify:

1. `best.pt` loads successfully.
2. Inference executes without gradients.
3. Exactly the configured 5-iteration pipeline is used.
4. Unicode Filipino text works.
5. Empty text does not crash.
6. Long input has defined behavior.
7. The same model and configuration produce deterministic inference where expected.

---

# 26. Pipeline Tests

Test separately:

```text
normalize(text)
```

```text
gec(text)
```

and:

```text
normalize(text)
→ gec(normalized_text)
```

Verify that `/analyze` NEVER performs:

```text
GEC → normalization
```

---

# 27. Critical Integration Test

Create a known example where the sentence contains both:

1. A nonstandard Filipino word.
2. A grammatical error.

The test must demonstrate:

```text
Original
↓
Normalization fixes the nonstandard word
↓
GEC receives the normalized version
↓
GEC fixes grammar
↓
Final result contains both corrections
```

The test should explicitly capture the input provided to GEC and confirm that it equals the normalizer output.

---

# 28. API Tests

Test:

```text
POST /api/v1/analyze
POST /api/v1/normalize
POST /api/v1/gec
GET /api/v1/health
```

Check:

* schema
* status codes
* Unicode
* empty input
* malformed requests
* excessively large requests
* normalizer availability
* GEC model availability

---

# 29. Frontend Tests

Verify:

* text can be entered
* Analyze works
* backend response renders correctly
* normalization tags display correctly
* grammar tags display correctly
* suggestions select corresponding text
* Accept changes text
* Ignore removes suggestion
* Accept All works
* stale offsets are not applied after text changes
* errors from the backend display gracefully

---

# 30. Phase 9 — Performance

The backend must load:

```text
normalization rules
vocabulary
GECToR model
tokenizer
tag vocabulary
```

once when the service starts.

Never reload `best.pt` for each `/analyze` request.

Expected lifecycle:

```text
FastAPI Startup
     ↓
Load Normalizer
     ↓
Load best.pt
     ↓
Models Ready
     ↓
Serve Requests
```

Use GPU inference when available.

Do not introduce complex distributed infrastructure for the thesis MVP.

---

# 31. Configuration

Use environment/configuration values such as:

```text
GECTOR_MODEL_PATH=models/gec/best.pt
NORMALIZER_RULES_PATH=artifacts/normalizer/rules.json
NORMALIZER_VOCAB_PATH=artifacts/normalizer/vocabulary.txt
GECTOR_ITERATIONS=5
MAX_INPUT_LENGTH=...
DEVICE=auto
```

For FILLY production configuration:

```text
GECTOR_ITERATIONS=5
```

must remain the default.

---

# 32. Logging

Useful development logs:

```text
request received
normalization started
normalization completed
number of normalization edits
GEC iteration 1 completed
GEC iteration 2 completed
GEC iteration 3 completed
GEC iteration 4 completed
GEC iteration 5 completed
number of grammar edits
request completed
inference duration
```

Do not log full user text by default.

---

# 33. Error Handling

The API must distinguish between:

```text
invalid request
normalization failure
GEC model failure
alignment failure
internal server error
```

Do not silently return uncorrected text when model inference crashes.

---

# 34. Preserve Research-Relevant Outputs

Because FILLY is also a research system, preserve the three important textual states:

```text
A. raw input
B. normalized input
C. final GEC output
```

For experimental code, allow:

```python
normalizer.normalize(text)

gec.correct(text)

pipeline.analyze(text)
```

This permits independent evaluation of the normalization module and GEC module.

---

# 35. Phase 10 — End-to-End Acceptance Scenario

Example conceptual input:

```text
aq punta sa paaralan kahapon
```

### Stage 1

Input:

```text
aq punta sa paaralan kahapon
```

Normalizer detects:

```text
aq → ako
```

Output:

```text
ako punta sa paaralan kahapon
```

### Stage 2

The exact normalized sentence is passed into GECToR.

```text
ako punta sa paaralan kahapon
```

### Stage 3

GECToR performs:

```text
iteration 1
iteration 2
iteration 3
iteration 4
iteration 5
```

producing the final corrected sentence according to the trained `best.pt` model.

### Stage 4

The backend returns:

```text
original
normalized
corrected
suggestions
```

### Stage 5

The frontend displays the corrections as clickable inline suggestions.

---

# 36. Implementation Order

The implementation must be completed in this order:

### Phase A — Repository and Backend Foundation

1. Create backend project.
2. Create configuration system.
3. Add FastAPI.
4. Add health endpoint.
5. Establish automated tests.

### Phase B — Normalization

6. Extract N-Gram + DLD code from the provided repository.
7. Remove experimental-only dependencies.
8. Refactor it into reusable classes.
9. Implement rule training.
10. Train FILLY normalization resources.
11. Persist the resulting rules.
12. Implement word-level inference.
13. Implement sentence-level normalization.
14. Test normalization independently.

### Phase C — GEC

15. Add GECToR inference code.
16. Load `best.pt`.
17. Restore the exact model/tag configuration used for training.
18. Implement single-pass inference.
19. Implement 5-iteration inference.
20. Test GEC independently.

### Phase D — FILLY Pipeline

21. Implement `FillyPipeline`.
22. Connect normalizer output to GEC input.
23. Preserve stage outputs.
24. Create correction alignment.
25. Generate frontend suggestions.
26. Test complete pipeline.

### Phase E — API

27. Implement `/normalize`.
28. Implement `/gec`.
29. Implement `/analyze`.
30. Add Pydantic request/response schemas.
31. Add validation and error handling.
32. Add API tests.

### Phase F — Frontend

33. Create React application.
34. Add TipTap editor.
35. Add API client.
36. Connect `/analyze`.
37. Add inline suggestion decorations.
38. Add suggestion sidebar.
39. Add normalization tags.
40. Add grammar tags.
41. Implement Accept.
42. Implement Ignore.
43. Implement Accept All.
44. Handle re-analysis after modifications.

### Phase G — Final Integration

45. Run end-to-end tests.
46. Test CPU inference.
47. Test GPU inference where available.
48. Validate normalizer → GEC order.
49. Validate five GEC iterations.
50. Validate correction offsets.
51. Validate frontend/backend contract.
52. Document installation and startup.

---

# 37. Definition of Done

FILLY is considered functionally complete only when all of the following work:

* [ ] User can type Filipino text into the frontend.
* [ ] Frontend sends the input to FastAPI.
* [ ] Backend receives the text.
* [ ] N-Gram + DLD normalization executes first.
* [ ] Trained normalization rules are loaded from persisted artifacts.
* [ ] Normalized text is produced.
* [ ] The exact normalized output becomes GECToR input.
* [ ] `best.pt` loads correctly.
* [ ] GECToR executes with `iteration_count = 5`.
* [ ] Final corrected text is produced.
* [ ] Original text is preserved.
* [ ] Normalized text is preserved.
* [ ] Final corrected text is preserved.
* [ ] Backend identifies individual corrections.
* [ ] Corrections contain valid editor offsets.
* [ ] Normalization and grammar corrections retain their source.
* [ ] Frontend renders inline correction tags.
* [ ] Suggestion cards are clickable.
* [ ] Accept works.
* [ ] Ignore works.
* [ ] Accept All works.
* [ ] Editing text does not cause stale suggestions to corrupt the document.
* [ ] Normalization can be tested independently.
* [ ] GEC can be tested independently.
* [ ] Full FILLY pipeline can be tested independently.
* [ ] CPU startup works.
* [ ] GPU inference works when compatible hardware exists.
* [ ] Models/resources are loaded once during backend startup.
* [ ] End-to-end automated tests pass.

---

# 38. Important Implementation Constraints

Do NOT:

* run GEC before normalization
* combine normalizer training with GEC training
* turn the normalizer into another neural model
* replace the N-Gram + DLD method with a different normalization algorithm
* allow normalization to perform grammar correction
* use `best.pt` without verifying its matching architecture and tag vocabulary
* reload `best.pt` for every request
* lose normalization edits after GEC inference
* return raw GEC offsets as original-text offsets without alignment
* couple experimental evaluation code directly to UI components
* make the normalizer and GEC impossible to call separately
* silently bypass either stage when `/analyze` is used

---

# 39. Final FILLY Architecture

```text
                       FILLY
                         │
                         ▼
                ┌─────────────────┐
                │ React + TipTap  │
                │     Editor      │
                └────────┬────────┘
                         │
                         │ /api/v1/analyze
                         ▼
                ┌─────────────────┐
                │     FastAPI     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ FILLY Pipeline  │
                └────────┬────────┘
                         │
              ┌──────────▼──────────┐
              │ Filipino Normalizer │
              │                     │
              │ N-Gram Rules        │
              │ Candidate Generator │
              │ DLD Ranking         │
              └──────────┬──────────┘
                         │
                         │ normalized_text
                         ▼
              ┌─────────────────────┐
              │      GECToR         │
              │                     │
              │      best.pt        │
              │                     │
              │ Iteration 1         │
              │ Iteration 2         │
              │ Iteration 3         │
              │ Iteration 4         │
              │ Iteration 5         │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Alignment +         │
              │ Suggestion Composer │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Grammarly-Like UI   │
              │                     │
              │ Normalization Tags  │
              │ Grammar Tags        │
              │ Accept / Ignore     │
              └─────────────────────┘
```

The central FILLY invariant is therefore:

```text
INPUT
  ↓
N-GRAM + DLD NORMALIZATION
  ↓
NORMALIZED FILIPINO
  ↓
GECTOR best.pt × 5 ITERATIONS
  ↓
FINAL CORRECTED FILIPINO
```
