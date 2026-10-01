# Capitalization investigation

## Diagnosis

Case 1: sentence-unit mismatch, confirmed against the local selected Stage-2
`best.pt`, not inferred from visible output. The old service passed all normalized
paragraph tokens to `predict([tokens], batch_size=1)`. There was no sentence split.
Punctuation was included; whitespace and newlines were retained for reconstruction
but excluded from model tokens.

For `ako ay umuwi. kumain ako.` the correction tokens were
`ako | ay | umuwi | . | kumain | ako | .`.

## Controlled checkpoint evidence before changes

The normalizer left all six inputs unchanged. Probabilities below are first-pass
raw correction-head softmax probabilities, before the KEEP adjustment. Detection
probabilities are separate; the API's confidence field uses detection probability.

| Input / relevant token | Emitted tag | Tag probability | Detection error probability |
| --- | --- | ---: | ---: |
| `kumain ako.` / `kumain` | `$TRANSFORM_CASE_CAPITAL` | 0.992920 | 0.994923 |
| `ako ay umuwi. kumain ako.` / second-sentence `kumain` | `$KEEP` | 0.997857 | 0.005222 |
| `kumain ako. ako ay umuwi.` / first `kumain` | `$TRANSFORM_CASE_CAPITAL` | 0.981880 | 0.994775 |
| Same input / second-sentence `ako` | `$KEEP` | 0.999965 | 0.000018 |
| `ako ay umuwi. Kumain ako.` / `Kumain` | `$KEEP` | 0.999862 | 0.000215 |
| `kumain ka ba? oo.` / `kumain` | `$KEEP` | 0.458871 | 0.370982 |
| Same input / `oo` | `$KEEP` | 0.999996 | 0.000004 |
| `umalis siya! bumalik siya.` / `umalis` | `$TRANSFORM_CASE_CAPITAL` | 0.989295 | 0.986225 |
| Same input / `bumalik` | `$KEEP` | 0.999590 | 0.000755 |

| Input | Five-pass model output before postprocessing | After old postprocessor |
| --- | --- | --- |
| `kumain ako.` | `Kumain ako.` | `Kumain ako.` |
| `ako ay umuwi. kumain ako.` | `Ako ay umuwi. kumakain ako.` | `Ako ay umuwi. Kumakain ako.` |
| `kumain ako. ako ay umuwi.` | `Kumain ako. ako ay umuwi.` | `Kumain ako. Ako ay umuwi.` |
| `ako ay umuwi. Kumain ako.` | `Ako ay umuwi. Kumain ako.` | `Ako ay umuwi. Kumain ako.` |
| `kumain ka ba? oo.` | `kumain ka ba? oo.` | `Kumain ka ba? Oo.` |
| `umalis siya! bumalik siya.` | `Umalis siya! babalik siya.` | `Umalis siya! Babalik siya.` |

All remaining standalone fragments emitted `$TRANSFORM_CASE_CAPITAL` in pass 1:

| Standalone input | Tag probability | Detection error probability |
| --- | ---: | ---: |
| `ako ay umuwi.` | 0.997465 | 0.996702 |
| `kumain ka ba?` | 0.990288 | 0.980647 |
| `oo.` | 0.993637 | 0.998503 |
| `umalis siya!` | 0.989002 | 0.980256 |
| `bumalik siya.` | 0.990825 | 0.996313 |

These observations support sentence-aware inference for these examples, not a
claim that the model capitalizes every possible sentence reliably.

## Model and training evidence

The active 892-label vocabulary contains `$TRANSFORM_CASE_CAPITAL` (70) and
`$TRANSFORM_CASE_LOWER` (71). It does not contain `$TRANSFORM_CASE_UPPER` or
`$TRANSFORM_CASE_CAPITAL_1`. The latter tags belong to a separate legacy runtime.
The active edit extractor and application code support CAPITAL and LOWER. Direct
label derivation and replay verified `kumain` -> CAPITAL -> `Kumain`.

No case-specific filtering exists in the active predictor. All labels share the
existing detection threshold (0.5) and KEEP adjustment (0.2).

The checkpoint declares a train-only vocabulary, which is evidence that case tags
were admitted from training labels. The original GEC training rows, preparation
manifest, and training repository are not present here; their sentence-unit
distribution, capitalization counts, and preprocessing cannot be independently
audited. `backend/normalization/data/train_pairs.csv` is a different dataset.

The checkpoint uses `jcblaise/roberta-tagalog-large` revision
`acb6b204dfb1afdd7476eae5da234cbcf8899846` and declares
`PROVISIONAL / NON_PRODUCTION / NOT_THESIS_PERFORMANCE` readiness.

## Remaining model limitation

On independent sentences, five-pass inference can amplify punctuation:
`Kumain ka ba??!!!`, `Oo?!!!`, and `Umalis siya!!!!!` were observed. These are real
model edits, not segmentation or recombination changes. The remediation does not
silently suppress model labels, change confidence thresholds, change iteration
count, or claim to resolve this separate model-quality issue. Further model/data
investigation would be needed; no retraining was performed.

## Implemented architecture

Before: normalization -> one paragraph GEC sequence per pass -> deterministic
sentence capitalization -> alignment -> API.

After: normalization -> lossless sentence segmentation -> batched sentence GEC
per pass -> exact-gap recombination -> alignment -> API.

Sentence slots remain fixed across the five passes; each sentence's output is its
next-pass input. Model calls batch up to 32 sentence rows. Local edits are shifted
by exact current sentence starts to maintain the existing paragraph-level pass
trace. Original-text recommendation alignment then replays those edits as before.
There are no uppercase operations in recombination. The obsolete
`sentence_case.py` postprocessor and its isolated tests were subsequently removed;
the model's final pass is the final output.

Segmentation preserves punctuation and whitespace rather than normalizing them.
Model-predicted punctuation edits still apply. Newlines also delimit model units.
The small abbreviation list includes `etc.` and `atbp.`; ambiguous abbreviation
sentence endings remain unsplit. Terminal punctuation without following whitespace
is conservatively left in its surrounding unit. URL tokens remain intact.
One oversized sentence still raises the existing input-length error.
In particular, `U.S. Kumain ako.` remains one unit: initialism endings are
ambiguous, and an uppercase following token can also be a proper noun within the
same sentence. Fixed slots also mean model-created or removed sentence boundaries
do not trigger resegmentation during later passes.

Post-change checkpoint checks produced real CAPITAL tags for both sentence starts
in `ako ay umuwi. kumain ako.` with final five-pass output
`Ako ay umuwi. Kumain ako.`. First-pass outputs for the question/exclamation cases
were `Kumain ka ba? Oo.` and `Umalis siya! Bumalik siya.`. Newline and
already-capitalized checks also passed. Five-pass punctuation amplification remains
as documented above.

## Validation commands

Run from `backend` in PowerShell:

```powershell
$env:GECTOR_LOCAL_FILES_ONLY='true'
$env:FILLY_REAL_GEC_TESTS='1'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp=.pytest-capitalization-validation -q
```

Result: **68 passed**, including the opt-in real-checkpoint five-pass diagnostic;
one existing Starlette/httpx deprecation warning. A first run without `--basetemp`
had 67 passes and one fixture setup permission error in the system temporary
directory; the workspace-local temporary directory resolved it.

Run from `frontend`:

```powershell
node --test tests/*.test.js
npm.cmd run build
```

Results: **20 passed**; Vite production build succeeded (21 modules). `npm.cmd` was
used because this shell blocks the `npm.ps1` wrapper. No frontend source changed.

Changed source/test files in this task:

- `backend/app/services/gec.py`
- `backend/app/services/gec_checkpoint/segmentation.py`
- `backend/app/services/filly_pipeline.py`
- `backend/tests/test_gec.py`
- `backend/tests/test_gec_segmentation.py`
- `backend/tests/test_alignment.py`
- `backend/tests/test_api.py`
- This report.

Pre-existing uncommitted changes were preserved. No training, commit, or push was
performed.

Two parallel read-only reviews found no GEC application, offset, or recombination
blocker. The integration review identified the conservative initialism ambiguity
described above; no heuristic expansion or additional code remediation was needed.
