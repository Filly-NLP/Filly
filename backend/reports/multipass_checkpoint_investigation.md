# FILLY multi-pass and checkpoint investigation

## 1. `sentence_case.py` cleanup

The obsolete `backend/app/services/sentence_case.py` and its isolated
`test_sentence_capitalization.py` were removed. Repository search found no
production imports or other callers of `capitalize_sentence_starts`. The earlier
capitalization report was updated. The sentence-aware GEC tests, including the
real-checkpoint `$TRANSFORM_CASE_CAPITAL` test, remain. Production has no
sentence-capitalization postprocessor after GEC. `alignment.py` retains a generic
optional postprocessing trace interface, but `FillyPipeline` passes no such edits.

## 2. Multi-pass architecture

`FillyPipeline.analyze` normalizes, calls `GECService.correct_iteratively`, and
aligns the resulting pass trace. `gec.py:573-616` segments the normalized text
**once**, then executes five passes by default (`DEFAULT_ITERATIONS=5`, override
`GECTOR_ITERATIONS`). Each pass tokenizes every current sentence slot afresh,
batches up to 32 rows, obtains correction and detection logits, applies labels
against current sentence text, shifts sentence-local edits to current paragraph
offsets, and recombines current slots with the original exact gaps. The next pass
receives those output slots. The loop records all configured passes even when
every tag is `$KEEP`; there is no equality or cycle early stop. Empty input has
zero passes. `predictor.py` adds 0.2 to KEEP's correction probability and forces
KEEP when detection error probability is below 0.5. Both thresholds are runtime
constants. Each predicted operation is applied once per pass; no edit history is
remembered. Alignment replays the per-pass edits against the normalized text and
maps suggestions back to original offsets.

## 3. Repeated punctuation root cause

**P6 combination of P1 and P5.** The model genuinely predicts another
punctuation edit on each updated input. Five configured passes amplify the
result. The output text changes on every damaging pass, so lack of early stopping
on unchanged text is not the direct cause. The new punctuation is retokenized and
visible in the following pass. Older punctuation tokens receive `$KEEP`; the
newest terminal token receives a new `$ADD_PUNC_*` label. This rules out an edit
replay or reconstruction bug in these probes. No punctuation cleanup heuristic
was added. Probabilities below are raw correction-tag softmax / detection-head
error probability, not the KEEP-adjusted selection score.

| Input | Pass | Output | Non-KEEP tag | Probabilities |
| --- | ---: | --- | --- | --- |
| `Umalis siya!` | 1 | `Umalis siya!!` | `$ADD_PUNC_EMARK` | .804343 / .935903 |
| | 2 | `Umalis siya!!!` | `$ADD_PUNC_EMARK` | .944398 / .982434 |
| | 3 | `Umalis siya!!!!` | `$ADD_PUNC_EMARK` | .946058 / .972739 |
| | 4 | `Umalis siya!!!!!` | `$ADD_PUNC_EMARK` | .950807 / .968891 |
| | 5 | `Umalis siya!!!!!!` | `$ADD_PUNC_EMARK` | .952839 / .964790 |
| `Kumain ka ba?` | 1 | `Kumain ka ba??` | `$ADD_PUNC_QMARK` | .444491 / .868248 |
| | 2 | `Kumain ka ba??!` | `$ADD_PUNC_EMARK` | .628656 / .932498 |
| | 3 | `Kumain ka ba??!!` | `$ADD_PUNC_EMARK` | .908448 / .965386 |
| | 4 | `Kumain ka ba??!!!` | `$ADD_PUNC_EMARK` | .910505 / .968268 |
| | 5 | `Kumain ka ba??!!!!` | `$ADD_PUNC_EMARK` | .894821 / .962901 |
| `Oo.` | 1 | `Oo?` | `$CHANGE_PUNC_QMARK` | .606967 / .933632 |
| | 2 | `Oo?!` | `$ADD_PUNC_EMARK` | .716694 / .916457 |
| | 3 | `Oo?!!` | `$ADD_PUNC_EMARK` | .951679 / .991122 |
| | 4 | `Oo?!!!` | `$ADD_PUNC_EMARK` | .953911 / .986906 |
| | 5 | `Oo?!!!!` | `$ADD_PUNC_EMARK` | .952810 / .977824 |

`Ako ay umuwi.` was an unchanged control through all five passes. These are
actual labels in the deployed 892-class vocabulary; the three probes do not
establish whether other punctuation tags would behave the same way.

## 4. Pass statistics

In the four-input diagnostic set, **3/4 changed on each of passes 1–5**; the
control changed on none. **3/4** gained same-character repeated punctuation by
pass 5: `Umalis siya!` and `Kumain ka ba?` first on pass 1, `Oo.` first on pass
3. The first two already degrade at pass 1, so reducing five passes alone does
not solve the observed issue. Later passes increase the repeated marks.

## 5. Fixed sentence boundaries

**B2: real BOS-sensitive limitation.** `segment_sentences` is called outside the
pass loop (`gec.py:581-583`); the loop reuses sentence slots and exact gaps.
Current slots are retokenized, but a new `.`, `?`, or `!` never creates a new
model row in that request. This was a conservative implementation for lossless
sentence batching; repository evidence does not establish a historical primary
motivation. Fixed slots simplify offset calculation, but alignment does not
require them.

Real-checkpoint controlled example: `kumain ka ba pa rin` initially has one
model row. Pass 1 labels `pa` and `rin` `$ADD_PUNC_QMARK`, yielding
`kumain ka ba pa? rin?`. Fresh segmentation of that output gives two units,
`kumain ka ba pa?` and `rin?`. Actual pass 2 still has one model row,
`('kumain','ka','ba','pa','?','rin','?')`, one `<s>` BOS token, and `$KEEP` on
`rin`; it produces `Kumain ka ba pa? rin?`. Calling fresh single-pass inference
on `kumain ka ba pa? rin?` makes `rin` a new row, predicts
`$TRANSFORM_CASE_CAPITAL`, and produces `Kumain ka ba pa? Rin?`. This is a
concrete capitalization correction missed by fixed boundaries. End-of-unit
period, question-mark, and exclamation-mark insertions were also observed;
only the question-mark example above introduced an interior boundary in the
sample. Other BOS-sensitive corrections are possible but unmeasured.

## 6. BOS and capitalization impact

Yes: a boundary introduced on an earlier pass can leave the next lexical token
interior to its original row, preventing the observed capitalization prediction.
The earlier real-checkpoint sentence-aware test still proves that an *initial*
second sentence receives `$TRANSFORM_CASE_CAPITAL` when it is segmented before
pass 1. No deterministic capitalization stage repairs a missed model edit.

## 7. Alignment impact

Suggestion spans originate in original text. Each GEC pass's changes use offsets
in that pass's full current paragraph; alignment replays them sequentially from
normalized text and maps final suggestions to original and final output spans.
Dynamic resegmentation could preserve exact mapping by segmenting each current
paragraph, calculating that pass's exact local-to-global starts, applying edits,
and replaying the resulting global edits. Fixed sentence IDs need not persist.
Raw tags and iteration numbers must stay attached to each pass's global edits.
Approximate substring matching is unnecessary. Dynamic grouping would change
model context, increase segmentation work, and could oscillate after boundary
creation/deletion; batching can still group all current sentence rows per pass.

## 8. `best.pt` plug-and-play verdict

**C6, combining C3, C4, and a bounded C5 risk.** A better checkpoint with the
same pinned encoder/tokenizer, compatible architecture, valid embedded labels
and executable tags can replace `best.pt` through `GECTOR_MODEL_PATH` or the
default filename. A generally more complete FILLY checkpoint cannot be assumed
to work by swapping only the `.pt` file.

| Future checkpoint | `.pt` replacement under current runtime |
| --- | --- |
| A. Same architecture, exact labels/order and tokenizer | Yes, if format/metadata and pinned encoder revision match. |
| B. More labels | Structurally possible: head size follows embedded vocabulary. Every tag must be executable and the saved head must match. |
| C. Reordered labels | Safe only when weights and embedded mapping were trained/packaged together. Mapping-only reorder can silently misdecode. |
| D. More verb/morphology transformations | Existing encoded verb prefixes can work; a new operation family needs a runtime handler. |
| E. Different base model/tokenizer | No; hardcoded identity/revision and shape checks require code/config changes. |
| F. Better weights, otherwise identical contract | Yes, with external pinned assets available. |

## 9. Checkpoint dependency map

`best.pt` is a 4,018,471,291-byte format-1 training checkpoint. Top-level keys:
`format_version`, `config`, `metadata`, `hashes`, `label_vocab`, `model_state`,
`optimizer_state`, `rng_state`, `scaler_state`, `scheduler_state`,
`training_state`. Its 395-tensor `model_state` contains the encoder and heads:
correction weight `(892, 1024)` and detection weight `(2, 1024)`.
`label_vocab` embeds all 892 indexed labels (`$KEEP=0`) and a checked hash.
Model ID/revision, stage, run metadata, training config and hashes are embedded.
Tokenizer files and Hugging Face AutoConfig are **external**, loaded from the
pinned `jcblaise/roberta-tagalog-large` revision
`acb6b204dfb1afdd7476eae5da234cbcf8899846`; offline startup needs that
snapshot cached. Transform handlers, punctuation names, verb registry, KEEP
bonus 0.2, detection threshold 0.5, and pass count are runtime code/config.
The loader requires Transformers 5.16.1. The filename is configurable via
`GECTOR_MODEL_PATH` or explicit `GECService(checkpoint_path=...)`; no directory
scan or deployment manifest exists. A 4 GB copy was unnecessary for validation.

## 10. Silent-failure risks

Ordinary missing files, stale label hashes, label/head dimension mismatch,
wrong base identity and incompatible tensor shapes fail loudly. An in-memory
test swapped two non-KEEP label IDs, recomputed the checkpoint's internal
hashes, and left classifier rows unchanged: `_validate_checkpoint` accepted
that semantically mismatched artifact. Thus the hash proves internal consistency,
not that a classifier column was trained for its declared label. A same-size
tokenizer ID permutation could similarly change meaning if metadata falsely
claimed the pinned revision. The loader does not prevalidate every label's
transformation handler; unknown tags can fail only when predicted.

## 11. Future deployment recommendation

Treat future models as **versioned checkpoint bundles**: `.pt` with its embedded
mapping plus the matching encoder config/tokenizer snapshot and a compact
manifest of checkpoint format, encoder/tokenizer identity and special-token
fingerprint, label hash, and supported transform taxonomy. The current narrow
contract supports a `.pt` plus fixed compatible runtime, but not arbitrary
single-file replacement. Keep `GECTOR_MODEL_PATH`; no filename migration is
needed. Startup validation of executable tags and the existing tokenizer
revision metadata would be a small later improvement. Neither can prove that
manually relabeled classifier columns preserve semantics.

## 12. Recommended remediation

Punctuation: document the model limitation and evaluate model/data or a
checkpoint-specific pass-count choice separately. An unchanged-text stop would
save calls for stable inputs but does not fix these changing outputs. A cycle
guard would not trigger on these monotonic additions. Do not hide them with
duplicate-punctuation cleanup. Boundaries: consider dynamic resegmentation in
a separately reviewed change, preserving paragraph-level pass traces and batch
calls; the observed capitalization miss justifies an experiment, but changing
model contexts across all passes warrants targeted outcome testing. Checkpoint:
add startup tag/tokenizer compatibility checks when preparing a future model
bundle. No speculative production inference or loader changes were made here.

## 13. Files changed

Removed `backend/app/services/sentence_case.py` and
`backend/tests/test_sentence_capitalization.py` (both were untracked when work
began). Updated `backend/reports/capitalization_investigation.md`. Added two
diagnostic tests to `backend/tests/test_gec.py`. Added this report. Other
pre-existing worktree changes were preserved.

## 14. Tests

From `backend`: `.\\.venv\\Scripts\\python.exe -B -m pytest -p no:cacheprovider
--basetemp=.pytest-sentence-case-cleanup -q tests/test_gec.py
tests/test_gec_segmentation.py tests/test_alignment.py tests/test_api.py`:
**41 passed, 1 skipped** immediately after cleanup. The same focused set with
`--basetemp=.pytest-gec-diagnostics` after the new tests: **43 passed, 1
skipped**. With `GECTOR_LOCAL_FILES_ONLY=true` and `FILLY_REAL_GEC_TESTS=1`,
`.\\.venv\\Scripts\\python.exe -B -m pytest -p no:cacheprovider
--basetemp=.pytest-final-backend -q`: **64 passed**. One pre-existing
Starlette/httpx deprecation warning appeared. No frontend behavior or API
contract changed; frontend tests/build were not rerun.

## 15. Reviewer findings

Three read-only reviews completed. Multi-pass review independently reproduced
fresh punctuation predictions and rejected replay/retokenization causes; it
classified P1 plus later-pass amplification P5. Segmentation review confirmed
fixed slots and that exact alignment can support a dynamically segmented pass;
the review's mocked test verifies the architectural behavior, while the direct
real-checkpoint probe above establishes the concrete miss. Checkpoint review
confirmed embedded labels, external pinned assets, explicit handlers, and the
semantic mapping risk; it rejected generic single-file plug-and-play. A reviewer
summary initially placed `Oo.`'s first duplicate on pass 2; the actual trace
shows pass 3 (`Oo?!!`).

## 16. Remaining limitations

Model: terminal punctuation predictions repeat on the tested inputs. Inference:
five passes always run, with no unchanged-text/cycle early stop. Segmentation:
new boundaries do not create BOS rows until a new request. Alignment: current
offset replay is exact, but dynamic grouping would need the same exact
per-pass global trace and fresh behavior tests. Checkpoint: compatible weights
can replace the artifact, while new encoder/tokenizer identities or operation
families require runtime changes; an internally rehashed but semantically wrong
label mapping cannot be detected from the artifact alone.
