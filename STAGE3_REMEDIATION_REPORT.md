# FILLY Stage 3 remediation

**FILLY STAGE 3 REMEDIATION PASSED** — 2026-10-01 (Asia/Taipei).

The current final Stage 3 `best.pt` now passes FILLY's checkpoint validation and
runs through the real backend and frontend proxy. Both read-only reviewers
returned PASS after one bounded remediation pass.

## Checkpoint identity and provenance

| Field | Verified value |
| --- | --- |
| File | `Filly/best.pt` |
| Size | 4,031,071,803 bytes |
| SHA-256 | `4928c916cc27e9a04d53263e50ced5473c4cbd17317fc23a2cde10aff87e4a09` |
| Provenance finding | **VERIFIED STAGE 3** |
| Run ID | `run` |
| Encoder | `jcblaise/roberta-tagalog-large` |
| Encoder revision | `acb6b204dfb1afdd7476eae5da234cbcf8899846` |
| Vocabulary count | 1,913; contiguous IDs 0–1912; `$KEEP` at 0 |
| Vocabulary scope | `train_stage2_and_stage3_union` in config and metadata |
| Vocabulary SHA-256 | `f9f585732da3bf7a9a8206ebf0a7edfde48068cef3364967795692ca99cf7664` |
| Correction classifier | Weight `(1913, 1024)`; bias `(1913,)` |
| Detection classifier | Weight `(2, 1024)`; bias `(2,)` |
| Parent checkpoint SHA-256 | `cbdb1e0e06f15e642c585f10ce90a9582e86b995f58f5eec52072c24ffba5c89` |
| Source manifest SHA-256 | `5a07eb31e98ce5c8b0630640d78c3a3480e18c8ab100e7bd32d9fdc416b6d772` |

The restricted, memory-mapped checkpoint inspection established `stage3`,
`selected_best=true`, selection by `dev_correction_f0_5`, best epoch 2, and global
step 14,700. Its config records Stage 3 train/dev views, 235,200 training rows,
50,400 development rows, three maximum epochs, encoder/head learning rates
`5e-6`/`1e-4`, and initialization from the `S2-ENC-LOW` Stage 2 trial. The saved
continuation directory identifies
`g4-r3-c054c64-v2-stage3-from-s2-enc-low-full-v1/stage3/run`.

The parent record binds the selected Stage 2 checkpoint, vocabulary, source
manifest, and pinned encoder. Reproducibility metadata identifies
`filly_hpo_stage3_continuation` and `synthetic_test_accessed=false`. These fields
match the actual GEC trainer and vocabulary-building implementation, and the
checkpoint digest matches the supplied audit identity. The checkpoint digest
and size were checked again after remediation and remained unchanged.

Matching external Stage 3 run manifests and training sidecars are unavailable
in the local GEC outputs directory. Their hashes cannot be independently bound
to external files here; provenance verification uses the supplied digest,
embedded training/parent evidence, and training-code contract.

## Previous rejection and implemented compatibility rules

FILLY hardcoded `vocabulary_support_scope == train_only`; its old regression
asserted an 892-class developmental Stage 2 checkpoint. The narrow Stage 2
confirmation workflow produces that scope. Production Stage 2 instead builds
the vocabulary from both physical Stage 2 and Stage 3 TRAIN views; Stage 3
reuses that frozen union. The prior universal scope check was stale.

| Stage | Accepted config scope | Additional requirements |
| --- | --- | --- |
| Stage 2 legacy | `train_only` | Matching stage; any declared metadata scope must match |
| Stage 2 production | `train_stage2_and_stage3_union` | Matching stage; any declared metadata scope must match |
| Stage 3 | `train_stage2_and_stage3_union` | Scope required in both config and metadata; complete, consistent Stage 2 parent identity |

Unknown stages/scopes, Stage 3 `train_only`, contradictory scope metadata, and
Stage 2 checkpoints declaring Stage 3 parent lineage are rejected. Stage 3's
config hash values for vocabulary, source manifest, and parent checkpoint must
exist and exactly match the validated identities; absent, null, and mismatched
values are rejected. Review found the initial null-value loophole, which was
closed in the single permitted remediation pass.

Validation also requires format version 1, selected-best metadata, the pinned
encoder identity/revision, integer contiguous label IDs with `$KEEP` at zero,
the exact ordered vocabulary hash in checkpoint hashes and metadata, source
manifest consistency, valid executable label forms, and both classifier weight
and bias dimensions. Runtime retains restricted unpickling, tokenizer/config
snapshot verification, architecture checks, and strict model-state loading.
No checkpoint hash was hardcoded into the production validator, and no fallback
or ignored compatibility exception was introduced.

## Label compatibility and faithful replay port

All 1,913 labels passed executable-form validation. All are classifier outputs;
no metadata-only output classes were found. The vocabulary contains:

- 539 replacements, including 16 fixed enclitic/ng–nang/pronoun replacements;
  32 appends; `$KEEP`; and deletion.
- Six punctuation-add classes (three ordinary, three gap-qualified), three
  punctuation-change classes, and two casing classes.
- Hyphen/space merge, split, and insertion classes.
- Eight concrete Filipino verb morphology families, totaling 1,323 classes.

Six classes were missing from FILLY's earlier replay implementation:

```text
$ADD_PUNC_EMARK@gap:%20,
$ADD_PUNC_PERIOD@gap:%20,
$ADD_PUNC_QMARK@gap:%20,
$REPLACE_Ang%20%E2%80%9C@gaps:,%20,
$REPLACE_Kung%20%E2%80%9C@gaps:,%20,
$REPLACE_Sa%20%E2%80%9C@gaps:,%20,
```

Their exact raw-gap behavior was ported from the current GEC transforms.
Replay consumes the required raw gaps, composes with following DELETE labels,
preserves untouched slices, enforces first-token replacement-gap placement,
and rejects overlapping patches. Replay and API change records share the same
patch generation, so edit spans and suggestions describe the actual output.
No new linguistic correction behavior was invented.

## Changed files and preservation

Changes relative to the starting working tree are confined to:

- `backend/app/services/gec.py`: stage-aware compatibility validation and shared
  exact change-record generation.
- `backend/app/services/gec_checkpoint/transforms.py`: raw-gap replay and label
  form validation.
- `backend/tests/test_gec.py`: small checkpoint fixtures, compatibility rejection
  tests, six-label replay/trace tests, and optional real-checkpoint diagnostics.

This report accompanies the Stage 3 commit; `.stage3-remediation/` audit evidence
remains untracked. The audit
directory holds the original diff/status/hash manifests, lightweight checkpoint
evidence, test logs, real runtime responses, and local process helpers.

Starting and final branch/HEAD for the initial remediation: `main` /
`503d5b4d57efd16afa5c9b5a42eb343f878b66fa`. All 28 pre-existing modified/untracked
file hashes still match. Across 132 baseline-tracked files, only the three
files above changed. Existing frontend, normalizer, alignment, pipeline, API
test, and bytecode modifications were preserved byte-for-byte. The tracked
database was preserved by using an audit-local runtime database. The initial
remediation performed no commit, stash, reset, clean, or unrelated cleanup. Neither parent
tool-defense Markdown file nor GEG/GEC source/artifacts were edited.

## Tests and real runtime acceptance

| Check | Final result |
| --- | --- |
| Focused GEC tests | 61 passed, 2 optional diagnostics skipped |
| Entire existing backend test directory | 108 passed, 2 skipped, 15.72 seconds |
| Frontend Node tests | 20 passed |
| Changed-file `git diff --check` | Passed |
| All actual vocabulary label forms | 1,913 accepted; zero invalid |
| Reviewer A: model/checkpoint compatibility | PASS |
| Reviewer B: preservation/regression/runtime safety | PASS |

Unit fixtures cover legitimate Stage 3 and retained Stage 2 scopes, unknown
scope/stage, label count/head mismatch, same-count label identity/order mutation,
encoder mismatch, malformed format/metadata/labels, parent binding, null hash
rejection, and inference initialization. Replay tests cover all six new label
variants, deletion composition, exact offsets/confidence, and pipeline suggestion
mapping. The two skipped optional diagnostics are gated by
`FILLY_REAL_GEC_TESTS`; real startup and API acceptance were run separately.
The suite emitted one existing Starlette `httpx` deprecation warning.

The final real backend command was:

```text
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

It used `backend/.venv` (PyTorch `2.14.0+cpu`, Transformers `5.16.1`), the
absolute current `best.pt` path, offline pinned Hugging Face assets,
`GECTOR_LOCAL_FILES_ONLY=true`, `HF_HUB_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`,
CPU, and four OMP/MKL threads. Uvicorn reached ready in **9.28 seconds** on the
final revision. Logs confirm the unchanged production normalizer bundle loaded
304 rules and 18,077 vocabulary entries, the GEC model loaded from the current
absolute checkpoint path, and application initialization completed.

`GET /api/v1/health` returned HTTP 200 with `ready=true`, CPU, and five iterations.
The real `POST /api/v1/analyze` result was:

| Field | Value |
| --- | --- |
| Input | `aq nman ay masaya kc kasama kita.` |
| Normalized text | `ako naman ay masaya ka kasama kita.` |
| Final text | `Ako naman ay masaya ka kasama kita.` |
| GEC iterations | 5; all passes chained; normalization output is pass 1 input |
| HTTP status | 200 |
| Direct API duration | 0.824 seconds |

This is runtime acceptance; grammatical accuracy was not evaluated. API metadata
confirms Stage 3, run `run`, union scope, 1,913 labels, the observed vocabulary
hash, and the absolute current checkpoint path. No old Stage 2 checkpoint was
used.

The frontend ran concurrently with:

```text
npm run dev -- --host localhost --port 5173 --strictPort
```

The page and served API module returned HTTP 200. FILLY's actual frontend API
client was invoked through the live Vite proxy: proxied health and correction
requests returned HTTP 200 with the same Stage 3 output and five-pass trace;
correction took 0.694 seconds. This verifies HTTP/client/proxy integration;
browser UI clicks were not exercised. IPv4-first Node DNS resolution was used
for the local proxy; no frontend code change was needed.

Both final processes were stopped through their Windows console process groups.
Uvicorn logged `Application shutdown complete`; the npm batch termination prompt
was completed. Both supervisors exited and neither port 8000 nor 5173 remained
listening. Signal-related child exit codes are retained in the process records.

## Remaining limitations

Source-dependent transform preconditions remain fail-closed, as in GEC: merge
neighbors/deletions, valid hyphen/split surfaces, punctuation sources, and
first-slot replacement gaps must fit the prediction context. This remediation
does not alter model predictions, the existing five-pass policy, sentence-slot
segmentation, or accuracy characteristics. Acceptance ran on CPU; CUDA was
unavailable. External training sidecars remain unavailable locally, and the API
retains `readiness_status=UNASSESSED` with unknown dataset/performance-status
fields. Runtime acceptance does not certify research performance.

**The current final Stage 3 checkpoint is now the checkpoint actually used by
FILLY.**

## Subsequent origin reconciliation and requested commit

At the user's subsequent request, `main` was fast-forwarded to origin commit
`772a0c2` (`feat: improved normalization`) before creating `feat: Stage 3 GEC`.
The incoming commit does not overlap the three Stage 3 source/test files. Its
normalizer metadata changes were composed with the pending local
`base_vocabulary.sha256` correction, retaining both incoming schema/curated
metadata and the local value. Every other pending local file remained
byte-for-byte unchanged during reconciliation.

After incorporating origin, the entire backend suite plus the normalizer CLI
tests passed: **117 passed, 2 optional skips**, with the same existing Starlette
deprecation warning, in 17.80 seconds. All **20 frontend tests** also passed.
The backend API tests use the actual application lifecycle and current Stage 3
checkpoint. The earlier standalone Uvicorn timings and sample normalization
above describe the initial remediation revision, before origin's normalization
enhancements.

The requested Stage 3 commit contains only the three backend implementation/test
files listed above and this report. Pending frontend, alignment/pipeline, API
test, normalizer artifact, and bytecode changes remain outside that commit.
The audit backups/logs remain local and untracked. No push was requested.
