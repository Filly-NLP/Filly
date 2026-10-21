# FILLY Diverse Duplicate Training Preparation

## Executive Result

The historical 15-word bottleneck is removed in the experimental generator and prepared augmentation: **530 unique case-folded forms**, with 75% of duplicate rows outside the old vocabulary. This is a data-preparation result, not evidence of model improvement. No training was launched. One independent read-only review found no confirmed blocker.

## Original Generator

The registered GEG `$DELETE` / `duplicate_word` transformation uses `_duplicate`, limited to 15 `PROVISIONAL_FUNCTION_CONTEXTS`: ang, ng, mga, sa, at, na, ay, si, ni, kay, para, kung, dahil, nang, upang. It excludes first/last whitespace-token positions and tokens with attached punctuation. Historical TRAIN had 31,871 generated duplicates across Stage 2+3; the canonical Stage 3 TRAIN view contains 6,374. Existing registered behavior, version identifiers and historical artifacts remain unchanged.

Candidates preserve the existing inverse-generation contract: clean target ? one erroneous insertion ? registered correction tag ? replayable operation. Operations record target/source offsets and surfaces; candidate and experiment identities retain deterministic provenance.

## New Generator

`duplicate-word-v2-diverse` is explicitly selected through `generate_duplicate_candidates`, sharing the existing candidate/tag/operation machinery. It accepts Unicode letter/mark lexical cores, including content words and long forms, at first eligible, interior, and last eligible positions. It retains terminal punctuation after the second copy. It rejects punctuation-only, numbers, URL/email, markup/control forms, internal hyphens/apostrophes, and adjacent case-fold-equivalent clean forms. Hyphenated reduplications are excluded rather than reinterpreted. Each row adds exactly one copy.

The generator reads no corpus files and has no checkpoint/tokenizer dependency. GEC owns TRAIN sampling, schema construction, canonical labels and mixing. Function/content is an explicit proxy using the existing 28-item GEG FUNCTION_WORDS set; ?content? means an eligible lexical form outside that set, not certified POS or language annotation. Vocabulary comes from TRAIN clean targets. The explicit forward helper and existing inverse replay both validate exact reconstruction. A bounded 1,000-source feasibility audit found 28 function forms and 4,653 outside forms.

## Generated Duplicate Augmentation

| Measure | Count |
|---|---:|
| Rows | 1,280 |
| Unique clean source sentences | 1,280 |
| Unique duplicate forms (case-folded) | 530 |
| Function rows | 640 |
| Content proxy rows | 640 |
| Initial | 320 |
| Interior | 640 |
| Final | 320 |
| Short (1?3 alphabetic characters) | 454 |
| Medium (4?7) | 660 |
| Long (8+) | 166 |
| Old-15 vocabulary | 320 (25%) |
| Outside old-15 vocabulary | 960 (75%) |
| Unique source documents | 1,257 |
| Maximum rows for one case-folded form | 25 (1.953125%) |

The top 20 forms are listed below; ties are alphabetically ordered.

| Form | Rows |
|---|---:|
| ako | 25 |
| ang | 25 |
| at | 25 |
| dahil | 25 |
| din | 25 |
| isang | 25 |
| ito | 25 |
| iyon | 25 |
| kung | 25 |
| mga | 25 |
| na | 25 |
| ng | 25 |
| ni | 25 |
| nila | 25 |
| niya | 25 |
| para | 25 |
| raw | 25 |
| rin | 25 |
| sa | 25 |
| si | 25 |

The builder uses SHA-ranked TRAIN clean-source order, SHA-ranked candidates within each source, one row per clean source, six exact class/position quotas, a 25-row per-form cap, and an old-15 aggregate cap of 320. It inspected 38,548 source sentences / 737,710 candidates. One existing TRAIN pair collision was rejected and replaced; frequency/old-vocabulary caps rejected other candidates. No rejected collision remains in the artifact. Source document counts and every duplicated form count are in the machine-readable manifest/distribution CSV.

## Ordinary GEC Replay

**1,920 rows, 1,920 unique original source pair IDs, 38 observed tag/family strata, 0 identity rows.** Source: `Grammar-Error-Correction/defense_prereqs/gec_cpu/production-data/train/dataset2_stage3.parquet`, SHA-256 `53113e24278fd0bbaff66243c2308cf1e1b12741fee02aba8199718be6bc2ed3`.

Sampling excludes historical duplicate rows and all 119,000 identities. It first allocates up to 20 rows per observed stratum (all rows if rarer), then distributes remaining capacity in proportion to square-root stratum capacity with a 10% stratum cap and deterministic remainder allocation. Within each stratum it selects ascending SHA256(seed, stratum, pair_id), rejecting collisions. The achieved largest stratum is 79 rows. Rare punctuation-change strata retain their one available row. These are discovered categories; absent categories were not invented.

| Correction tag / family | Rows |
|---|---:|
| `$ADD_PUNC_EMARK :: punctuation` | 47 |
| `$ADD_PUNC_PERIOD :: punctuation` | 78 |
| `$ADD_PUNC_QMARK :: punctuation` | 51 |
| `$APPEND_t1 :: missing_word` | 79 |
| `$CHANGE_PUNC_EMARK :: punctuation` | 1 |
| `$CHANGE_PUNC_PERIOD :: punctuation` | 1 |
| `$CHANGE_PUNC_QMARK :: punctuation` | 43 |
| `$MERGE_HYPHEN :: hyphen` | 12 |
| `$MERGE_SPACE :: space` | 46 |
| `$REPLACE_daw :: enclitic` | 51 |
| `$REPLACE_din :: enclitic` | 56 |
| `$REPLACE_dito :: enclitic` | 39 |
| `$REPLACE_diyan :: enclitic` | 30 |
| `$REPLACE_doon :: enclitic` | 29 |
| `$REPLACE_nang :: ng_nang` | 64 |
| `$REPLACE_ng :: ng_nang` | 78 |
| `$REPLACE_nila :: pronoun` | 60 |
| `$REPLACE_niya :: pronoun` | 78 |
| `$REPLACE_raw :: enclitic` | 51 |
| `$REPLACE_rin :: enclitic` | 72 |
| `$REPLACE_rito :: enclitic` | 35 |
| `$REPLACE_riyan :: enclitic` | 31 |
| `$REPLACE_roon :: enclitic` | 26 |
| `$REPLACE_sila :: pronoun` | 54 |
| `$REPLACE_siya :: pronoun` | 74 |
| `$TRANSFORM_CASE_CAPITAL :: casing` | 79 |
| `$TRANSFORM_CASE_LOWER :: casing` | 79 |
| `$TRANSFORM_INSERT_HYPHEN :: hyphen` | 26 |
| `$TRANSFORM_SPLIT_HYPHEN :: hyphen` | 5 |
| `$TRANSFORM_SPLIT_SPACE :: space` | 4 |
| `$TRANSFORM_VERB_BASE :: morphology` | 39 |
| `$TRANSFORM_VERB_COMPACT :: morphology` | 79 |
| `$TRANSFORM_VERB_COMPOBJ :: morphology` | 79 |
| `$TRANSFORM_VERB_CONTACT :: morphology` | 78 |
| `$TRANSFORM_VERB_CONTOBJ :: morphology` | 78 |
| `$TRANSFORM_VERB_INCACT :: morphology` | 78 |
| `$TRANSFORM_VERB_INCOBJ :: morphology` | 78 |
| `$TRANSFORM_VERB_RECCOMP :: morphology` | 32 |

## Mixed Bounded Dataset

**3,200 total = 1,280 diversified duplicates (40%) + 1,920 ordinary replay (60%); seed 42.** The prior report's 3,200 total cap is retained; the newer explicit 40/60 instruction supersedes its unexecuted proposed component counts. This ratio is an experimental configuration, not an optimality claim.

Output: `Grammar-Error-Correction/outputs/duplicate_bounded_v1/duplicate_bounded_mixed_v1.parquet`.

SHA-256: `5f41490677888ecd074ef298c503cbcedbed0ca4e84a6087d67cbd2a633b8079`.

The Parquet Arrow schema exactly matches canonical Stage 3 TRAIN. All rows pass native GEC schema, alignment, metadata-compatibility and label replay. SHA256(seed, NUL, mixed-row, NUL, pair_id) ordering interleaves the components deterministically. Component Parquets, resolved configuration, native six-view descriptor, full distributions, source/input/output hashes and GEG worktree identity are retained. TEST descriptors remain sealed metadata.


A second complete seed-42 build in `outputs/duplicate_bounded_v1_reproducibility_check/` produced byte-identical duplicate, replay, mixed, distribution and resolved-config files. Manifest bytes may differ because they record output locations, commands and timing; row artifacts/order are identical.

| Artifact | SHA-256 |
|---|---|
| distribution | `5844f4f3d1fab38c8ac5fb019c431ebbd115d8b75c334db491f32068a6310b0d` |
| duplicates | `ee86f92d7aa9de530164e075e91a93581ed25d8a7095ecbc6098717c78632d6f` |
| mixed | `5f41490677888ecd074ef298c503cbcedbed0ca4e84a6087d67cbd2a633b8079` |
| replay | `be49b7ecfdbcee00554c9bebbf59ccaa5983d94ebc87eadc02b1ebaa402be5a8` |
| Mixed manifest | `e1f0093eae3234c2a03481299cdad1cacb451c8b054b7f02e24f4055ecca46e1` |
| Duplicate manifest | `3fa9bba09a6ccbdea4a7d016fe8cde9663b3207bdf878cdb82cf5b834bc32336` |

## Leakage Checks

| Check | Rows |
|---|---:|
| DEV in mix | 0 |
| TEST in mix | 0 |
| Diagnostic stress in mix | 0 |
| Manual examples intentionally used | 0 |

The builder pins the physical canonical TRAIN path and descriptor hash, validates every input split/stage, retains clean/document/source pair provenance, and verifies generated source pair IDs against TRAIN. Neither diagnostic nor manual examples were opened as augmentation input. TEST row files were not opened. DEV is used only by the non-training continuation preflight and future evaluation plan.

## Integrity

| Check | Failures remaining |
|---|---:|
| Forward/inverse operation replay | 0 |
| Canonical GEC alignment/reconstruction | 0 |
| Pair/row ID collisions | 0 |
| Source/target pair collisions | 0 |
| Duplicate erroneous-text collisions | 0 |
| Duplicate provenance collisions | 0 |

All 3,200 pair IDs and source/target pairs are unique. The source TRAIN hash before/after is identical. Fifty-one protected production app/frontend/checkpoint files also match their captured baseline hashes; see `diverse_duplicate_preparation/protected_validation.json`.

## Tests

From `Grammar-Error-Generator`:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_duplicate_word_v2.py tests/test_phase5_8.py tests/test_remediation_core.py tests/test_gec_dataset_contract.py -q -p no:cacheprovider
```

**35 passed, 6 subtests passed in 8.64s; exit 0.** The worker's full `pytest tests -q` attempt was interrupted after an early failure marker and has no complete result; it is not reported as a pass.

From `Grammar-Error-Correction`:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -B -m unittest tests.test_duplicate_mixer tests.test_experimental_stage3_continuation -v
```

**14 tests passed in 4.227s; exit 0.** Includes split rejection, per-form quota exhaustion, exact mix constants, seeded shuffle, schema read-back, IDs/provenance, hard update cap, and partial accumulation behavior using a tiny software model.

Existing native unittest suites for data tagging, tagged dataset, model training/inference and streaming evaluation: **36 passed in 0.492s; exit 0**. The exact discovery command is retained in `diverse_duplicate_preparation/root_validation.json`. GEC's venv lacks pytest; native unittest was used successfully. The broader GEC discovery (`$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B -m unittest discover -s tests -v`) ran 327 tests in 39.870s with one import error and 3 skips: untouched `tests/test_gate4_finalization.py:389` raises `SyntaxError: too many statically nested blocks`. No unrelated test/source was changed to mask it.

Scoped `git diff --check -- <task paths>` passed for all three touched repositories (exit 0); all eight edited/new Python source/test files have zero trailing-whitespace findings. Full results are in `final_git_state.json` and `scoped_whitespace_validation.json`. Dataset-backed preflight results are recorded below. The single [independent review](diverse_duplicate_preparation/independent_review.md) found no confirmed blocker; no recursive review cycle was performed.

The final dataset-backed continuation preflight **exited 0**, bound to mixed-manifest SHA `e1f0093eae3234c2a03481299cdad1cacb451c8b054b7f02e24f4055ecca46e1`. It admitted **3,200 TRAIN + 50,400 unchanged DEV rows with 0 quarantines**, fully audited both with the pinned tokenizer at cap 512, and found **0 truncations**. Maximum observed lengths are 220 TRAIN / 187 DEV. All TRAIN labels fit the frozen 1,913-label head. The unchanged DEV view has 39 edit-label occurrences outside that frozen head, handled by existing DEV masking semantics; no vocabulary was expanded. The preflight performs no real-model forward/backward pass; strict full state loading remains an execute-time check. The planned batch configuration is 400 micro-batches of 8, accumulated by 4, yielding 100 updates. See `diverse_duplicate_preparation/continuation_preflight_final_e1.json`.

An initial successful-validation preflight exited 1 while printing Unicode diagnostics under CP1252; output was corrected to an ASCII-safe summary and the final command exited 0. Earlier preflight reports are retained as evidence and are superseded by the final e1 report.

## Performance

A warm 1,000-sentence synthetic benchmark with alignment and an explicit empty in-memory resource context measured legacy 0.123s / 1,750 candidates versus v2 0.401s / 5,500 candidates: about 0.070 vs 0.073 ms per candidate. Historical default generation remains unchanged. Generation uses cached/indexed state and per-sentence token scans, with no corpus-wide scan per output row.

The real build took **50.50s**: TRAIN scan 15.87s, replay sampling 12.78s, diversified selection 17.86s, output validation 2.97s. A 250-source TRAIN benchmark without unnecessary candidate alignment took 0.089s for 4,950 candidates; selected pairs still undergo canonical validation. Timing measures preparation only.

## Next Training Command

The exact prepared invocation is retained in `diverse_duplicate_preparation/next_training_command.ps1`. **It was not executed.** The native trainer's Stage 2 initialization / same-data resume guards remain unchanged; a separate experimental entry point accepts the selected Stage 3 parent and fresh mixed manifest, fresh optimizer, fixed vocabulary, existing encoder/tokenizer and hard 100-update cap. Candidate output uses a fresh directory and is never named production `best.pt`.


```powershell
$env:PYTHONPATH='src'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe -B -m filly_gec.experimental_stage3_continuation `
  --manifest outputs\duplicate_bounded_v1\duplicate_bounded_mixed_v1_manifest.json `
  --parent-checkpoint ..\Filly\best.pt `
  --expected-parent-sha256 4928c916cc27e9a04d53263e50ced5473c4cbd17317fc23a2cde10aff87e4a09 `
  --tokenizer-cache "$env:USERPROFILE\.cache\huggingface\hub" `
  --cache-root outputs\duplicate_bounded_v1\continuation_tagged_cache `
  --output-dir ..\Filly\backend\reports\diverse_duplicate_preparation\candidate_stage3_mixed3200_seed42 `
  --report ..\Filly\backend\reports\diverse_duplicate_preparation\continuation_run_seed42.json `
  --seed 42 --max-updates 100 --micro-batch-size 8 `
  --gradient-accumulation-steps 4 --effective-batch-size 32 `
  --learning-rate 5e-6 --head-learning-rate 1e-4 `
  --precision bf16 --max-subword-tokens 512 --execute
```

## Candidate Acceptance Plan

Use official Stage 3 DEV, historical duplicate DEV, the frozen 100-case diagnostic suite, identity controls, and other actual DEV categories, comparing current checkpoint against candidate under identical frozen thresholds and decoder. Report duplicate precision/recall/F0.5, unseen-word recall, content-word recall, identity false positives, overall GEC F0.5/ERR, major-category regressions, raw paired counts and uncertainty. Keep token-slot and final-output scores separate and report frozen pass 1 and pass 5 separately.

Require meaningful improvement from zero unseen/content recall and materially improved overall diagnostic recall. Retain prior DEV protection gates: at least five-point duplicate recall improvement on the designated confirmation suite, duplicate precision >=95%, duplicate F0.5 loss <=1 point, overall GEC F0.5/ERR loss <=0.5 points, and no broad false deletion of intentional repetitions. Keep clean/document groups together; a small suite alone cannot establish release readiness. Do not require 10/10 manual cases. TEST remains sealed during selection. Full plan: `diverse_duplicate_preparation/candidate_acceptance_plan.md`.

## Files Changed

- GEG `src/geg/generators.py`: explicit versioned diverse candidate helper; historical route preserved.
- GEG `src/geg/alignment.py`: validated clean-to-erroneous replay.
- GEG `tests/test_duplicate_word_v2.py`: lexical/position/exclusion/determinism/replay tests.
- GEC `src/filly_gec/training/duplicate_mixer.py`: strict TRAIN scan, deterministic stratified selection, schema and integrity checks.
- GEC `scripts/build_duplicate_bounded_mixed.py`: experiment-only artifact/manifest CLI.
- GEC `tests/test_duplicate_mixer.py`: split, quotas, caps, mix and schema/provenance tests.
- GEC `src/filly_gec/experimental_stage3_continuation.py`: isolated explicit Stage 3 warm-start runner and non-training preflight.
- GEC `tests/test_experimental_stage3_continuation.py`: hard-cap and batching/token-cap tests.
- GEC `outputs/duplicate_bounded_v1/`: separate experiment Parquets, manifests, resolved config and distributions.
- FILLY `backend/reports/diverse_duplicate_preparation.md`: this integrated report.
- FILLY `backend/reports/diverse_duplicate_preparation/`: baseline/final git evidence, protection hashes, benchmark, tests, DEV acceptance plan, architecture notes, independent review and non-training preflight/next-command evidence.

These are the task's changes. Existing dirty production files and unrelated work were preserved. The explicit per-file paths, purposes, sizes and SHA-256 values (including generated cache files, reproducibility outputs and evidence) are listed in `diverse_duplicate_preparation/files_changed_manifest.json`. The failed-build orphan was removed after checking its absolute path and exact single-file contents; successful artifacts remain intact.

## Git State

Baseline branches/HEAD/status are captured in `diverse_duplicate_preparation/baseline_git_state.json`. All repositories are on `main`: FILLY `c86f4fd2f6813f675c175e32aa15149a18f50cc0`, GEC `c054c64c15d665d13f152bc495bd58ee84b1c1d1`, GEG `bae14dac955a6f22d84c32e2f35d0c6f4eeb69ab`. Full final `git status --short`, `git diff --stat`, and scoped `git diff --check` results for each touched repository are in `diverse_duplicate_preparation/final_git_state.json`; unrelated pre-existing dirty work is included for audit, not claimed as this task's edits.

## Safety Confirmation

Canonical Stage 3 TRAIN was not overwritten. No DEV, TEST or diagnostic examples were trained on; no real-model training was launched. Production `best.pt`, inference, thresholds, frontend and normalization were not modified. No duplicate postprocessor was added. Nothing was committed or pushed. No final TEST rows were opened.
