# Independent Read-only Review

## Result

No confirmed blocker remains in the bounded preparation. The new generator and 40/60 artifact meet the reviewed provenance, diversity, replay, schema, and checkpoint-compatibility requirements. This review did not launch training.

## Generator and source provenance

The historical registered duplicate path remains unchanged: it uses the 15 provisional function-word contexts and excludes first and last whitespace-token positions. The separately versioned `duplicate-word-v2-diverse` helper accepts safe Unicode letter/mark tokens, records the duplicated surface, class, position, offsets, and generator version, and preserves the existing `$DELETE` correction tag and replay contract.

The v2 eligibility rules reject punctuation-only, numeric, markup, URL/email, and internal-hyphen/apostrophe forms. The generator checks case-folded adjacent forms in the clean source and skips those insertion sites; tests also cover lexicalized reduplication. Focused tests exercise function and content forms, initial/interior/final positions, long words, deterministic selection, forward and correction replay, and filtering.

The builder pins the canonical Stage 3 TRAIN path and source descriptor hash before scanning. Its public duplicate sampler rejects `dev`, `synthetic_test`, and non-Stage-3 provenance; its replay sampler rejects both non-TRAIN splits. The generated rows retain the source TRAIN pair, clean ID, and document ID. I independently checked each generated duplicate's source pair and clean target against canonical Stage 3 TRAIN, and verified its forward operation reconstructs the erroneous text.

## Dataset review

The published mixed artifact contains 3,200 rows: 1,280 duplicates and 1,920 ordinary replay rows. It has 530 duplicated forms, including 640 content-class and 640 function-class rows. Positions are exactly 320 initial, 640 interior, and 320 final. The longest length bucket has 166 rows; the middle and short buckets have 660 and 454. Each form is capped at 25 rows. The historical 15-form vocabulary accounts for 320 rows (25%); 960 rows use other forms. The 1,280 duplicate sources span 1,257 documents.

Replay sampling covers 38 observed tag/family strata, excludes all 119,000 identity rows, and has a maximum of 79 rows in any stratum. It includes the available punctuation, capitalization, spacing, ng/nang, enclitic, pronoun, morphology, and other registered correction categories. The algorithm applies a minimum allocation where capacity permits, then uses square-root-capacity allocation with a 10% stratum cap and stable SHA-256 ranks.

The mixed ordering matches the recorded seed-42 SHA sort; duplicate rows are distributed throughout the artifact rather than stored as a component block. I independently checked all 3,200 pair IDs, the row ordering, duplicate TRAIN provenance, forward replay, and absence of collisions with original Stage 3 TRAIN source/pair text. Manifest counts report zero DEV, TEST, diagnostic-stress, or manual-example rows; no DEV or TEST row file was opened during this review.

Canonical alignment and operation replay failures are both zero. The output schema matches Stage 3 TRAIN. The canonical source artifact still hashes to `53113e24278fd0bbaff66243c2308cf1e1b12741fee02aba8199718be6bc2ed3`; the selected production checkpoint still hashes to `4928c916cc27e9a04d53263e50ced5473c4cbd17317fc23a2cde10aff87e4a09`.

The mixed artifact SHA-256 is `5f41490677888ecd074ef298c503cbcedbed0ca4e84a6087d67cbd2a633b8079`. Its manifest SHA-256 is `5f0a053a32ea799fe91c4aea152c36de55f9b64b6753517bd0f8bd7c3cbd699f`.

## Runner compatibility

The separate experimental Stage 3 continuation runner pins the parent checkpoint hash and validates the Stage 3/selected-best metadata, model and tokenizer revision, frozen label-vocabulary hash, label count, and task-head tensor shapes. I ran its parent validation read-only against the current `Filly/best.pt`: it passed with 1,913 labels, model revision `acb6b204dfb1afdd7476eae5da234cbcf8899846`, correction head shape `(1913, 1024)`, and detection head shape `(2, 1024)`. The runner uses that same pinned tokenizer revision and rejects unsupported labels before training.

The default 3,200-row plan uses micro-batches of 8 and accumulation of 4, yielding exactly 100 optimizer updates. The runner enforces a hard maximum of 100 and requires a fresh candidate output directory. The native Stage 2-to-Stage 3 path remains separate. No tokenizer preflight or real-model forward/backward pass was run as part of this review.

## Verification

- GEG focused suite: 35 tests passed, with 6 passing subtests, as recorded in `root_validation.json`.
- GEC mixer and experimental-runner suites: 14 tests passed, as recorded in `root_validation.json`.
- I independently reran `test_duplicate_mixer.py`: all 9 tests passed.
- Bounded generation and build: 50.50 seconds total, as recorded in the artifact manifest. The 1,000-sentence benchmark reports about 0.073 ms per v2 candidate, compared with about 0.070 ms per legacy candidate.
- Read-only integrity checks confirmed exact seeded row ordering, canonical TRAIN and production checkpoint hashes, and zero replay/alignment/collision failures.

The remaining step is the separately authorized training decision and, before any optimization, the runner's non-training tokenizer/data preflight. The review does not authorize training or checkpoint promotion.
