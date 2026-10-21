# FILLY Bounded Duplicate Fine-Tuning — Colab Preparation

## Inputs

The finalized inputs are frozen. Exact absolute paths, filenames, byte sizes, row counts and SHA-256 values are listed in the input table below and in `Grammar-Error-Correction/experiments/duplicate_bounded_colab_v1/local_input_inventory.json`. No data was regenerated, resampled or retuned.

| Artifact | Absolute local path | Rows | Bytes | SHA-256 | Colab/Drive destination |
|---|---|---:|---:|---|---|
| BOUNDED_TRAIN | `C:\Users\admin\Documents\College\4th Year\random\FILLY_Tool\Grammar-Error-Correction\outputs\duplicate_bounded_v1\duplicate_bounded_mixed_v1.parquet` | 3200 | 779183 | `5f41490677888ecd074ef298c503cbcedbed0ca4e84a6087d67cbd2a633b8079` | bundle/inputs/duplicate_bounded_mixed_v1.parquet |
| STAGE3_DEV | `C:\Users\admin\Documents\College\4th Year\random\FILLY_Tool\Grammar-Error-Correction\defense_prereqs\gec_cpu\production-data\dev\dataset2_stage3.parquet` | 50400 | 9346846 | `5a20d0e81c60791f537207635e74d89b1a57c49e371d65d67ab141ab19ef353e` | bundle/inputs/dataset2_stage3.parquet |
| PARENT_CHECKPOINT | `C:\Users\admin\Documents\College\4th Year\random\FILLY_Tool\Filly\best.pt` | ? | 4031071803 | `4928c916cc27e9a04d53263e50ced5473c4cbd17317fc23a2cde10aff87e4a09` | historical verified Drive parent or inputs/parent_stage3_best.pt |
| FINAL_MANIFEST | `C:\Users\admin\Documents\College\4th Year\random\FILLY_Tool\Grammar-Error-Correction\outputs\duplicate_bounded_v1\duplicate_bounded_mixed_v1_manifest.json` | ? | 67623 | `e1f0093eae3234c2a03481299cdad1cacb451c8b054b7f02e24f4055ecca46e1` | bundle/inputs/duplicate_bounded_mixed_v1_manifest.json |
| MIXED_CONFIG | `C:\Users\admin\Documents\College\4th Year\random\FILLY_Tool\Grammar-Error-Correction\outputs\duplicate_bounded_v1\duplicate_bounded_mixed_v1_resolved_config.json` | ? | 655 | `38760d53fdad5f1ddaeec2070c3fde64da7549a1fb94bf87440a1125911c0ab2` | bundle/inputs/duplicate_bounded_mixed_v1_resolved_config.json |
| FINAL_PREFLIGHT | `C:\Users\admin\Documents\College\4th Year\random\FILLY_Tool\Filly\backend\reports\diverse_duplicate_preparation\continuation_preflight_final_e1.json` | ? | 136297 | `a8f4f462d6baf62f971bef85ccfdbf0c1ac63b2d1954dd5e0e857a32333d9173` | bundle/inputs/continuation_preflight_final_e1.json |
| DIAGNOSTIC_CASES | `C:\Users\admin\Documents\College\4th Year\random\FILLY_Tool\Filly\backend\reports\duplicate_inference_stress_cases.jsonl` | 120 | 297998 | `ae6680ea78851303ac1d9c3fd0ac158d5c89408f734d549e3aac48dafb50be16` | local post-Colab evaluation only |
| DIAGNOSTIC_MANIFEST | `C:\Users\admin\Documents\College\4th Year\random\FILLY_Tool\Filly\backend\reports\duplicate_inference_stress_manifest.json` | ? | 2775 | `4a37b407ee28ba3b796c0e1621187a50fae95d993150cc01285500b2ddc80097` | local post-Colab evaluation only |


The small upload archive contains the exact GEC source/config worktree, mixed TRAIN, finalized manifest/config, final local preflight, unchanged DEV and pinned tokenizer/config assets. The 4 GB parent is excluded so an existing matching Drive copy can be reused. No full Stage 3 TRAIN or final TEST file is bundled.

## Parent Checkpoint

`Filly/best.pt`, SHA `4928c916cc27e9a04d53263e50ced5473c4cbd17317fc23a2cde10aff87e4a09`, is selected Stage 3. Architecture: GECToRTagalogLarge/RoBERTa, hidden dimension 1024. Encoder/tokenizer: `jcblaise/roberta-tagalog-large`, revision `acb6b204dfb1afdd7476eae5da234cbcf8899846`. Correction head: 1,913×1024; detection head: 2×1024. Frozen vocabulary SHA: `f9f585732da3bf7a9a8206ebf0a7edfde48068cef3364967795692ca99cf7664`.

Local hash, metadata, head dimensions and tokenizer/config preflight pass. Strict full weight loading and a finite-loss zero-update forward are prepared for Colab; neither was executed locally in this phase.

Parent configuration records this historical Drive location:

`/content/drive/MyDrive/FILLY_GEC/hpo/stage2/colab-c054c64-v2/durable-stage3/g4-r3-c054c64-v2-stage3-from-s2-enc-low-full-v1/stage3/run/best.pt`

Remote existence/content is not assumed. Colab checks its SHA, or uses an explicit upload of the local Stage 3 parent as `inputs/parent_stage3_best.pt`. No Stage 2/base substitution is allowed.

## Colab Experiment

- Experiment ID: `duplicate-bounded-mixed-v1-seed42`.
- Drive root: `/content/drive/MyDrive/FILLY_GEC/experiments/duplicate-bounded-mixed-v1-seed42/`.
- Each attempt: `runs/run-<UTC>-<UUID>/`, with no automatic resume or overwrite.
- Seed 42; effective batch 32 = micro 8 × accumulation 4. 3,200/32 = 100 updates = one epoch.
- Encoder LR 5e-6; head LR 1e-4; decay .01; detection weight 1.0; clip 1.0; warmup ratio .06.
- bf16 when supported; otherwise explicitly recorded fp16. Select L4/A100 to retain parent bf16.
- Gradient checkpointing is **enabled**, preserving the recorded HPO Stage 3 lineage: `run_stage3_continuation` calls `_build_training_components`, which explicitly passes `gradient_checkpointing=True`. The resolved checkpoint config omits the field; the earlier generic-CLI false fallback assumption was corrected by tracing the actual parent path.
- Hard maximum: 100 successful optimizer calls. No HPO, vocabulary expansion, decoder/threshold change or full retraining.

CUDA and Google Colab are required for model dry-run/training. Frozen Windows paths are relocated through a separate Linux manifest overlay; finalized input bytes remain unchanged. The adapter reuses the existing isolated continuation loop and native model/data/evaluation machinery.

## Colab Cells Prepared

Notebook: `Grammar-Error-Correction/notebooks/FILLY_Duplicate_Bounded_Colab.ipynb`.

1. Environment/GPU verification and hard CUDA assertion.
2. Drive mount and experiment input directory.
3. Archive SHA verification and compatible pinned dependencies; preserve Colab CUDA PyTorch.
4. Exact source worktree/per-file verification and pinned offline tokenizer/config.
5. Parent path/hash, scratch staging and fresh durable attempt directory.
6. Critical input SHA verification table.
7. Dataset schema/count/provenance/label/tokenizer preflight.
8. Strict weight restore and zero-update GPU forward, with no optimizer/backward.
9. Parent DEV baseline metrics and loss, using the candidate's evaluation pipeline.
10. **USER RUNS THIS CELL IN GOOGLE COLAB**: explicit bounded training start.
11. Candidate hashes and durable manifests.
12. Export/results summary and post-Colab handoff.

Non-Torch versions match the existing verified GEC environment: Transformers 4.57.6, Accelerate 1.15.0, PyArrow 24.0.0, PyYAML 6.0.3, Tokenizers .22.2 and Safetensors .8.0. Colab CUDA Torch is retained, API-checked (>=2.2) and recorded with GPU/CUDA/Python versions. No blind newest-version upgrade occurs.

## Training Guard

**Local optimizer steps in this Colab-preparation phase: 0. Training has NOT started.** No notebook cell, GPU forward/backward, parent baseline inference or candidate evaluation was run locally. Tests exercise static/config/path/hash/guard/scoring logic only. The preceding data-preparation phase used tiny software-model optimizer tests before the new restriction; no FILLY model optimization occurred then. This phase did not rerun those tests.

Prepared Colab CLI shapes (not executed locally):

```bash
python -B -m filly_gec.duplicate_colab_runtime dry-run --config "$CONFIG" --run-dir "$RUN_DIR"
# USER-RUN COLAB ONLY:
python -B -m filly_gec.duplicate_colab_runtime train --config "$CONFIG" --run-dir "$RUN_DIR" --start-training
```

The cap is enforced in config, native bounded loop and adapter. Successful optimizer calls and skipped attempts are recorded durably. AMP may skip an optimizer call on gradient overflow; actual-update accounting follows [PyTorch AMP semantics](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

## Colab Runbook

1. Open the delivered notebook in Colab and select a GPU runtime.
2. Upload `filly_duplicate_colab_bundle.zip` to the experiment `inputs/` directory; cell 2 can prompt for it. The notebook pins its expected SHA.
3. Reuse the exact-hash historical Stage 3 parent or upload local `Filly/best.pt` as `inputs/parent_stage3_best.pt`. DEV is included in the small bundle.
4. Run cells 1–9. Inspect the verified hashes, `ready_for_training=true` zero-update report and saved parent DEV baseline.
5. Manually run the marked training cell. It stops by 100 updates and saves to Drive.
6. Run cells 11–12. Download the candidate, SHA report and complete attempt directory. Preserve interrupted attempts; retry from the original parent in a fresh run directory.

## Expected Outputs

```text
FILLY_GEC/experiments/duplicate-bounded-mixed-v1-seed42/
  inputs/filly_duplicate_colab_bundle.zip
  inputs/parent_stage3_best.pt          # only if historical copy unavailable
  runs/run-<UTC>-<UUID>/
    colab_config.json, environment.json, pip_freeze.txt, source_provenance.json
    input_verification.json, preflight_report.json, preparation_manifest.json
    dry_run_report.json, resolved_config.json, training_config.json
    training_manifest.json, hash_report.json, final_report.json
    metrics/parent_dev_metrics.json, candidate_dev_metrics.json, metrics.json
    logs/*.log and durable progress/event records
    checkpoints/candidate/candidate_stage3_update100.pt
    checkpoints/candidate/run_manifest.json
    candidate_hashes.json
```

The candidate is a fixed-budget final point, never production `best.pt`. Start/end UTC, hardware/environment, source commit/worktree, input/parent hashes, optimizer plan and actual count, precision/checkpointing, evaluation point and candidate SHA are retained. There is no unverified automatic resume.

## Post-Colab Evaluation

After downloading the candidate and Colab `hash_report.json`, from the workspace root (replace `YOUR_COLAB_RUN_ID` with the actual attempt ID):

```powershell
$RunId = 'YOUR_COLAB_RUN_ID'
.\Grammar-Error-Correction\.venv\Scripts\python.exe -B Filly/backend/scripts/compare_duplicate_candidate.py compare `
  --baseline-checkpoint Filly/best.pt `
  --candidate-checkpoint "Filly/backend/reports/colab_download/$RunId/candidate_stage3_update100.pt" `
  --candidate-hash-report "Filly/backend/reports/colab_download/$RunId/hash_report.json" `
  --output-dir "Filly/backend/reports/duplicate_candidate_comparison-$RunId" `
  --device auto
```

Use the GEC virtual environment: it has both PyArrow and the backend service dependencies; the backend environment alone lacks PyArrow. Imports were verified without creating a model instance. The candidate SHA is checked before loading; source/diagnostic hashes and the evaluator code fingerprint are recorded. All comparisons use newly computed baseline outputs. Metrics include duplicate P/R/F0.5/ERR/exact recovery; historical seen/unseen, function/content, initial/interior/final and subword groups; ordinary actual tags/families; identity false deletions/other false corrections; and collateral edits for passes 1?5.

The prepared comparator has not run. It recomputes both baseline and candidate with identical code on official DEV, historical duplicate DEV, frozen 100 stress cases, identity controls and ordinary categories. It reports dependent passes 1–5 with identical thresholds. Existing production defaults are untouched; historical five-pass descriptions differ from current working-tree settings, so evaluation requests five explicitly.

## Acceptance Criteria

The classifier emits exactly one of:

- `PROCEED TO CANONICAL RETRAINING`
- `RUN ANOTHER BOUNDED EXPERIMENT`
- `DO NOT PROCEED`

Require meaningful unseen/content recall above zero, material overall duplicate recall improvement, strong precision, stable overall GEC F0.5/ERR, reasonable stability across actual categories and low identity false deletion. Recompute baseline; do not hard-code 23/100 or require a predetermined 80% target. Historical seen/unseen definitions remain frozen; augmented-vocabulary coverage is secondary. Report paired counts/uncertainty. Final TEST stays sealed.

If successful, assess surgical canonical replacement using provenance: retain unaffected rows/tags, remove only identified historical duplicate-word rows, insert v2 rows under existing split assignments with collision/replay audits, then create a new versioned dataset for proper Stage 2/3 retraining. Do not regenerate unaffected tags or execute that future strategy now.

## Files Changed

- GEC `notebooks/FILLY_Duplicate_Bounded_Colab.ipynb`: unexecuted 12-stage notebook, archive SHA pinned.
- GEC `src/filly_gec/duplicate_colab_runtime.py`: Colab path overlay, guards, verification, zero-update dry run, DEV baseline/candidate scoring and durable handoff.
- GEC `src/filly_gec/experimental_stage3_continuation.py`: minimal successful-step hook and progress callback accounting; shared training algorithm retained.
- GEC `tests/test_duplicate_colab_runtime.py`: five static/config/guard tests with no optimizer calls.
- GEC `experiments/duplicate_bounded_colab_v1/prepare_bundle.py`: frozen-input/source packaging, no data generation or training.
- GEC `experiments/duplicate_bounded_colab_v1/colab_experiment_config.json`: pinned inputs and experiment settings; notebook fills fresh runtime paths automatically.
- GEC `experiments/duplicate_bounded_colab_v1/requirements-colab-bounded.txt`: verified compatible non-Torch pins.
- GEC `experiments/duplicate_bounded_colab_v1/README.md`: short manual Colab runbook and Drive layout.
- GEC `experiments/duplicate_bounded_colab_v1/test_handoff.py`: notebook/config/archive safety checks.
- GEC `experiments/duplicate_bounded_colab_v1/`: explicit input inventory, local validation configs/results, static review, package-asset check, safety audit, git evidence and per-file SHA inventory.
- GEC `outputs/duplicate_colab_handoff_v1/`: upload ZIP, expected ZIP SHA sidecar, archive member inventory/summary, ready-to-open notebook copy and runbook copy.
- FILLY `backend/scripts/compare_duplicate_candidate.py`: future paired DEV/stress/identity/category comparison, candidate hash guard and P1/P5 acceptance classifier.
- FILLY `backend/tests/test_compare_duplicate_candidate.py`: eight synthetic scoring/decision/hash tests; no inference or optimizer.
- FILLY `backend/reports/duplicate_colab_preparation.md`: this complete report.

Every evidence/output file is enumerated individually with size, purpose and SHA in `Grammar-Error-Correction/experiments/duplicate_bounded_colab_v1/files_changed_manifest.json`. Unrelated production files and frozen input artifacts are preserved.



The handoff ZIP is **10,966,566 bytes** (about 11 MB), SHA-256 `303d820d1751b7efcc214a26fd18859e4487d4125268adfb64dbcfdc6a3433aa`. It is located at `Grammar-Error-Correction/outputs/duplicate_colab_handoff_v1/filly_duplicate_colab_bundle.zip`; a notebook copy and short runbook sit alongside it. All 83 inventoried members match their SHA values, all 53 packaged Python files compile without execution, and packaged assets load offline as RobertaConfig (hidden 1024) / RobertaTokenizerFast (30,000 vocabulary) at the exact pinned revision, without constructing a model.

Local validation: **4 handoff tests + 5 adapter tests + 8 comparator tests pass**. Exact commands/results are in `validation_results.json`. A bounded independent static review found no confirmed blockers at its recorded snapshot. A subsequent source-lineage correction preserves gradient checkpointing=true; the extra pure-setting test and `gradient_checkpointing_lineage_validation.json` verify the parent HPO path without model execution. The upload bundle was refreshed and pinned after that correction. The final hash/schema/provenance check passes (3,200 TRAIN, 50,400 DEV, 1,280/1,920 mix, unique IDs, null structural fields 0). `local_safety_audit.json` confirms all frozen inputs and all 51 protected production files still match baseline hashes, no new test invokes step/backward, and local optimizer activity in this phase is 0. No candidate or Colab runtime result exists yet.

## Git State

Full baseline/final `git status --short`, `git diff --stat` and scoped whitespace checks are retained in `Grammar-Error-Correction/experiments/duplicate_bounded_colab_v1/`. Unrelated dirty work is preserved. The previously reported unrelated `test_gate4_finalization.py` import error was not modified and does not block the new scoped tests.

## Safety Confirmation

The finalized 3,200-row dataset was not regenerated. Canonical Stage 3 TRAIN was not overwritten. DEV was not used for local training. Final TEST was not accessed for candidate selection. Production best.pt/inference were not modified. No optimizer steps were executed locally during this phase. No full Stage 2/Stage 3 training started. Nothing was committed or pushed. Optimization remains a manual user action in Google Colab.
