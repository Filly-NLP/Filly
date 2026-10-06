# FILLY research evaluation CLI

Run the CLI from the `Filly/` repository directory. It loads the configured checkpoint and normalizer from the backend settings and runs the same GEC decoder used by the application.

```powershell
# GEC only
python filly_cli.py evaluate test.csv --normalization off

# Normalization followed by GEC
python filly_cli.py evaluate test.csv --normalization on

# Paired comparison on the same rows, with machine-readable reports
python filly_cli.py evaluate test.csv --normalization both --output-dir reports/evaluation
```

The whole CSV is checked before model loading, including when `--limit` is used. The limit evaluates the first N validated records; the report records both the CSV row count and evaluated count. No rows are balanced, sampled, or silently dropped. Type labels are used for grouping only, so results do not imply statistical significance.

The manuscript describes iterative GEC without fixing a pass count. The CLI defaults to one pass based on the bounded single-pass recommendation in `reports/GEC_FIVE_PASS_ITERATION_ANALYSIS.md` (line 35); this is an evaluation choice, not a thesis-prescribed count. Production currently uses five passes. Use `--gec-passes 5` to evaluate that setting, or pass another positive number to run that many dependent decoder passes. `--device auto|cpu|cuda` overrides the backend `DEVICE`. The CLI reads `backend/.env` when present and honors the same checkpoint and normalizer paths as the application.

The edit scorer is provided by the adjacent `Grammar-Error-Correction/src` checkout. If that source tree is unavailable, install that project in editable mode from its repository root (`python -m pip install -e .`) before evaluating. The CLI fails before loading the large checkpoint when the scorer cannot be found.

## CSV fields

The required fields are source input text and its expected final target. `gec_correction` and `unnormalized_type` are optional. The CLI accepts these case-insensitive aliases:

| Meaning | Accepted names |
| --- | --- |
| Input | `input`, `source`, `source_text`, `text`, `noisy_text`, `original` |
| Target | `target`, `target_text`, `reference`, `gold`, `expected` |
| Optional correction metadata | `gec_correction`, `correction`, `correction_tag`, `correction_tags`, `gec_tag`, `expected_gec_tag`, `generation_tag`, `error_type` |
| Optional type | `unnormalized_type`, `normalization_type`, `noise_type`, `informal_type`, `category` |

Column aliases are resolved without guessing. If more than one candidate exists, select the intended field explicitly. For example, an existing GEC export may use `source_text` and `target_text`, with `correction_tags` as optional metadata:

```powershell
python filly_cli.py evaluate gec_test.csv --normalization both `
  --input-column source_text --target-column target_text `
  --correction-column correction_tags --type-column unnormalized_type
```

In the GEC dataset, `source_text` maps to `input` and `target_text` maps to `target`. `correction_tags` can be selected as `gec_correction`; it is descriptive metadata and never replaces the expected output. `generation_tag` and `expected_gec_tag` have different meanings in the data and remain separate original fields. If both are present, choose the correction field with `--correction-column`; both values remain in each prediction's `original_columns_json`. Fields such as `error_families` and `generation_operation` are retained there too. The current GEC evaluation Parquet does not provide `unnormalized_type`; it cannot be used for type breakdowns until real dominant-category labels are supplied. The CLI never infers normalization labels from GEC correction metadata.

Format-only example (synthetic text, not research data):

```csv
input,target,gec_correction,unnormalized_type
"informal sample input","expected final text",example_tag,slang
```

The manuscript assigns each sample one dominant informal feature: `slang`, `abbreviation`, or `spelling variation`. Use one type per row. Semicolon-separated multi-type values are rejected. Hyphens, underscores, case, and plural forms are normalized for grouping; the original type string is retained in the prediction output. Other single labels are included as additional groups. A blank type is allowed and contributes to Overall only.

## Metrics and interpretation

Precision, Recall, and F0.5 use FILLY's authoritative GEC edit scorer. Overall contains every evaluated row exactly once; the exclusive dominant type selects one category group.

The manuscript's Eq. 4 defines Error Reduction Rate (ERR) as a percentage measured before GEC, comparing error before and after normalization (Chapters 1-3, manuscript PDF page 67, printed page 63). The category description says each sample is assigned its dominant informal language feature (PDF page 62, printed page 58). The evaluation CSV supplies one final corrected/normalized target rather than a separate normalization-only gold. FILLY therefore operationalizes the stage comparison as:

```text
errors_before_normalization = edits(raw input -> shared final target)
errors_after_normalization  = edits(normalized input -> shared final target)
ERR (%) = 100 * (sum(errors_before_normalization) - sum(errors_after_normalization))
          / sum(errors_before_normalization)
```

Counts are summed before computing ERR; per-row percentages are not averaged. A zero denominator is N/A, and a negative ERR means normalization increased the edits needed to reach the target. This uses the final target as a shared operational reference for the normalization stage. It is not a normalization-only accuracy score. In GEC-only mode, normalization is bypassed, so ERR is 0% when the baseline has errors and N/A when the baseline is zero.

The edit metrics aggregate TP, FP, and FN, with `Precision = TP / (TP + FP)`, `Recall = TP / (TP + FN)`, and `F0.5 = (1 + 0.5^2) * Precision * Recall / (0.5^2 * Precision + Recall)`. Zero denominators are reported safely as zero. Final-system ERR is `(sum(errors_before) - sum(final_errors_after)) / sum(errors_before)`; it is a fraction, with N/A for a zero baseline.

The report also includes `final_system_error_reduction_rate`, a separate fraction calculated from raw input to final GEC output. It answers a different question from the manuscript's normalization-stage ERR. With `--normalization both`, the comparison table reports with-minus-without deltas for both values.

## Reports

When `--output-dir` is supplied, the CLI writes:

- `evaluation_summary.json` with metrics, comparison, configuration, data/model/resource hashes, source-code hashes, timestamp, and Git status.
- `evaluation_metrics.csv` with counts and metrics for Overall and each category.
- `evaluation_predictions.csv` with paired outputs for `both` or only the relevant mode columns for a single mode. `original_columns_json` preserves every input CSV column without overwriting fields that share names with generated outputs.
- `evaluation_comparison.csv` for `both` mode.

The summary records the dataset SHA-256, checkpoint SHA-256, tokenizer assets and tokenizer source, normalizer resource hashes when normalization is used, the scorer source and hash, effective pass/device settings, FILLY and scorer repository status, and code hashes. The timestamp does not affect metric values.
