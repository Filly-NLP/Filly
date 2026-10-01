# Normalization command-line tools

This directory preserves the junior branch's command-line entry points while routing normalization, artifact generation, and evaluation through the current FILLY backend. The inference command loads the checksummed artifacts in `backend/artifacts/normalizer` and calls the single `app.services.normalizer.FilipinoNormalizer`. It does not load a GEC model or checkpoint. That component applies exact curated whole-form normalization rules first, then uses automatic N-Gram + DLD normalization for unmatched forms.

Run commands from the repository root with the backend environment installed from `backend/requirements.txt`.

## Normalize text

```powershell
python .\backend\n-gram_normalization\normalize.py "aq"
python .\backend\n-gram_normalization\normalize.py "aq prang" --max-edit-distance 2
```

The positional text and `--max-edit-distance` use the current normalizer. `--artifact-dir` selects a different current artifact directory. The old `--model` spelling remains as an alias for `--artifact-dir`, which must name a directory containing `rules.json`, `vocabulary.txt`, `curated_mappings.json`, and `metadata.json`. The old `--cutoff` option is accepted only when it matches the cutoff recorded in those artifacts; change it by rebuilding artifacts, not at inference time.

## Rebuild artifacts

```powershell
python .\backend\n-gram_normalization\train_normalizer.py --output "$env:TEMP\filly-normalizer-trial"
```

This command forwards to `backend/scripts/train_normalizer.py`. The default build reads the project-owned extraction at `backend/normalization/data/curated_colloquial_source.csv`; it does not need the Excel workbook at runtime. Conflicting forms are reported and excluded from direct lookup. The legacy options map as follows:

| Junior option | Current backend option |
| --- | --- |
| `--data` | `--pairs` |
| `--max-sub` | `--max-ngram` |
| `--vocab` | `--base-vocabulary` |
| `--output` | `--output` artifact directory |
| `--candidate-cutoff` | `--candidate-cutoff` |
| `--curated-source` | `--curated-source` |

The current trainer's defaults apply when an option is omitted. Use a separate `--output` directory when checking a build; the backend trainer's default output is the live artifact directory.

## Rule sources and ablation

FILLY's N-Gram Rule-Based Text Normalization combines curated whole-form rules from rows categorized exactly as `slang` with automatic N-Gram rule induction and DLD candidate ranking. Curated rules use deterministic, case-aware, boundary-safe whole-form matches; the automatic path handles unmatched abbreviations and spelling variations. The supplied `gora` rows conflict and are retained in artifact diagnostics, but excluded from direct mappings. Changes retain their original spans and carry `strategy` (`curated_rule` or `ngram_dld`) plus a curated `source_id` where applicable.

The production default uses both paths. For evaluator ablation, pass `--no-curated-mappings` (alias `--ngram-only`) to run automatic N-Gram + DLD only. CSV input with a `Category` header receives category-level metrics. Results measured on the source CSV used to build curated rules describe source-set coverage only, not independent generalization.

## Evaluate

```powershell
python .\backend\n-gram_normalization\main_algo.py --mode test --skip-runtime --output "$env:TEMP\filly-normalizer-test.json"
python .\backend\n-gram_normalization\main_algo.py --split validation --skip-runtime --output "$env:TEMP\filly-normalizer-validation.json"
```

The compatibility command calls `backend/scripts/evaluate_normalizer.py`. With no `--split`, the legacy `--mode test` selects the current reserved test split. Current evaluator options such as `--gold`, `--artifact-dir`, `--benchmark-sizes`, and `--skip-runtime` pass through. `--use_dld False` is rejected because the current engine always ranks candidates using Damerau-Levenshtein distance. Legacy `--max_sub` and `--cutoff` values are accepted as checks against the current artifact metadata. The old random `--mode eval_cv` is reported as unsupported; use the evaluator's deterministic validation or reserved test split.

## Integration scope and attribution

The optimized FILLY implementation, generated artifacts, training inputs, evaluator, and benchmark tools remain under their current `backend/app`, `backend/scripts`, and `backend/artifacts` paths. This directory contains thin command adapters, this usage guide, and an integration test. The branch's virtual environment, experiment logs/results, locally supplied datasets, generated legacy model, bundled stemmer files, GPU experiments, and duplicate legacy normalization implementation are excluded from the merge result. The original junior commit remains in Git ancestry.

The junior branch documented its research basis as Lorenzo Jaime Flores and Dragomir Radev, *Look Ma, Only 400 Samples! Revisiting the Effectiveness of Automatic N-Gram Rule Generation for Spelling Normalization in Filipino* (SustaiNLP at EMNLP 2022): <https://aclanthology.org/2022.sustainlp-1.5>. Its source repository is <https://github.com/ljyflores/efficient-spelling-normalization-filipino>.

The junior README did not identify a license for the upstream normalizer source or its datasets. This integration does not redistribute those files or add a license; confirm reuse terms with the upstream author before publishing derived source beyond the retained command adapters.
