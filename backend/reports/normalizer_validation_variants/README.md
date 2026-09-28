# Validation-only normalizer variants

All variants use the same 78-row validation split, unchanged N-gram rules and
candidate cutoff, and the same 6,074 eligible valid-word checks. The reserved
20-row test split remains untouched. The frozen uncapped baseline is
`../normalizer_validation_baseline.json`.

| Maximum DLD | Exact match | Changed precision | Changed recall | Decision F1 | Wrong replacements | Valid-word FPR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline, uncapped | 37/78 (47.44%) | 37/63 (58.73%) | 37/77 (48.05%) | 0.5286 | 25/63 (39.68%) | 1263/6074 (20.79%) |
| 1 | 28/78 (35.90%) | 28/30 (93.33%) | 28/77 (36.36%) | 0.5234 | 1/30 (3.33%) | 239/6074 (3.93%) |
| **2 (selected)** | **36/78 (46.15%)** | **36/44 (81.82%)** | **36/77 (46.75%)** | **0.5950** | **7/44 (15.91%)** | **438/6074 (7.21%)** |
| 3 | 37/78 (47.44%) | 37/50 (74.00%) | 37/77 (48.05%) | 0.5827 | 12/50 (24.00%) | 675/6074 (11.11%) |
| 4 | 37/78 (47.44%) | 37/53 (69.81%) | 37/77 (48.05%) | 0.5692 | 15/53 (28.30%) | 898/6074 (14.78%) |

The selected DLD limit of 2 follows the false-positive priority: versus the
baseline it reduces wrong replacements from 25 to 7 and valid-word changes
from 1,263 to 438, while raising decision F1 from 0.5286 to 0.5950. It gives
up one exact validation match (36 instead of 37). Against DLD 3, it gives up
one correct change and avoids five wrong replacements; recall differs by one
row (36/77 vs. 37/77), while precision, F1, and valid-word FPR improve.

## Reproduction

Run from the repository root. Each command writes a new report and refuses to
overwrite an existing report.

```powershell
backend/.venv/Scripts/python.exe backend/scripts/evaluate_normalizer.py --split validation --max-edit-distance 1 --output backend/reports/normalizer_validation_variants/normalizer_validation_dld_1.json --skip-runtime
backend/.venv/Scripts/python.exe backend/scripts/evaluate_normalizer.py --split validation --max-edit-distance 2 --output backend/reports/normalizer_validation_variants/normalizer_validation_dld_2.json --skip-runtime
backend/.venv/Scripts/python.exe backend/scripts/evaluate_normalizer.py --split validation --max-edit-distance 3 --output backend/reports/normalizer_validation_variants/normalizer_validation_dld_3.json --skip-runtime
backend/.venv/Scripts/python.exe backend/scripts/evaluate_normalizer.py --split validation --max-edit-distance 4 --output backend/reports/normalizer_validation_variants/normalizer_validation_dld_4.json --skip-runtime
```

The JSON files retain row-level traces, split hashes, valid-word counts, and
artifact hashes. The active normalizer default is DLD 2; the CLI option allows
the other variants to be reproduced without changing learned artifacts.
