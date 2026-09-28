# Normalizer baseline report status

`normalizer_baseline.json` is the frozen exploratory baseline captured before
normalizer implementation edits. It scored all 98 archive rows remaining after
the two training-source overlaps were excluded. Its original `split: "test"`
label refers to the archive member name; it is not the reserved final test
partition described below. Do not use its all-row metrics to tune a candidate.

New evaluator runs partition those 98 rows deterministically using seed 42 and
SHA-256 ranking. The lowest 20 ranks form the 20-row final test set; the other
78 rows form validation. Tune only against `normalizer_validation_baseline.json`
and later validation reports. Evaluate the reserved test partition after
selecting the candidate. Partition row numbers and dataset hashes are stored in
each report.

Runtime measurements always use the complete 98-row ordered input list, with
gold labels excluded from the benchmark call. Reports record the ordered
workload SHA-256 so baseline and optimized timings can be compared on identical
inputs even when accuracy evaluation uses only validation or test rows.
