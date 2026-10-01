"""Evaluate and benchmark FILLY's standalone Filipino spelling normalizer.

This script imports only ``app.services.normalizer``. It never imports or calls
GEC code. The default held-out pairs and valid-word list are read directly from
the supplied research archive; archive test rows that reuse a training source
are reported and excluded from pair metrics.

Example (from the repository root)::

    python backend/scripts/evaluate_normalizer.py --split test \
        --output backend/reports/normalizer_test_baseline.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import statistics
import subprocess
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SOURCE_ARCHIVE = PROJECT_ROOT / "efficient-spelling-normalization-filipino-main (1).zip"
SOURCE_ARCHIVE_SHA256 = "f001e49b9e7854925068774f0c600e1a83b9d00dc895a49377e648041e494b7c"
TEST_MEMBER = "efficient-spelling-normalization-filipino-main/data/test_words.csv"
VALID_WORD_MEMBER = "efficient-spelling-normalization-filipino-main/TagalogStemmerPython/validation.txt"
TRAIN_PAIRS = BACKEND_ROOT / "normalization" / "data" / "train_pairs.csv"
DEFAULT_ARTIFACT_DIR = BACKEND_ROOT / "artifacts" / "normalizer"
DEFAULT_BENCHMARK_SIZES = (1, 100, 1_000, 10_000)
PARTITION_SEED = 42
TEST_FRACTION = 0.2
DIAGNOSTIC_PARAGRAPH = (
    "aq ay pumunta sa skul kahapon pra kunin yung mga gamit ko pero ndi ko nakita ang "
    "teacher namin. sabi ng kaibigan ko na baka umuwi na siya kc masama raw ang "
    "pakiramdam nya. kaya nag decide ako na bumalik nalang bukas at kukunin ko nalang "
    "yung mga gamit kapag nandun na siya."
)
KNOWN_DIAGNOSTICS = {
    "skul": ["school", "eskuwela"],
    "kc": ["kasi"],
    "ndi": ["hindi"],
    "nya": ["niya"],
    "nalang": ["na lang"],
    "nandun": ["nandoon"],
}


@dataclass(frozen=True)
class GoldPair:
    row_number: int
    source: str
    expected: str
    category: str | None = None


def _decode_text(raw: bytes, source: str) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return raw.decode("latin-1")
        except UnicodeDecodeError as exc:  # latin-1 accepts every byte; defensive only
            raise ValueError(f"Could not decode {source}: {exc}") from exc


def _column_index(column: int | str, header: Sequence[str] | None, role: str) -> int:
    if isinstance(column, int):
        if column < 0:
            raise ValueError(f"{role} column index must be non-negative")
        return column
    if header is None:
        raise ValueError(f"Named {role} column {column!r} requires a CSV header")
    normalized = [name.strip().lower() for name in header]
    try:
        return normalized.index(column.strip().lower())
    except ValueError as exc:
        raise ValueError(f"CSV has no {role} column named {column!r}") from exc


def _header_roles(row: Sequence[str]) -> tuple[int, int] | None:
    names = [value.strip().lower() for value in row]
    source_aliases = {"input", "source", "wrong", "misspelled", "original", "informal"}
    expected_aliases = {"expected", "target", "correct", "normalized", "gold"}
    source = next((i for i, value in enumerate(names) if value in source_aliases), None)
    expected = next((i for i, value in enumerate(names) if value in expected_aliases), None)
    if source is not None and expected is not None:
        return source, expected
    return None


def parse_gold_pairs(
    text: str,
    *,
    input_column: int | str = 0,
    expected_column: int | str = 1,
    category_column: int | str | None = None,
    source_name: str = "gold CSV",
) -> list[GoldPair]:
    """Parse headered or headerless input/expected CSV pairs."""
    try:
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except csv.Error as exc:
        raise ValueError(f"Invalid CSV in {source_name}: {exc}") from exc
    rows = [(index, row) for index, row in enumerate(rows, start=1) if row and any(x.strip() for x in row)]
    if not rows:
        raise ValueError(f"No gold rows found in {source_name}")

    header_roles = _header_roles(rows[0][1])
    header: Sequence[str] | None = None
    first_data_index = 0
    if header_roles is not None:
        header = rows[0][1]
        first_data_index = 1
        if input_column == 0:
            input_column = header_roles[0]
        if expected_column == 1:
            expected_column = header_roles[1]

    input_index = _column_index(input_column, header, "input")
    expected_index = _column_index(expected_column, header, "expected")
    if category_column is None and header is not None:
        category_column = next(
            (index for index, name in enumerate(header) if name.strip().casefold() == "category"),
            None,
        )
    category_index = (
        _column_index(category_column, header, "category")
        if category_column is not None
        else None
    )
    pairs: list[GoldPair] = []
    for row_number, row in rows[first_data_index:]:
        required_index = max(input_index, expected_index, category_index or 0)
        if required_index >= len(row):
            raise ValueError(
                f"Expected input and target columns in {source_name} at row {row_number}"
            )
        source, expected = row[input_index].strip(), row[expected_index].strip()
        if not source or not expected:
            raise ValueError(f"Empty input or expected value in {source_name} at row {row_number}")
        category = row[category_index].strip() if category_index is not None else None
        pairs.append(GoldPair(row_number, source, expected, category or None))
    if not pairs:
        raise ValueError(f"No usable gold pairs found in {source_name}")
    return pairs


def read_pairs_file(path: Path) -> list[GoldPair]:
    try:
        text = _decode_text(path.read_bytes(), str(path))
    except OSError as exc:
        raise ValueError(f"Could not read pair file {path}: {exc}") from exc
    return parse_gold_pairs(text, source_name=str(path))


def load_archive_member(archive_path: Path, member: str) -> bytes:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            return archive.read(member)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Could not read {member!r} from {archive_path}: {exc}") from exc


def load_archive_gold(archive_path: Path = SOURCE_ARCHIVE) -> list[GoldPair]:
    return parse_gold_pairs(
        _decode_text(load_archive_member(archive_path, TEST_MEMBER), TEST_MEMBER),
        source_name=f"{archive_path}::{TEST_MEMBER}",
    )


def load_training_pairs(path: Path = TRAIN_PAIRS) -> list[GoldPair]:
    return read_pairs_file(path)


def filter_training_overlaps(
    pairs: Sequence[GoldPair],
    training_pairs: Sequence[GoldPair],
    *,
    exclude_source_overlaps: bool = True,
) -> tuple[list[GoldPair], list[dict[str, object]], dict[str, int]]:
    """Optionally exclude exact/source overlaps and keep an audit list.

    A target-only overlap is not an input leak; it is counted but retained.
    Source overlaps are excluded because the model has already seen that input.
    """
    train_pair_set = {(pair.source.lower(), pair.expected.lower()) for pair in training_pairs}
    train_sources = {pair.source.lower() for pair in training_pairs}
    train_targets = {pair.expected.lower() for pair in training_pairs}
    kept: list[GoldPair] = []
    excluded: list[dict[str, object]] = []
    counts = Counter(exact_pair=0, source=0, target=0, excluded=0)
    for pair in pairs:
        source, expected = pair.source.lower(), pair.expected.lower()
        exact = (source, expected) in train_pair_set
        source_overlap = source in train_sources
        target_overlap = expected in train_targets
        counts["exact_pair"] += int(exact)
        counts["source"] += int(source_overlap)
        counts["target"] += int(target_overlap)
        reasons = []
        if exact:
            reasons.append("exact_train_pair")
        if source_overlap:
            reasons.append("train_source")
        should_exclude = exclude_source_overlaps and (exact or source_overlap)
        if should_exclude:
            counts["excluded"] += 1
            excluded.append(
                {
                    "row_number": pair.row_number,
                    "input": pair.source,
                    "expected": pair.expected,
                    "reasons": reasons,
                    "target_overlaps_train_target": target_overlap,
                }
            )
        else:
            kept.append(pair)
    return kept, excluded, dict(counts)


def _pair_rows_sha256(pairs: Sequence[GoldPair]) -> str:
    payload = [[pair.row_number, pair.source, pair.expected] for pair in pairs]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_validation_test_split(
    pairs: Sequence[GoldPair], *, seed: int = PARTITION_SEED, test_fraction: float = TEST_FRACTION
) -> tuple[list[GoldPair], list[GoldPair], dict[str, object]]:
    """Split fixed gold rows reproducibly without Python-version RNG behavior."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    if len(pairs) < 2:
        raise ValueError("At least two rows are required for a validation/test split")
    ranked = []
    for index, pair in enumerate(pairs):
        rank_input = f"{seed}\0{pair.row_number}\0{pair.source}\0{pair.expected}".encode("utf-8")
        ranked.append((hashlib.sha256(rank_input).hexdigest(), index))
    test_count = min(len(pairs) - 1, max(1, math.floor(len(pairs) * test_fraction + 0.5)))
    test_indexes = {index for _digest, index in sorted(ranked)[:test_count]}
    validation = [pair for index, pair in enumerate(pairs) if index not in test_indexes]
    test = [pair for index, pair in enumerate(pairs) if index in test_indexes]
    metadata = {
        "method": "SHA-256 rank by seed, source row number, input, and expected value; lowest ranks assigned to test",
        "seed": seed,
        "test_fraction": test_fraction,
        "source_pair_count": len(pairs),
        "validation_pair_count": len(validation),
        "test_pair_count": len(test),
        "source_rows_sha256": _pair_rows_sha256(pairs),
        "validation_rows_sha256": _pair_rows_sha256(validation),
        "test_rows_sha256": _pair_rows_sha256(test),
        "validation_source_rows": [pair.row_number for pair in validation],
        "test_source_rows": [pair.row_number for pair in test],
    }
    return validation, test, metadata


def runtime_workload_metadata(words: Sequence[str]) -> dict[str, object]:
    """Fingerprint the fixed ordered word list used for every runtime report."""
    encoded = json.dumps(list(words), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "word_count": len(words),
        "ordered_words_sha256": hashlib.sha256(encoded).hexdigest(),
        "order": "archive/input row order after exact/source training-overlap exclusion",
        "gold_labels_passed_to_benchmark": False,
    }


def load_valid_words(archive_path: Path = SOURCE_ARCHIVE) -> list[str]:
    text = _decode_text(load_archive_member(archive_path, VALID_WORD_MEMBER), VALID_WORD_MEMBER)
    return _unique_words(text.splitlines())


def _unique_words(lines: Iterable[str]) -> list[str]:
    words = [line.strip() for line in lines if line.strip()]
    # Preserve first occurrence order while treating case variants as duplicates.
    unique: dict[str, str] = {}
    for word in words:
        unique.setdefault(word.lower(), word)
    return list(unique.values())


def read_valid_words_file(path: Path) -> list[str]:
    try:
        text = _decode_text(path.read_bytes(), str(path))
    except OSError as exc:
        raise ValueError(f"Could not read valid-word file {path}: {exc}") from exc
    return _unique_words(text.splitlines())


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_pair_metrics(
    pairs: Sequence[GoldPair], predictions: Sequence[str]
) -> dict[str, object]:
    if len(pairs) != len(predictions):
        raise ValueError("Pair and prediction counts differ")
    total = len(pairs)
    changed = 0
    correct_changes = 0
    normalization_needed = 0
    unchanged_failures = 0
    wrong_replacements = 0
    exact_matches = 0
    false_positive_changes = 0
    identity_rows = 0
    for pair, prediction in zip(pairs, predictions):
        identity = pair.source == pair.expected
        changed_this = prediction != pair.source
        correct = prediction == pair.expected
        exact_matches += int(correct)
        identity_rows += int(identity)
        changed += int(changed_this)
        if identity:
            false_positive_changes += int(changed_this)
        else:
            normalization_needed += 1
            if correct and changed_this:
                correct_changes += 1
            elif not changed_this:
                unchanged_failures += 1
            else:
                wrong_replacements += 1
    changed_precision = _ratio(correct_changes, changed)
    changed_recall = _ratio(correct_changes, normalization_needed)
    # Undefined precision/recall remains null when its denominator is zero;
    # otherwise both zero yields a meaningful F1 of zero.
    f1 = (
        2 * changed_precision * changed_recall / (changed_precision + changed_recall)
        if changed_precision is not None
        and changed_recall is not None
        and changed_precision + changed_recall > 0
        else 0.0 if changed_precision is not None and changed_recall is not None else None
    )
    return {
        "pair_count": total,
        "exact_match": {"correct": exact_matches, "denominator": total, "rate": _ratio(exact_matches, total)},
        "normalization_needed": normalization_needed,
        "changed_predictions": changed,
        "correct_changes": correct_changes,
        "changed_word_precision": {"numerator": correct_changes, "denominator": changed, "rate": changed_precision},
        "changed_word_recall": {
            "numerator": correct_changes,
            "denominator": normalization_needed,
            "rate": changed_recall,
        },
        "changed_decision_f1": f1,
        "identity_rows": identity_rows,
        "false_positive_changes": false_positive_changes,
        "false_positive_rate_on_identity_pairs": {
            "numerator": false_positive_changes,
            "denominator": identity_rows,
            "rate": _ratio(false_positive_changes, identity_rows),
        },
        "unchanged_failure_rate": {
            "numerator": unchanged_failures,
            "denominator": normalization_needed,
            "rate": _ratio(unchanged_failures, normalization_needed),
        },
        "wrong_replacement_rate": {
            "numerator": wrong_replacements,
            "denominator": changed,
            "rate": _ratio(wrong_replacements, changed),
            "of_normalization_needed": _ratio(wrong_replacements, normalization_needed),
        },
    }


def evaluate_valid_words(
    words: Sequence[str],
    predictions: Sequence[str],
    *,
    training_sources: set[str],
) -> dict[str, object]:
    if len(words) != len(predictions):
        raise ValueError("Valid-word and prediction counts differ")
    excluded_source_overlap = 0
    evaluated = 0
    changed = 0
    excluded_examples: list[str] = []
    for word, prediction in zip(words, predictions):
        if word.lower() in training_sources:
            excluded_source_overlap += 1
            if len(excluded_examples) < 20:
                excluded_examples.append(word)
            continue
        evaluated += 1
        changed += int(prediction != word)
    return {
        "source": "archive TagalogStemmerPython/validation.txt; distinct case-insensitive words",
        "unique_word_count": len(words),
        "train_source_overlaps_excluded": excluded_source_overlap,
        "evaluated_word_count": evaluated,
        "changed_valid_words": changed,
        "false_positive_rate": {"numerator": changed, "denominator": evaluated, "rate": _ratio(changed, evaluated)},
        "excluded_examples": excluded_examples,
    }


def _candidate_diagnostics(normalizer: object, source: str, expected: str, prediction: str) -> dict[str, object]:
    lower = source.lower()
    in_vocabulary = lower in normalizer.vocabulary
    generated: dict[str, float] = {}
    if not in_vocabulary and 1 < len(lower) <= 128:
        generated = normalizer._collect_candidates(
            lower, normalizer.rules, normalizer.max_ngram, normalizer.candidate_cutoff
        )
    survivors = [
        candidate
        for candidate in generated
        if candidate.strip()
        and all(part.lower() in normalizer.vocabulary for part in candidate.strip().split())
    ]
    expected_key = expected.lower()
    expected_generated = expected_key in {candidate.lower() for candidate in generated}
    expected_survives = expected_key in {candidate.lower() for candidate in survivors}
    ranked = sorted(
        (
            (normalizer_distance(lower, candidate), index, candidate)
            for index, candidate in enumerate(survivors)
        ),
        key=lambda row: (row[0], row[1]),
    )
    top_candidates = [
        {"candidate": candidate, "damerau_levenshtein": distance}
        for distance, _original_index, candidate in ranked[:10]
    ]
    expected_distance = normalizer_distance(lower, expected_key) if expected_survives else None
    max_edit_distance = normalizer.max_edit_distance
    if expected_key == lower and in_vocabulary:
        expected_status = "input_in_vocabulary"
    elif not expected_generated:
        expected_status = "absent_from_generated_candidates"
    elif not expected_survives:
        expected_status = "filtered_by_vocabulary"
    elif max_edit_distance is not None and expected_distance > max_edit_distance:
        expected_status = "distance_gated"
    else:
        expected_status = "available"

    if source.casefold() == expected.casefold() and prediction != source:
        failure_type = "false_positive"
    elif prediction == source:
        if in_vocabulary:
            failure_type = "input_already_in_vocabulary"
        elif expected_status == "distance_gated":
            failure_type = "expected_candidate_distance_gated"
        elif " " in expected and not expected_generated:
            failure_type = "multi_token_unsupported"
        elif expected_generated and not expected_survives:
            failure_type = "expected_candidate_filtered_out"
        elif not expected_generated:
            failure_type = "expected_candidate_not_generated"
        elif not survivors:
            failure_type = "no_candidate"
        else:
            failure_type = "normalizer_made_no_change"
    elif expected_generated and not expected_survives:
        failure_type = "expected_candidate_filtered_out"
    elif expected_status == "distance_gated":
        failure_type = "expected_candidate_distance_gated"
    elif " " in expected and not expected_generated:
        failure_type = "multi_token_unsupported"
    elif not expected_generated:
        failure_type = "expected_candidate_not_generated"
    elif not survivors:
        failure_type = "no_candidate"
    else:
        failure_type = "wrong_candidate_ranking"

    diagnostics = {
        "input_in_vocabulary": in_vocabulary,
        "generated_candidate_count": len(generated),
        "vocabulary_survivor_count": len(survivors),
        "expected_candidate_generated": expected_generated,
        "expected_candidate_survives_vocabulary_filter": expected_survives,
        "expected_candidate_status": expected_status,
        "expected_candidate_damerau_levenshtein": expected_distance,
        "maximum_applied_damerau_levenshtein": max_edit_distance,
        "failure_type": failure_type,
        "top_candidates_by_dld": top_candidates,
    }
    explain = getattr(normalizer, "explain_word", None)
    if callable(explain):
        decision = explain(source)
        diagnostics.update(
            {
                "strategy": decision["strategy"],
                "source_id": decision["source_id"],
                "curated_rule_status": decision["curated_rule_status"],
            }
        )
    else:
        diagnostics.update(
            {"strategy": "ngram_dld", "source_id": None, "curated_rule_status": "unavailable"}
        )
    return diagnostics


def normalizer_distance(left: str, right: str) -> int:
    # Import lazily alongside the production normalizer, keeping the module's
    # utility functions testable without loading application services.
    return _damerau_levenshtein_distance(left, right)


_damerau_levenshtein_distance = None


def _load_normalizer(
    artifact_dir: Path,
    *,
    max_edit_distance: int | None = 2,
    use_curated_mappings: bool = True,
):
    backend_text = str(BACKEND_ROOT)
    if backend_text not in sys.path:
        sys.path.insert(0, backend_text)
    # This is intentionally the only application import in the evaluator.
    from app.services.normalizer import FilipinoNormalizer, damerau_levenshtein_distance

    global _damerau_levenshtein_distance
    _damerau_levenshtein_distance = damerau_levenshtein_distance
    return FilipinoNormalizer(
        artifact_dir=artifact_dir,
        max_edit_distance=max_edit_distance,
        use_curated_mappings=use_curated_mappings,
    )


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _timing_summary(values: Sequence[float]) -> dict[str, object]:
    return {
        "samples": len(values),
        "runs_ms": [round(value, 6) for value in values],
        "median_ms": round(statistics.median(values), 6) if values else None,
        "p95_ms": round(_percentile(values, 0.95), 6) if values else None,
    }


def _cyclic(items: Sequence[str], count: int) -> list[str]:
    if not items:
        raise ValueError("Cannot benchmark an empty input set")
    return [items[index % len(items)] for index in range(count)]


def benchmark_runtime(
    words: Sequence[str],
    *,
    artifact_dir: Path,
    max_edit_distance: int | None = 2,
    use_curated_mappings: bool = True,
    sizes: Sequence[int] = DEFAULT_BENCHMARK_SIZES,
    repeats: int = 5,
) -> dict[str, object]:
    if repeats < 1:
        raise ValueError("Benchmark repeats must be positive")
    if any(size < 1 for size in sizes):
        raise ValueError("Benchmark sizes must be positive")
    unique_warmup_words = list(dict.fromkeys(words))[:100]
    output: dict[str, object] = {"repeats": repeats, "sizes_words": list(sizes), "word": {}, "sentence": {}}

    for mode in ("word", "sentence"):
        mode_results: dict[str, object] = {}
        for size in sizes:
            inputs = _cyclic(words, size)
            normalizer = _load_normalizer(
                artifact_dir,
                max_edit_distance=max_edit_distance,
                use_curated_mappings=use_curated_mappings,
            )
            warmup_start = time.perf_counter()
            if mode == "word":
                for word in unique_warmup_words:
                    normalizer.normalize_word(word)
            else:
                normalizer.normalize_text(" ".join(unique_warmup_words))
            warmup_ms = (time.perf_counter() - warmup_start) * 1_000

            batch_times: list[float] = []
            if mode == "word":
                for _ in range(repeats):
                    start = time.perf_counter()
                    for word in inputs:
                        normalizer.normalize_word(word)
                    batch_times.append((time.perf_counter() - start) * 1_000)
                latency_words = _cyclic(words, min(max(100, size), 500))
                individual_latencies: list[float] = []
                for word in latency_words:
                    start = time.perf_counter()
                    normalizer.normalize_word(word)
                    individual_latencies.append((time.perf_counter() - start) * 1_000)
                request_latency = _timing_summary(individual_latencies)
            else:
                sentence = " ".join(inputs)
                for _ in range(repeats):
                    start = time.perf_counter()
                    normalizer.normalize_text(sentence)
                    batch_times.append((time.perf_counter() - start) * 1_000)
                request_latency = _timing_summary(batch_times)

            median_total = statistics.median(batch_times)
            mode_results[str(size)] = {
                "words_per_batch": size,
                "repeats": repeats,
                "warmup_unique_words": len(unique_warmup_words),
                "warmup_elapsed_ms": round(warmup_ms, 6),
                "total_elapsed_ms_median": round(median_total, 6),
                "total_elapsed_ms_p95": round(_percentile(batch_times, 0.95), 6),
                "batch_elapsed_ms": [round(value, 6) for value in batch_times],
                "words_per_second_median": round(size / (median_total / 1_000), 6) if median_total else None,
                "mean_elapsed_ms_per_word": round(median_total / size, 6),
                "request_latency_ms": request_latency,
            }
        output[mode] = mode_results
    return output


def _candidate_count_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    counts = [int(row["generated_candidate_count"]) for row in rows]
    survivors = [int(row["vocabulary_survivor_count"]) for row in rows]
    return {
        "inputs_measured": len(counts),
        "generated": {
            "mean": statistics.mean(counts) if counts else None,
            "median": statistics.median(counts) if counts else None,
            "p95": _percentile(counts, 0.95),
            "max": max(counts, default=None),
        },
        "vocabulary_survivors": {
            "mean": statistics.mean(survivors) if survivors else None,
            "median": statistics.median(survivors) if survivors else None,
            "p95": _percentile(survivors, 0.95),
            "max": max(survivors, default=None),
        },
    }


def _git_metadata() -> dict[str, str | None]:
    def git_value(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None

    return {
        "branch": git_value("branch", "--show-current"),
        "head": git_value("rev-parse", "HEAD"),
    }


def evaluate(
    *,
    split: str = "validation",
    gold_path: Path | None = None,
    archive_path: Path = SOURCE_ARCHIVE,
    training_path: Path = TRAIN_PAIRS,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    max_edit_distance: int | None = 2,
    use_curated_mappings: bool = True,
    valid_words_path: Path | None = None,
    exclude_train_overlaps: bool = True,
    benchmark_sizes: Sequence[int] = DEFAULT_BENCHMARK_SIZES,
    benchmark_repeats: int = 5,
    include_runtime: bool = True,
) -> dict[str, object]:
    if gold_path is None and split == "custom":
        raise ValueError("The archive dataset may be evaluated only through its validation or reserved test split")
    all_pairs = read_pairs_file(gold_path) if gold_path else load_archive_gold(archive_path)
    return _evaluate_loaded(
        all_pairs,
        split=split,
        training_path=training_path,
        artifact_dir=artifact_dir,
        max_edit_distance=max_edit_distance,
        use_curated_mappings=use_curated_mappings,
        archive_path=archive_path,
        valid_words_path=valid_words_path,
        exclude_train_overlaps=exclude_train_overlaps,
        benchmark_sizes=benchmark_sizes,
        benchmark_repeats=benchmark_repeats,
        include_runtime=include_runtime,
        dataset_source=str(gold_path) if gold_path else f"{archive_path.name}::{TEST_MEMBER}",
    )


def _parse_column(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _parse_max_edit_distance(value: str) -> int | None:
    if value.strip().casefold() in {"none", "uncapped", "unlimited"}:
        return None
    try:
        distance = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a non-negative integer or 'none' for uncapped baseline replay"
        ) from exc
    if distance < 0:
        raise argparse.ArgumentTypeError(
            "must be a non-negative integer or 'none' for uncapped baseline replay"
        )
    return distance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test", "custom"))
    parser.add_argument("--gold", type=Path, help="CSV containing input and expected columns; defaults to archive test_words.csv")
    parser.add_argument("--input-column", type=_parse_column, default=0)
    parser.add_argument("--expected-column", type=_parse_column, default=1)
    parser.add_argument("--category-column", type=_parse_column, help="Optional category column; detected by the header when present")
    parser.add_argument(
        "--no-curated-mappings",
        "--ngram-only",
        dest="use_curated_mappings",
        action="store_false",
        default=True,
        help="Run the automatic N-Gram + DLD baseline for a research ablation",
    )
    parser.add_argument("--training-pairs", type=Path, default=TRAIN_PAIRS)
    parser.add_argument("--archive", type=Path, default=SOURCE_ARCHIVE)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--max-edit-distance",
        type=_parse_max_edit_distance,
        default=2,
        help="Maximum Damerau-Levenshtein distance for an applied correction (default: 2; use 'none' for uncapped baseline replay)",
    )
    parser.add_argument("--valid-words", type=Path, help="Optional one-word-per-row validation set; defaults to archive validation.txt")
    parser.add_argument("--include-train-overlaps", action="store_true", help="Report rather than exclude exact/source overlaps (for separately labeled train-set diagnostics)")
    parser.add_argument("--output", type=Path, help="New report path; defaults to reports/normalizer_<split>_baseline.json and refuses overwrite")
    parser.add_argument("--benchmark-sizes", type=int, nargs="+", default=list(DEFAULT_BENCHMARK_SIZES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-runtime", action="store_true", help="Skip the repeated 1/100/1000/10000-word timing matrix")
    args = parser.parse_args(argv)

    split = args.split or ("custom" if args.gold is not None else "validation")
    if args.gold is None and split == "custom":
        parser.error("--split custom requires --gold; archive rows must use validation or the reserved test split")
    output_path = args.output or (BACKEND_ROOT / "reports" / f"normalizer_{split}_baseline.json")
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite existing evaluation report: {output_path}")
    if args.gold is not None:
        # Parse with caller-selected columns before evaluate() reads the pairs.
        custom_pairs = parse_gold_pairs(
            _decode_text(args.gold.read_bytes(), str(args.gold)),
            input_column=args.input_column,
            expected_column=args.expected_column,
            category_column=args.category_column,
            source_name=str(args.gold),
        )
        # Save no temporary data: pass the parsed rows through the normal
        # evaluation path using the same loader contract.
        report = evaluate_from_pairs(
            custom_pairs,
            split=split,
            training_path=args.training_pairs,
            artifact_dir=args.artifact_dir,
            max_edit_distance=args.max_edit_distance,
            use_curated_mappings=args.use_curated_mappings,
            archive_path=args.archive,
            valid_words_path=args.valid_words,
            exclude_train_overlaps=not args.include_train_overlaps,
            benchmark_sizes=args.benchmark_sizes,
            benchmark_repeats=args.repeats,
            include_runtime=not args.skip_runtime,
            dataset_source=str(args.gold),
        )
    else:
        report = evaluate(
            split=split,
            archive_path=args.archive,
            training_path=args.training_pairs,
            artifact_dir=args.artifact_dir,
            max_edit_distance=args.max_edit_distance,
            use_curated_mappings=args.use_curated_mappings,
            valid_words_path=args.valid_words,
            exclude_train_overlaps=not args.include_train_overlaps,
            benchmark_sizes=args.benchmark_sizes,
            benchmark_repeats=args.repeats,
            include_runtime=not args.skip_runtime,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Check again immediately before creation to preserve immutable reports.
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite existing evaluation report: {output_path}")
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Evaluated {report['evaluation']['evaluated_pair_count']} held-out pairs; "
        f"exact match={report['evaluation']['metrics']['exact_match']['rate']}; "
        f"report={output_path}"
    )
    return 0


def evaluate_from_pairs(
    pairs: Sequence[GoldPair],
    *,
    split: str,
    training_path: Path = TRAIN_PAIRS,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    max_edit_distance: int | None = 2,
    use_curated_mappings: bool = True,
    archive_path: Path = SOURCE_ARCHIVE,
    valid_words_path: Path | None = None,
    exclude_train_overlaps: bool = True,
    benchmark_sizes: Sequence[int] = DEFAULT_BENCHMARK_SIZES,
    benchmark_repeats: int = 5,
    include_runtime: bool = True,
    dataset_source: str = "custom CSV",
) -> dict[str, object]:
    """Evaluation core accepting already-parsed custom rows for the CLI."""
    # The built-in evaluator shares the same report-producing path by using an
    # internal parser shim rather than writing a temporary dataset.
    return _evaluate_loaded(
        pairs,
        split=split,
        training_path=training_path,
        artifact_dir=artifact_dir,
        max_edit_distance=max_edit_distance,
        use_curated_mappings=use_curated_mappings,
        archive_path=archive_path,
        valid_words_path=valid_words_path,
        exclude_train_overlaps=exclude_train_overlaps,
        benchmark_sizes=benchmark_sizes,
        benchmark_repeats=benchmark_repeats,
        include_runtime=include_runtime,
        dataset_source=dataset_source,
    )


def _evaluate_loaded(
    all_pairs: Sequence[GoldPair],
    *,
    split: str,
    training_path: Path,
    artifact_dir: Path,
    max_edit_distance: int | None,
    use_curated_mappings: bool,
    archive_path: Path,
    valid_words_path: Path | None,
    exclude_train_overlaps: bool,
    benchmark_sizes: Sequence[int],
    benchmark_repeats: int,
    include_runtime: bool,
    dataset_source: str,
) -> dict[str, object]:
    """Shared evaluator implementation for archive and caller-supplied data."""
    train_pairs = load_training_pairs(training_path)
    pairs, excluded, overlap_counts = filter_training_overlaps(
        all_pairs, train_pairs, exclude_source_overlaps=exclude_train_overlaps
    )
    if not pairs:
        raise ValueError("No held-out gold pairs remain after excluding training overlaps")
    nonoverlap_pairs = pairs
    if split in {"validation", "test"}:
        validation_pairs, test_pairs, partition_metadata = deterministic_validation_test_split(nonoverlap_pairs)
        pairs = validation_pairs if split == "validation" else test_pairs
    elif split == "custom":
        partition_metadata = None
    else:
        raise ValueError(f"Unsupported evaluation split: {split}")
    runtime_words = [pair.source for pair in nonoverlap_pairs]
    runtime_workload = runtime_workload_metadata(runtime_words)
    source_archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if source_archive_hash != SOURCE_ARCHIVE_SHA256:
        raise ValueError(f"Source archive SHA-256 mismatch: expected {SOURCE_ARCHIVE_SHA256}, got {source_archive_hash}")
    valid_words = read_valid_words_file(valid_words_path) if valid_words_path else load_valid_words(archive_path)
    training_sources = {pair.source.lower() for pair in train_pairs}
    load_started = time.perf_counter()
    normalizer = _load_normalizer(
        artifact_dir,
        max_edit_distance=max_edit_distance,
        use_curated_mappings=use_curated_mappings,
    )
    artifact_load_ms = (time.perf_counter() - load_started) * 1_000
    first_word = runtime_words[0]
    cold_word_started = time.perf_counter()
    cold_word_prediction = normalizer.normalize_word(first_word)
    cold_word_ms = (time.perf_counter() - cold_word_started) * 1_000
    cold_sentence_normalizer = _load_normalizer(
        artifact_dir,
        max_edit_distance=max_edit_distance,
        use_curated_mappings=use_curated_mappings,
    )
    cold_sentence_started = time.perf_counter()
    cold_sentence_output, _ = cold_sentence_normalizer.normalize_text(first_word + " po.")
    cold_sentence_ms = (time.perf_counter() - cold_sentence_started) * 1_000

    predictions: list[str] = []
    traces: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for pair in pairs:
        prediction = normalizer.normalize_word(pair.source)
        predictions.append(prediction)
        candidate = _candidate_diagnostics(normalizer, pair.source, pair.expected, prediction)
        candidate.update(
            {
                "row_number": pair.row_number,
                "input": pair.source,
                "expected": pair.expected,
                "prediction": prediction,
                "category": pair.category,
            }
        )
        candidate_rows.append(candidate)
        if prediction != pair.expected:
            traces.append(candidate)
    metrics = compute_pair_metrics(pairs, predictions)
    category_predictions: dict[str, list[tuple[GoldPair, str]]] = {}
    for pair, prediction in zip(pairs, predictions):
        if pair.category:
            category_predictions.setdefault(pair.category.strip().casefold(), []).append((pair, prediction))
    category_metrics = {
        category: {
            "row_count": len(rows),
            "metrics": compute_pair_metrics(
                [pair for pair, _prediction in rows],
                [prediction for _pair, prediction in rows],
            ),
        }
        for category, rows in sorted(category_predictions.items())
    }
    category_counts = Counter(str(row["failure_type"]) for row in traces)
    valid_predictions = [normalizer.normalize_word(word) for word in valid_words]
    valid_metrics = evaluate_valid_words(valid_words, valid_predictions, training_sources=training_sources)

    known_diagnostics = []
    for source, expected_options in KNOWN_DIAGNOSTICS.items():
        prediction = normalizer.normalize_word(source)
        candidate = _candidate_diagnostics(normalizer, source, expected_options[0], prediction)
        known_diagnostics.append({
            "input": source,
            "expected_options_for_investigation_only": expected_options,
            "prediction": prediction,
            "matches_any_expected_option": prediction.casefold() in {x.casefold() for x in expected_options},
            **candidate,
        })
    diagnostic_sentence_output, diagnostic_sentence_changes = normalizer.normalize_text(DIAGNOSTIC_PARAGRAPH)
    runtime = benchmark_runtime(
        runtime_words,
        artifact_dir=artifact_dir,
        max_edit_distance=max_edit_distance,
        use_curated_mappings=use_curated_mappings,
        sizes=benchmark_sizes,
        repeats=benchmark_repeats,
    ) if include_runtime else None

    artifact_hashes = {
        filename: hashlib.sha256((artifact_dir / filename).read_bytes()).hexdigest()
        for filename in ("rules.json", "vocabulary.txt", "curated_mappings.json", "metadata.json")
    }
    training_sources = {pair.source.lower() for pair in train_pairs}
    training_targets = {pair.expected.lower() for pair in train_pairs}
    valid_metrics["training_overlap_counts"] = {
        "train_source": sum(word.lower() in training_sources for word in valid_words),
        "train_target": sum(word.lower() in training_targets for word in valid_words),
    }
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation": {
            "split": split,
            "dataset_source": dataset_source,
            "split_strategy": partition_metadata["method"] if partition_metadata else "all non-overlap rows; exploratory/custom only",
            "seed": PARTITION_SEED,
            "final_test_reserved_for_final_evaluation": split == "validation",
            "is_final_test_evaluation": split == "test",
            "train_overlap_policy": "exclude exact pair and any training-source overlap" if exclude_train_overlaps else "report only",
            "training_pair_count": len(train_pairs),
            "normalizer_configuration": {
                "candidate_ranking": "unrestricted Damerau-Levenshtein distance",
                "maximum_applied_correction_distance": max_edit_distance,
                "use_curated_mappings": use_curated_mappings,
                "curated_mapping_count": len(normalizer.curated_mappings),
            },
            "raw_gold_pair_count": len(all_pairs),
            "nonoverlap_gold_pair_count": len(nonoverlap_pairs),
            "evaluated_pair_count": len(pairs),
            "partition": partition_metadata,
            "train_overlap_counts": overlap_counts,
            "excluded_train_overlap_examples": excluded,
            "target_only_overlaps_are_retained": True,
            "metrics": metrics,
            "category_metrics": category_metrics,
            "category_metrics_interpretation": (
                "Source-set category metrics are descriptive coverage only when the evaluated CSV supplied the curated rules; "
                "they are not an independent generalization estimate."
            ),
            "valid_word_set": valid_metrics,
            "error_category_counts": dict(sorted(category_counts.items())),
            "incorrect_prediction_traces": traces,
            "candidate_count_summary": _candidate_count_summary(candidate_rows),
            "known_diagnostic_inputs_not_used_for_tuning": known_diagnostics,
            "sentence_diagnostic": {
                "input": DIAGNOSTIC_PARAGRAPH,
                "normalized_output": diagnostic_sentence_output,
                "changes": [
                    {
                        "word": item.word,
                        "suggestion": item.suggestion,
                        "start": item.start,
                        "end": item.end,
                        "strategy": item.strategy,
                        "source_id": item.source_id,
                    }
                    for item in diagnostic_sentence_changes
                ],
            },
        },
        "runtime": {
            "scope": "direct normalizer calls only; no HTTP or GEC",
            "artifact_loading_ms": round(artifact_load_ms, 6),
            "cold_first_word_request_ms": round(cold_word_ms, 6),
            "cold_first_word_input": first_word,
            "cold_first_word_output": cold_word_prediction,
            "cold_first_sentence_request_ms": round(cold_sentence_ms, 6),
            "cold_first_sentence_input": first_word + " po.",
            "cold_first_sentence_output": cold_sentence_output,
            "fixed_workload": runtime_workload,
            "benchmark": runtime,
        },
        "provenance": {
            "git": _git_metadata(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "source_archive": archive_path.name,
            "source_archive_sha256": source_archive_hash,
            "source_members": {"gold_pairs": TEST_MEMBER, "valid_words": VALID_WORD_MEMBER},
            "training_pairs_path": str(training_path),
            "training_pairs_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
            "normalizer_artifacts": artifact_hashes,
            "random_seed": 42,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
