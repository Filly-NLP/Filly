"""Profile normalizer runtime without importing GEC or reading held-out data.

Run from the repository root, for example::

    python backend/scripts/profile_normalizer_runtime.py --phase baseline --repeats 5 \
        --output backend/reports/normalizer_runtime_baseline_v2.json
    python backend/scripts/profile_normalizer_runtime.py --phase optimized --repeats 5 \
        --compare backend/reports/normalizer_runtime_baseline_v2.json \
        --output backend/reports/normalizer_runtime_optimized_v2.json

Comparison rejects reports with different repeats, configuration, inputs,
artifacts, Python/platform provenance, or timing definitions. Source hashes may
differ because comparison is intended to measure a code change.

The workload is built only from ``backend/normalization/data/train_pairs.csv``.
Import and artifact loading are recorded separately and excluded from call
timings. The fixed workload sizes are 1, 100, 1,000, and 10,000 tokens.
The first call for each workload is timed before that workload's warmup; it is
not a process-cold measurement because imports, artifacts, and earlier matrix
entries may already have warmed the process.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import hashlib
import inspect
import json
import math
import platform
import pstats
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
TRAIN_PAIRS = BACKEND_ROOT / "normalization" / "data" / "train_pairs.csv"
DEFAULT_SIZES = (1, 100, 1_000, 10_000)
WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(encoded.encode("utf-8"))


def _workload_words() -> list[str]:
    words: list[str] = []
    with TRAIN_PAIRS.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            words.extend(WORD_PATTERN.findall(row[0]))
    if not words:
        raise ValueError(f"No source words found in training pairs: {TRAIN_PAIRS}")
    return words


def _sentence_payload(normalized: str, changes: list[Any]) -> dict[str, Any]:
    return {
        "text": normalized,
        "changes": [
            {
                "word": item.word,
                "suggestion": item.suggestion,
                "start": item.start,
                "end": item.end,
                "type": item.type,
                "confidence": item.confidence,
                "category": item.category,
            }
            for item in changes
        ],
    }


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _measure(call: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = call()
    return result, (time.perf_counter() - started) * 1_000


def _profile_stages(module: Any, normalizer: Any, kind: str, words: list[str]) -> dict[str, Any]:
    if kind == "word":
        call = lambda: [normalizer.normalize_word(word) for word in words]
    else:
        text = " ".join(words)
        call = lambda: _sentence_payload(*normalizer.normalize_text(text))

    profiler = cProfile.Profile()
    profiler.runcall(call)
    stats = pstats.Stats(profiler).stats
    normalizer_file = Path(module.__file__).resolve()

    def matching(name: str, *, line: int | None = None) -> list[tuple[int, float, float]]:
        found = []
        for (filename, function_line, function_name), value in stats.items():
            try:
                same_file = Path(filename).resolve() == normalizer_file
            except (OSError, ValueError):
                same_file = False
            if same_file and function_name == name and (line is None or function_line == line):
                _primitive_calls, _calls, own, cumulative, _callers = value
                found.append((function_line, own, cumulative))
        return found

    normalize_lines, first_line = inspect.getsourcelines(module.FilipinoNormalizer._normalize_lowercase)
    vocabulary_line = first_line + next(
        offset for offset, source_line in enumerate(normalize_lines)
        if source_line.strip() == "matches = ["
    )
    ranking_line = next(
        (
            first_line + offset
            for offset, source_line in enumerate(normalize_lines)
            if "lambda" in source_line and "candidate_and_distance" in source_line
        ),
        None,
    )

    def total(name: str, column: int, *, line: int | None = None) -> float:
        return sum(record[column] for record in matching(name, line=line))

    collect_own = total("_collect_candidates", 1)
    recursive_own = total("generate", 1)
    vocabulary = total("<listcomp>", 2, line=vocabulary_line)
    distance = total("damerau_levenshtein_distance", 2)
    rank_lambda = total("<lambda>", 1, line=ranking_line) if ranking_line is not None else 0.0
    builtin_ranking_own = (
        sum(
            value[2]
            for (filename, _line, function_name), value in stats.items()
            if filename in {"~", "<built-in>"} and function_name in {"min", "sorted"}
        )
        if ranking_line is not None
        else 0.0
    )
    tokenization_own = total("normalize_text", 1)
    tokenization_helpers = total("_protected_spans", 2) + total("_is_word_letter", 2)
    tokenization = tokenization_own + tokenization_helpers

    return {
        "scope": f"{kind} workload; {len(words)} tokens; import and artifact load excluded",
        "profiled_calls": 1,
        "seconds_by_stage": {
            "candidate_generation": collect_own + recursive_own,
            "vocabulary_filter": vocabulary,
            "damerau_levenshtein": distance,
            "ranking_overhead": rank_lambda + builtin_ranking_own,
            "tokenization": tokenization,
        },
        "raw_function_seconds": {
            "collect_candidates_own": collect_own,
            "recursive_generate_own": recursive_own,
            "vocabulary_list_comprehension_cumulative": vocabulary,
            "distance_cumulative": distance,
            "ranking_lambda_own": rank_lambda,
            "ranking_builtin_own": builtin_ranking_own,
            "normalize_text_own": tokenization_own,
            "protected_span_helper_cumulative": total("_protected_spans", 2),
            "word_letter_helper_cumulative": total("_is_word_letter", 2),
        },
    }


def _candidate_counts(normalizer: Any, words: list[str]) -> dict[str, int | float]:
    frequencies = Counter(words)
    input_tokens = len(words)
    words_requiring_generation = 0
    generated_total = 0
    survivor_total = 0
    max_generated = 0
    max_survivors = 0

    for word, frequency in frequencies.items():
        lowered = word.lower()
        if len(lowered) < 2 or len(lowered) > 128 or lowered in normalizer.vocabulary:
            continue
        candidates = normalizer._collect_candidates(
            lowered,
            normalizer.rules,
            normalizer.max_ngram,
            normalizer.candidate_cutoff,
        )
        survivors = [
            candidate
            for candidate in candidates
            if candidate.strip()
            and all(part.lower() in normalizer.vocabulary for part in candidate.strip().split())
        ]
        words_requiring_generation += frequency
        generated_total += len(candidates) * frequency
        survivor_total += len(survivors) * frequency
        max_generated = max(max_generated, len(candidates))
        max_survivors = max(max_survivors, len(survivors))

    denominator = words_requiring_generation
    return {
        "input_tokens": input_tokens,
        "words_requiring_generation": words_requiring_generation,
        "unique_words_requiring_generation": sum(
            1
            for word in frequencies
            if 2 <= len(word.lower()) <= 128 and word.lower() not in normalizer.vocabulary
        ),
        "generated_candidates_total": generated_total,
        "vocabulary_survivors_total": survivor_total,
        "mean_generated_candidates_per_generated_word": (
            generated_total / denominator if denominator else 0.0
        ),
        "mean_vocabulary_survivors_per_generated_word": (
            survivor_total / denominator if denominator else 0.0
        ),
        "max_generated_candidates_for_one_word": max_generated,
        "max_vocabulary_survivors_for_one_word": max_survivors,
    }


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_comparable_reports(baseline: dict[str, Any], current: dict[str, Any]) -> None:
    """Reject comparisons whose measurements do not share a valid basis."""
    if baseline.get("schema_version") != 2 or current.get("schema_version") != 2:
        raise ValueError("Runtime comparison requires schema_version 2 reports")

    equal_fields = (
        ("normalizer configuration", baseline.get("normalizer_configuration"), current.get("normalizer_configuration")),
        ("benchmark configuration", baseline.get("benchmark_configuration"), current.get("benchmark_configuration")),
    )
    for label, before, after in equal_fields:
        if before != after:
            raise ValueError(f"Runtime reports have incompatible {label}")
    normalizer_config = baseline.get("normalizer_configuration")
    if not isinstance(normalizer_config, dict) or not {
        "max_edit_distance", "candidate_cutoff", "max_ngram"
    }.issubset(normalizer_config):
        raise ValueError("Runtime reports are missing required normalizer configuration")
    benchmark_config = baseline.get("benchmark_configuration")
    if not isinstance(benchmark_config, dict) or not {
        "sizes", "repeats", "profile_words", "warmup_passes_per_workload"
    }.issubset(benchmark_config):
        raise ValueError("Runtime reports are missing required benchmark configuration")
    if (
        not isinstance(benchmark_config["sizes"], list)
        or not benchmark_config["sizes"]
        or any(not isinstance(size, int) or size < 1 for size in benchmark_config["sizes"])
        or len(set(benchmark_config["sizes"])) != len(benchmark_config["sizes"])
        or not isinstance(benchmark_config["repeats"], int)
        or benchmark_config["repeats"] < 1
        or not isinstance(benchmark_config["profile_words"], int)
        or benchmark_config["profile_words"] < 1
        or benchmark_config["warmup_passes_per_workload"] != 1
    ):
        raise ValueError("Runtime reports have invalid benchmark configuration")

    for report_name, report in (("baseline", baseline), ("current", current)):
        scope = report.get("measurement_scope", {})
        if (
            scope.get("normalizer_only") is not True
            or scope.get("gec_called") is not False
            or scope.get("reserved_test_accessed") is not False
            or scope.get("source") != "backend/normalization/data/train_pairs.csv source column only"
        ):
            raise ValueError(f"{report_name} report has an incompatible measurement scope")

    before_provenance = baseline.get("provenance", {})
    after_provenance = current.get("provenance", {})
    required_source_hashes = (
        before_provenance.get("normalizer_source_sha256"),
        after_provenance.get("normalizer_source_sha256"),
    )
    if not all(required_source_hashes):
        raise ValueError("Runtime reports must record each normalizer source SHA-256")
    provenance_fields = (
        "python",
        "platform",
        "training_pairs_sha256",
        "rules_sha256",
        "vocabulary_sha256",
        "metadata_sha256",
        "workload_word_list_sha256",
    )
    mismatched = [
        field
        for field in provenance_fields
        if not before_provenance.get(field)
        or not after_provenance.get(field)
        or before_provenance.get(field) != after_provenance.get(field)
    ]
    if mismatched:
        raise ValueError(f"Runtime reports have incompatible provenance fields: {mismatched}")

    for timing_field in ("first_call_definition", "warm_definition"):
        before_timing = baseline["measurement_scope"].get(timing_field)
        after_timing = current["measurement_scope"].get(timing_field)
        if not before_timing or before_timing != after_timing:
            raise ValueError(f"Runtime reports have incompatible {timing_field}")

    required_kinds = {"word", "sentence"}
    if set(baseline.get("workloads", {})) != required_kinds or set(current.get("workloads", {})) != required_kinds:
        raise ValueError("Runtime reports have incompatible workload kinds")
    expected_sizes = {str(size) for size in benchmark_config["sizes"]}
    for kind in sorted(required_kinds):
        before = baseline["workloads"][kind]
        after = current["workloads"][kind]
        if set(before) != expected_sizes or set(after) != expected_sizes:
            raise ValueError(f"Runtime reports have incompatible {kind} workload sizes")
        for size in expected_sizes:
            expected_repeats = benchmark_config["repeats"]
            for row in (before[size], after[size]):
                if row.get("warm_repetitions") != expected_repeats:
                    raise ValueError(f"Runtime reports have incompatible repeat counts for {kind}:{size}")
                if (
                    row.get("token_count") != int(size)
                    or len(row.get("warm_batch_ms", [])) != expected_repeats
                    or not isinstance(row.get("warm_median_ms"), (int, float))
                ):
                    raise ValueError(f"Runtime reports have incompatible workload sample data for {kind}:{size}")

    expected_fingerprints = {
        f"{kind}:{size}" for kind in required_kinds for size in expected_sizes
    }
    if (
        set(baseline.get("output_fingerprints", {})) != expected_fingerprints
        or set(current.get("output_fingerprints", {})) != expected_fingerprints
    ):
        raise ValueError("Runtime reports have incompatible output fingerprint workloads")


def _comparison_summary(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    _validate_comparable_reports(baseline, current)
    workloads: dict[str, Any] = {}
    for kind in ("word", "sentence"):
        workloads[kind] = {}
        for size in map(str, DEFAULT_SIZES):
            before = baseline["workloads"][kind][size]
            after = current["workloads"][kind][size]
            baseline_ms = before["warm_median_ms"]
            current_ms = after["warm_median_ms"]
            workloads[kind][size] = {
                "baseline_warm_median_ms": baseline_ms,
                "current_warm_median_ms": current_ms,
                "warm_change_percent": round((current_ms / baseline_ms - 1) * 100, 3)
                if baseline_ms
                else None,
                "candidate_counts_match": baseline["candidate_counts"][kind][size]
                == current["candidate_counts"][kind][size],
            }
    stages: dict[str, Any] = {}
    for kind in ("word", "sentence"):
        before = baseline["stage_profiles"][kind]["seconds_by_stage"]
        after = current["stage_profiles"][kind]["seconds_by_stage"]
        stages[kind] = {
            stage: {
                "baseline_seconds": before[stage],
                "current_seconds": after[stage],
                "change_percent": round((after[stage] / before[stage] - 1) * 100, 3)
                if before[stage]
                else None,
            }
            for stage in before
        }
    return {
        "baseline_report": str(current["_compared_report"]),
        "source_sha256": {
            "baseline": baseline["provenance"]["normalizer_source_sha256"],
            "current": current["provenance"]["normalizer_source_sha256"],
        },
        "git_head": {
            "baseline": baseline["provenance"].get("git_head"),
            "current": current["provenance"].get("git_head"),
        },
        "all_output_fingerprints_match": (
            baseline.get("output_fingerprints") == current["output_fingerprints"]
        ),
        "workloads": workloads,
        "profiled_stage_changes": stages,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(BACKEND_ROOT))
    import_started = time.perf_counter()
    from app.services import normalizer as module

    import_elapsed_ms = (time.perf_counter() - import_started) * 1_000
    load_started = time.perf_counter()
    normalizer = module.FilipinoNormalizer()
    artifact_load_elapsed_ms = (time.perf_counter() - load_started) * 1_000

    workload_words = _workload_words()
    report: dict[str, Any] = {
        "schema_version": 2,
        "phase": args.phase,
        "measurement_scope": {
            "normalizer_only": True,
            "gec_called": False,
            "reserved_test_accessed": False,
            "source": "backend/normalization/data/train_pairs.csv source column only",
            "import_and_artifact_load_excluded_from_workload_timings": True,
            "first_call_definition": (
                "one timed call before this workload's warmup; imports, artifact loading, "
                "and earlier workloads may already have warmed the process"
            ),
            "warm_definition": "one untimed full workload pass, followed by timed repetitions",
        },
        "setup": {
            "module_import_ms": round(import_elapsed_ms, 4),
            "artifact_load_ms": round(artifact_load_elapsed_ms, 4),
        },
        "normalizer_configuration": {
            "max_edit_distance": normalizer.max_edit_distance,
            "candidate_cutoff": normalizer.candidate_cutoff,
            "max_ngram": normalizer.max_ngram,
        },
        "benchmark_configuration": {
            "sizes": list(DEFAULT_SIZES),
            "repeats": args.repeats,
            "profile_words": args.profile_words,
            "warmup_passes_per_workload": 1,
        },
        "provenance": {
            "git_head": _git_head(),
            "python": sys.version,
            "platform": platform.platform(),
            "normalizer_source_sha256": _sha256_bytes(Path(module.__file__).read_bytes()),
            "training_pairs_sha256": _sha256_bytes(TRAIN_PAIRS.read_bytes()),
            "rules_sha256": _sha256_bytes(normalizer._resource_signature[0].read_bytes()),
            "vocabulary_sha256": _sha256_bytes(normalizer._resource_signature[1].read_bytes()),
            "metadata_sha256": _sha256_bytes(normalizer._resource_signature[2].read_bytes()),
            "workload_word_list_sha256": _json_digest(workload_words),
        },
        "workloads": {},
        "candidate_counts": {"word": {}, "sentence": {}},
        "stage_profiles": {},
        "output_fingerprints": {},
    }

    output_fingerprints: dict[str, str] = {}
    for kind in ("word", "sentence"):
        report["workloads"][kind] = {}
        report["candidate_counts"][kind] = {}
        for size in DEFAULT_SIZES:
            words = [workload_words[index % len(workload_words)] for index in range(size)]
            if kind == "word":
                call = lambda words=words: [normalizer.normalize_word(word) for word in words]
            else:
                sentence = " ".join(words)
                call = lambda sentence=sentence: _sentence_payload(*normalizer.normalize_text(sentence))

            _first_output, first_ms = _measure(call)
            warm_output, _warmup_ms = _measure(call)
            timed_runs: list[float] = []
            final_output = warm_output
            for _ in range(args.repeats):
                final_output, elapsed_ms = _measure(call)
                timed_runs.append(elapsed_ms)
            key = f"{kind}:{size}"
            output_fingerprints[key] = _json_digest(final_output)
            median_ms = statistics.median(timed_runs)
            report["workloads"][kind][str(size)] = {
                "token_count": size,
                "first_call_before_workload_warmup_ms": round(first_ms, 4),
                "warm_repetitions": len(timed_runs),
                "warm_batch_ms": [round(value, 4) for value in timed_runs],
                "warm_median_ms": round(median_ms, 4),
                "warm_p95_ms": round(_p95(timed_runs), 4),
                "warm_tokens_per_second": round(size / (median_ms / 1_000), 3)
                if median_ms
                else None,
            }
            report["candidate_counts"][kind][str(size)] = _candidate_counts(normalizer, words)

    profile_size = args.profile_words
    profile_words = [workload_words[index % len(workload_words)] for index in range(profile_size)]
    for kind in ("word", "sentence"):
        report["stage_profiles"][kind] = _profile_stages(module, normalizer, kind, profile_words)
    report["output_fingerprints"] = output_fingerprints

    if args.compare:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        report["_compared_report"] = str(args.compare)
        _validate_comparable_reports(baseline, report)
        baseline_fingerprints = baseline.get("output_fingerprints", {})
        if output_fingerprints != baseline_fingerprints:
            changed = sorted(
                key
                for key in set(output_fingerprints) | set(baseline_fingerprints)
                if output_fingerprints.get(key) != baseline_fingerprints.get(key)
            )
            raise AssertionError(f"Output parity failed for workloads: {changed}")
        report["parity"] = {
            "compared_report": str(args.compare),
            "all_output_fingerprints_match": True,
            "workloads_compared": len(output_fingerprints),
        }
        report["comparison"] = _comparison_summary(baseline, report)
        del report["_compared_report"]
    else:
        report["parity"] = None
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("baseline", "optimized"), default="baseline")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-words", type=int, default=1_000)
    args = parser.parse_args()
    if args.repeats < 1 or args.profile_words < 1:
        parser.error("--repeats and --profile-words must be positive")

    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Runtime profile written to {args.output}")
    if report["parity"]:
        print(f"Output parity passed for {report['parity']['workloads_compared']} workloads")


if __name__ == "__main__":
    main()
